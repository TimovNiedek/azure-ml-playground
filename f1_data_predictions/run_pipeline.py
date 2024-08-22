from pathlib import Path

import typer
from azure.ai.ml import Input, load_component
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.sweep import BanditPolicy, Choice, LogUniform, RandomSamplingAlgorithm
from utils import get_ml_client

ml_client = get_ml_client()

app = typer.Typer()


def build_pipeline():
    d = Path(__file__).parent

    # Load the component
    prepare_data = load_component(source=d / "prepare_data.yml")
    split_data = load_component(source=d / "split.yml")
    train = load_component(source=d / "train.yml")

    # Define the pipeline
    @pipeline(
        name="f1-training-pipeline",
        description="F1 training pipeline",
        experiment_name="f1_predictions_dev",
    )
    def f1_training_pipeline(pipeline_job_input):
        prepare_data_component = prepare_data(raw_data=pipeline_job_input)
        split_data_component = split_data(
            input_data=prepare_data_component.outputs.cleaned_data,
            seed=42,
            test_size=0.2,
        )
        train_component = train(
            train_data=split_data_component.outputs.train_data,
            task="classification",
            threshold_for_unknown_category=3,
        )
        return {
            "train_data": split_data_component.outputs.train_data,
            "test_data": split_data_component.outputs.test_data,
            "model": train_component.outputs.model,
        }

    pipeline_job = f1_training_pipeline(
        Input(type=AssetTypes.URI_FILE, path="azureml:f1-data:4")
    )
    pipeline_job.settings.default_compute = "cpu-cluster"
    pipeline_job.settings.default_datastore = "ml_datastore"
    print(pipeline_job)

    return pipeline_job


@app.command()
def train():
    pipeline_job = build_pipeline()
    ml_client.jobs.create_or_update(pipeline_job)


@app.command()
def sweep():
    d = Path(__file__).resolve().parent
    print(d)

    # Load the component
    train = load_component(source=d / "train.yml")

    command_job_for_sweep = train(
        train_data=Input(type=AssetTypes.URI_FILE, path="azureml:train_data:2"),
        task="classification",
        threshold_for_unknown_category=3,
        svm_kernel=Choice(values=["linear", "rbf", "sigmoid"]),
        svm_reg_rate=LogUniform(1e-5, 10),
    )

    sweep_job = command_job_for_sweep.sweep(
        compute="cpu-cluster",
        sampling_algorithm=RandomSamplingAlgorithm(
            seed=42, rule="sobol"
        ),  # "grid", "random" or "bayesian"
        primary_metric="training_accuracy_score",  # must be logged in mlflow
        goal="maximize",  # "minimize" or "maximize"
        early_termination_policy=BanditPolicy(
            delay_evaluation=5, evaluation_interval=1, slack_amount=0.2
        ),  # "bandit", "median_stopping", or "truncation_selection"
        max_total_trials=10,
        timeout=7200,  # in seconds, timeout after which the job will be cancelled
        max_concurrent_trials=2,
    )
    # can also use set_limits to set the max_total_trials, max_concurrent_trials, and timeout
    sweep_job.experiment_name = "sweep-f1-predictions"

    returned_sweep_job = ml_client.create_or_update(sweep_job)
    aml_url = returned_sweep_job.studio_url
    print("Monitor your job at", aml_url)

    return returned_sweep_job


if __name__ == "__main__":
    app()
