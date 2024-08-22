from pathlib import Path

import pandas as pd
import typer
from azure.ai.ml import Input, load_component
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import (
    CronTrigger,
    Data,
    JobSchedule,
    Model,
    RecurrenceTrigger,
)
from azure.ai.ml.sweep import BanditPolicy, Choice, LogUniform, RandomSamplingAlgorithm
from utils import get_ml_client

from f1_data_predictions.rai.rai import run_rai

ml_client = get_ml_client()

app = typer.Typer(pretty_exceptions_enable=False)


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


@app.command()
def schedule(
    frequency: str | None = None, interval: int | None = None, cron: str | None = None
):
    if cron is not None:
        recurrence = CronTrigger(expression=cron)
    elif frequency is not None and interval is not None:
        recurrence = RecurrenceTrigger(frequency=frequency, interval=interval)
    else:
        raise ValueError("Either cron or frequency and interval must be provided")
    pipeline_job = build_pipeline()
    job_schedule = JobSchedule(
        name="f1-training-pipeline-schedule",
        create_job=pipeline_job,
        trigger=recurrence,
    )
    job_schedule: JobSchedule = ml_client.schedules.begin_create_or_update(
        schedule=job_schedule
    ).result()
    print(
        f"Job schedule created with id: {job_schedule.id} and name: {job_schedule.name}"
    )


@app.command()
def remove_schedule():
    ml_client.schedules.begin_disable(name="f1-training-pipeline-schedule").result()
    ml_client.schedules.begin_delete(name="f1-training-pipeline-schedule").result()
    print("Job schedule deleted")


@app.command()
def register_model(job_id: str):
    model = Model(
        path=f"azureml://jobs/{job_id}/outputs/artifacts/paths/f1_model/",
        type=AssetTypes.MLFLOW_MODEL,
        name="f1-model",
        description="Model for predicting outcome of F1 races.",
    )

    registered_model = ml_client.models.create_or_update(model)
    print(registered_model)


def make_parquet_asset(
    df: pd.DataFrame, local_path: str, name: str, version: str
) -> str:
    df.to_parquet(Path(local_path) / f"{name}.parquet", engine="pyarrow")

    data = Data(
        path=local_path,
        type=AssetTypes.MLTABLE,
        description=f"F1 {name} data",
        name=name + "_parquet",
        version=version,
    )
    ml_client.data.create_or_update(data)
    return data.name


@app.command()
def prepare_rai(
    train_data_dir: str, test_data_dir: str, target_column_name: str = "pointsFinish"
):
    train_data = ml_client.data.get(
        name="train_data",
        version=2,
    )
    test_data = ml_client.data.get(
        name="test_data",
        version=2,
    )

    drop_target_cols = ["positionOrder", "pointsFinish", "wonRace", "podiumFinish"]
    drop_target_cols.remove(target_column_name)
    train_data = pd.read_csv(train_data.path)
    train_data = train_data.drop(columns=drop_target_cols)
    test_data = pd.read_csv(test_data.path)
    test_data = test_data.drop(columns=drop_target_cols)

    try:
        make_parquet_asset(train_data, train_data_dir, "train_data", "2")
        make_parquet_asset(test_data, test_data_dir, "test_data", "2")
    except Exception:
        print("RAI data already exists")

    run_rai(
        "azureml:train_data_parquet:2",
        "azureml:test_data_parquet:2",
    )


if __name__ == "__main__":
    app()
