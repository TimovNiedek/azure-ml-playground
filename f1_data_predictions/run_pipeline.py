from pathlib import Path

from azure.ai.ml import Input, load_component
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from utils import get_ml_client

ml_client = get_ml_client()


def build_pipeline():
    d = Path(__file__).resolve().parent.parent / "f1_data_predictions"

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


if __name__ == "__main__":
    pipeline_job = build_pipeline()
    ml_client.jobs.create_or_update(pipeline_job)
