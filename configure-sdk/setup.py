import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, load_component
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import AmlCompute, Compute, Environment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

ml_client = MLClient(
    DefaultAzureCredential(),
    os.getenv("AZURE_SUBSCRIPTION_ID"),
    os.getenv("AZURE_RESOURCE_GROUP"),
    os.getenv("AZURE_ML_WORKSPACE"),
)

AZURE_CONTAINER_REGISTRY_NAME = os.environ["AZURE_CONTAINER_REGISTRY_NAME"]


def create_compute(
    name: str, min_instances: int = 0, max_instances: int = 2
) -> Compute:
    try:
        cpu_cluster = ml_client.compute.get(name)
        print(f"You already have a cluster named {name}, we'll reuse it as is.")
    except Exception:
        print("Creating a new cpu compute target...")

        # Let's create the Azure ML compute object with the intended parameters
        cpu_cluster = AmlCompute(
            name=name,
            # Azure ML Compute is the on-demand VM service
            type="amlcompute",
            # VM Family
            size="STANDARD_DS11_V2",  # 2 cores, 14 GB RAM
            # size = "STANDARD_DS3_v2",  # 4 cores, 14 GB RAM
            # size = "STANDARD_DS12_v2",  # 4 cores, 28 GB RAM
            # size = "STANDARD_DS13_v2",  # 8 cores, 56 GB RAM
            # Minimum running nodes when there is no job running
            min_instances=min_instances,
            # Nodes in cluster
            max_instances=max_instances,
            # How many seconds will the node running after the job termination
            idle_time_before_scale_down=120,
            # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
            tier="Dedicated",
        )

        # Now, we pass the object to MLClient's create_or_update method
        cpu_cluster: Compute = ml_client.compute.begin_create_or_update(
            cpu_cluster
        ).result(timeout=120)
    return cpu_cluster


def create_environment(
    name: str, description: str, image: str, conda_file: str | None = None
) -> Environment:
    env_docker_image = Environment(
        image=image,
        conda_file=conda_file,
        name=name,
        description=description,
    )
    return ml_client.environments.create_or_update(env_docker_image)


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
    create_compute("cpu-cluster")
    create_environment(
        "f1-predictions-env",
        description="F1 predictions training & inference environment",
        image=f"{AZURE_CONTAINER_REGISTRY_NAME}.azurecr.io/playground/f1-training-environment",
    )
    pipeline_job = build_pipeline()
    ml_client.jobs.create_or_update(pipeline_job)
