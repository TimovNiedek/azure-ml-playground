import os

from azure.ai.ml import MLClient
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


if __name__ == "__main__":
    create_compute("cpu-cluster")
    create_environment(
        "f1-predictions-env",
        description="F1 predictions training & inference environment",
        image=f"{AZURE_CONTAINER_REGISTRY_NAME}.azurecr.io/playground/f1-training-environment",
    )
