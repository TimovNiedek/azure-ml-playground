from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

import os


def get_ml_client():
    ml_client = MLClient(
        DefaultAzureCredential(),
        os.getenv("AZURE_SUBSCRIPTION_ID"),
        os.getenv("AZURE_RESOURCE_GROUP"),
        os.getenv("AZURE_ML_WORKSPACE"),
    )
    return ml_client
