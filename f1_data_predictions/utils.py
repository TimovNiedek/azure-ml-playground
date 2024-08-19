import os

import dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

dotenv.load_dotenv()


CATEGORICAL_COLS = ["constructorId", "year", "driverId", "circuitId"]
NUMERICAL_COLS = ["year", "qualifyingPosition"]
FEATURE_COLS = ["constructorId", "year", "driverId", "circuitId", "qualifyingPosition"]
BOOL_COLS = ["pointsFinish", "wonRace", "podiumFinish"]
TARGET_COLS = ["positionOrder", "pointsFinish", "wonRace", "podiumFinish"]


def get_ml_client():
    ml_client = MLClient(
        DefaultAzureCredential(),
        os.getenv("AZURE_SUBSCRIPTION_ID"),
        os.getenv("AZURE_RESOURCE_GROUP"),
        os.getenv("AZURE_ML_WORKSPACE"),
    )
    return ml_client
