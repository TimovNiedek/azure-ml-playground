import pandas as pd
from utils import get_ml_client
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from dotenv import load_dotenv

load_dotenv()

ml_client = get_ml_client()


def join_data() -> pd.DataFrame:
    races_data = pd.read_csv("data/races.csv")
    quali_data = pd.read_csv("data/qualifying.csv")
    results_data = pd.read_csv("data/results.csv")

    print(results_data.head())

    # Add a boolean column whether the driver achieved a points finish
    results_data["pointsFinish"] = results_data["points"] > 0

    # Add a boolean column whether the driver won the race
    results_data["wonRace"] = results_data["positionOrder"] == 1

    # Add a boolean column whether the driver achieved a podium finish
    results_data["podiumFinish"] = results_data["positionOrder"].isin([1, 2, 3])

    quali_races = (
        quali_data[["raceId", "qualifyId", "driverId", "constructorId", "position"]]
        .merge(races_data[["raceId", "year", "circuitId"]], on="raceId", how="left")
        .rename(columns={"position": "qualifyingPosition"})
    )

    print(quali_races.head())

    quali_race_results = quali_races.merge(
        results_data[
            [
                "raceId",
                "driverId",
                "constructorId",
                "positionOrder",
                "pointsFinish",
                "wonRace",
                "podiumFinish",
            ]
        ],
        on=["raceId", "driverId", "constructorId"],
        how="left",
    )

    print("=" * 6 + "Combined data" + "=" * 6)
    print(quali_race_results.head())
    assert len(quali_race_results) == len(quali_data)

    return quali_race_results


def upload_data(path: str):
    my_data = Data(
        path=path,
        type=AssetTypes.URI_FILE,
        description="Data on Formula 1 races, qualifying and results",
        name="f1-data",
    )

    ml_client.data.create_or_update(my_data)


def prepare_data():
    df = join_data()
    data_path = "data/f1-data.csv"
    df.to_csv(data_path, index=False)
    upload_data(data_path)


if __name__ == "__main__":
    prepare_data()
