from enum import Enum

import mlflow
import pandas as pd
import typer
from mlflow.models import infer_signature
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC, SVR
from typing_extensions import Annotated

CATEGORICAL_COLS = ["constructorId", "year", "driverId", "circuitId"]
NUMERICAL_COLS = ["year", "qualifyingPosition"]
FEATURE_COLS = ["constructorId", "year", "driverId", "circuitId", "qualifyingPosition"]
BOOL_COLS = ["pointsFinish", "wonRace", "podiumFinish"]
TARGET_COLS = ["positionOrder", "pointsFinish", "wonRace", "podiumFinish"]


class Task(Enum):
    classification = "classification"
    regression = "regression"


def get_features_labels(
    df: pd.DataFrame, target_col: str = "pointsFinish"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target_col in FEATURE_COLS:
        raise ValueError(
            f"target_col {target_col} is part of input features, cannot use as prediction target. Choose one of {TARGET_COLS}"
        )

    if target_col in BOOL_COLS:
        targets = df[target_col].astype(int)
    else:
        targets = df[target_col]

    return df[FEATURE_COLS], targets


def build_pipeline(
    task: Task = Task.classification, threshold_for_unknown_category: int = 3
) -> Pipeline:
    column_transformer = make_column_transformer(
        (
            OneHotEncoder(
                min_frequency=threshold_for_unknown_category,
                handle_unknown="infrequent_if_exist",
            ),
            CATEGORICAL_COLS,
        ),
        (MinMaxScaler(), NUMERICAL_COLS),
    )
    if task == Task.classification:
        pipeline = make_pipeline(column_transformer, SVC(gamma="auto"))
    else:
        pipeline = make_pipeline(column_transformer, SVR(gamma="auto"))

    return pipeline


def train(
    input_train_data_path: Annotated[
        str, typer.Argument(help="Path to the train data")
    ],
    output_model_path: Annotated[
        str, typer.Argument(help="Path to save the trained model")
    ],
    task: Annotated[
        Task, typer.Option(help="Task type", case_sensitive=False)
    ] = Task.classification,
    threshold_for_unknown_category: Annotated[
        int,
        typer.Option(
            help="Categories below this threshold are grouped as 'infrequent' category"
        ),
    ] = 3,
):
    mlflow.autolog(log_models=False)
    df = pd.read_csv(input_train_data_path)
    X_train, y_train = get_features_labels(df)
    pipeline = build_pipeline(task, threshold_for_unknown_category)
    pipeline.fit(X_train, y_train)
    train_score = pipeline.score(X_train, y_train)
    mlflow.log_metric("train_score", train_score)
    signature = infer_signature(X_train, pipeline.predict(X_train))
    mlflow.sklearn.save_model(pipeline, output_model_path, signature=signature)


if __name__ == "__main__":
    typer.run(train)
