from enum import Enum

import mlflow
import pandas as pd
import typer
from mlflow.models import infer_signature
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    task: Task = Task.classification,
    threshold_for_unknown_category: int = 3,
    kernel: str = "rbf",
    reg_rate: float = 1.0,
) -> Pipeline:
    column_transformer = make_column_transformer(
        (
            OneHotEncoder(
                min_frequency=threshold_for_unknown_category,
                handle_unknown="infrequent_if_exist",
            ),
            CATEGORICAL_COLS,
        ),
        (StandardScaler(), NUMERICAL_COLS),
    )
    if task == Task.classification:
        pipeline = make_pipeline(
            column_transformer, SVC(gamma="auto", C=1 / reg_rate, kernel=kernel)
        )
    else:
        pipeline = make_pipeline(
            column_transformer, SVR(gamma="auto", C=1 / reg_rate, kernel=kernel)
        )

    return pipeline


def train(
    input_train_data_path: Annotated[
        str, typer.Argument(help="Path to the train data")
    ],
    output_model_path: Annotated[
        str, typer.Argument(help="Path to save the trained model")
    ],
    target_column: Annotated[
        str,
        typer.Option(
            help=f"Column to predict, choose from {TARGET_COLS}", case_sensitive=False
        ),
    ] = "pointsFinish",
    task: Annotated[
        Task, typer.Option(help="Task type", case_sensitive=False)
    ] = Task.classification,
    threshold_for_unknown_category: Annotated[
        int,
        typer.Option(
            help="Categories below this threshold are grouped as 'infrequent' category"
        ),
    ] = 3,
    svm_kernel: Annotated[
        str, typer.Option(help="SVM kernel type", case_sensitive=False)
    ] = "rbf",
    svm_reg_rate: Annotated[
        float, typer.Option(help="Regularization rate for SVM")
    ] = 1.0,
):
    mlflow.start_run()
    mlflow.sklearn.autolog(log_models=False)

    df = pd.read_csv(input_train_data_path)
    X_train, y_train = get_features_labels(df, target_column)
    pipeline = build_pipeline(
        task, threshold_for_unknown_category, svm_kernel, svm_reg_rate
    )
    pipeline.fit(X_train, y_train)
    train_score = pipeline.score(X_train, y_train)
    mlflow.log_metric("train_score", train_score)
    signature = infer_signature(X_train, pipeline.predict(X_train))
    mlflow.sklearn.log_model(pipeline, "f1_model", signature=signature)
    mlflow.sklearn.save_model(pipeline, output_model_path)
    mlflow.end_run()


if __name__ == "__main__":
    typer.run(train)
