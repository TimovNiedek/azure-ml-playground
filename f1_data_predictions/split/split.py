import pandas as pd
import typer
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated


def split_data(
    input_data_path: Annotated[str, typer.Argument(help="Path to the data")],
    output_train_path: Annotated[
        str, typer.Argument(help="Path to save the train data")
    ],
    output_test_path: Annotated[str, typer.Argument(help="Path to save the test data")],
    seed: Annotated[
        str, typer.Option(help="Seed used for random train / test split")
    ] = 42,
    test_size: Annotated[
        float, typer.Option(help="Fraction of the data to use as test set")
    ] = 0.2,
):
    df = pd.read_csv(input_data_path)
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["year"]
    )
    train.to_csv(output_train_path, index=False)
    test.to_csv(output_test_path, index=False)
    return train, test


if __name__ == "__main__":
    typer.run(split_data)
