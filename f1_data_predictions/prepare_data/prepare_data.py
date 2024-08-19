import pandas as pd
import typer
from typing_extensions import Annotated


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def prepare_data(
    raw_data: Annotated[str, typer.Argument(help="Path to the raw data (csv)")],
    output_data: Annotated[str, typer.Argument(help="Path to the output data")],
):
    print(f"Preparing data from {raw_data} to {output_data}")
    df = pd.read_csv(raw_data)
    print(f"Data loaded: {df.shape[0]} rows")
    df = handle_missing_values(df)
    print(f"Data cleaned: {df.shape[0]} rows")
    df.to_csv(output_data, index=False)
    print("Data prepared")


if __name__ == "__main__":
    typer.run(prepare_data)
