from pathlib import Path

import pandas as pd

from domain.exceptions import DatasetLoadError, DatasetNotFoundError, EmptyDatasetError

def load_dataset(
    path: Path, *, encoding: str = "utf-8", separator: str = ","
) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.is_file():
        raise DatasetNotFoundError(f"Dataset file was not found: {source_path}")

    try:
        dataset = pd.read_csv(source_path, encoding=encoding, sep=separator)
    except pd.errors.EmptyDataError as error:
        raise EmptyDatasetError(f"Dataset could not be read: {source_path}") from error

    return dataset
