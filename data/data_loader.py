"""
Data loading module for the Mobile Price Classification project.

This module contains functions used to load dataset files into pandas DataFrames.
"""

from pathlib import Path

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.

    This function reads a CSV file and returns the dataset as a pandas DataFrame.
    It also checks whether the file exists, whether the file is a CSV file,
    and whether the loaded dataset is empty.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file path does not exist.
        ValueError: If the file is not a CSV file or if the dataset is empty.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Invalid file type. Expected a CSV file: {file_path}")

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(f"The dataset is empty: {file_path}")

    return data
