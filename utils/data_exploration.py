"""
Data exploration module for the Mobile Price Classification project.

This module contains helper functions for displaying basic information about
the dataset before preprocessing and model training.
"""

import pandas as pd


def display_basic_info(data: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.

    This function prints dataset structure, descriptive statistics, number of
    unique values, missing values, and the first five rows of the dataset.

    Parameters:
        data (pandas.DataFrame): Dataset to be explored.

    Raises:
        TypeError: If the input is not a pandas DataFrame.

    Returns:
        None
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    print("Dataset Information:")
    data.info()

    print("\nDataset Description:")
    print(data.describe())

    print("\nNumber of Unique Values:")
    print(data.nunique())

    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\nFirst 5 Rows:")
    print(data.head())
