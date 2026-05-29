"""
Preprocessing module for the Mobile Price Classification project.

This module contains functions for separating features and target values,
splitting the dataset into training and testing sets, and scaling feature
values using MinMaxScaler.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def split_features_target(
    data: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate the dataset into input features and target variable.

    Parameters:
        data (pandas.DataFrame): The full dataset.
        target_column (str): The name of the target column.

    Returns:
        tuple: A tuple containing:
            - X (pandas.DataFrame): Feature columns.
            - y (pandas.Series): Target column.

    Raises:
        ValueError: If the target column is not found in the dataset.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the dataset into training and testing sets.

    Stratification is used to keep the target class distribution balanced
    in both the training and testing sets.

    Parameters:
        X (pandas.DataFrame): Feature data.
        y (pandas.Series): Target data.
        test_size (float): Proportion of the dataset used for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def scale_train_test_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Scale training and testing feature values using MinMaxScaler.

    The scaler is fitted only on the training data, then applied to both
    training and testing data. This prevents data leakage from the test set.

    Parameters:
        X_train (pandas.DataFrame): Training feature data.
        X_test (pandas.DataFrame): Testing feature data.

    Returns:
        tuple: A tuple containing:
            - X_train_scaled: Scaled training features.
            - X_test_scaled: Scaled testing features.
            - scaler: Fitted MinMaxScaler object.
    """
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
