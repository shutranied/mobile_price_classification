"""
Visualization module for the Mobile Price Classification project.

This module contains functions for creating and saving exploratory data
visualizations. The generated images are saved inside the outputs folder.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


OUTPUT_DIR = Path("outputs")


def create_output_folder() -> None:
    """
    Create the output folder used to save visualization images.

    Returns:
        None
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _check_columns_exist(data: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Check whether required columns exist in the dataset.

    Parameters:
        data (pandas.DataFrame): Dataset to be checked.
        required_columns (list): List of column names required for plotting.

    Raises:
        ValueError: If one or more required columns are missing.

    Returns:
        None
    """
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns for visualization: {missing_columns}")


def plot_class_balance(data: pd.DataFrame, target_column: str = "price_range") -> None:
    """
    Plot and save the relative frequency of each price range class.

    Parameters:
        data (pandas.DataFrame): Dataset containing the target column.
        target_column (str): Name of the target column.

    Returns:
        None
    """
    _check_columns_exist(data, [target_column])
    create_output_folder()

    class_frequency = data[target_column].value_counts(normalize=True).sort_index()

    plt.figure(figsize=(8, 5))
    class_frequency.plot(kind="bar")
    plt.xlabel("Price Range Classes")
    plt.ylabel("Relative Frequency")
    plt.title("Class Balance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_balance.png")
    plt.close()


def plot_ram_distribution(
    data: pd.DataFrame,
    target_column: str = "price_range",
) -> None:
    """
    Plot and save the distribution of RAM based on price range classes.

    Parameters:
        data (pandas.DataFrame): Dataset containing RAM and target columns.
        target_column (str): Name of the target column.

    Returns:
        None
    """
    _check_columns_exist(data, [target_column, "ram"])
    create_output_folder()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=target_column, y="ram", data=data)
    plt.xlabel("Price Range Classes")
    plt.ylabel("RAM")
    plt.title("Distribution of RAM by Price Range Class")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ram_distribution.png")
    plt.close()


def plot_battery_power_distribution(
    data: pd.DataFrame,
    target_column: str = "price_range",
) -> None:
    """
    Plot and save the distribution of battery power based on price range classes.

    Parameters:
        data (pandas.DataFrame): Dataset containing battery power and target columns.
        target_column (str): Name of the target column.

    Returns:
        None
    """
    _check_columns_exist(data, [target_column, "battery_power"])
    create_output_folder()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=target_column, y="battery_power", data=data)
    plt.xlabel("Price Range Classes")
    plt.ylabel("Battery Power")
    plt.title("Distribution of Battery Power by Price Range Class")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "battery_power_distribution.png")
    plt.close()


def plot_three_g_support(data: pd.DataFrame) -> None:
    """
    Plot and save the percentage of phones that support 3G.

    Parameters:
        data (pandas.DataFrame): Dataset containing the three_g column.

    Returns:
        None
    """
    _check_columns_exist(data, ["three_g"])
    create_output_folder()

    labels = ["Not Supported", "3G Supported"]
    values = data["three_g"].value_counts().reindex([0, 1], fill_value=0).values

    plt.figure(figsize=(6, 6))
    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    plt.title("3G Supported Phones")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "three_g_support.png")
    plt.close()
