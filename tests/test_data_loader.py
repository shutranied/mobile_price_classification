import pandas as pd
import pytest

from data.data_loader import load_data


def test_load_data_returns_dataframe(tmp_path):
    """
    Test that load_data() correctly loads a CSV file
    and returns a pandas DataFrame.
    """
    test_file = tmp_path / "sample_data.csv"

    test_file.write_text(
        "battery_power,ram,price_range\n"
        "1500,4096,3\n"
        "1200,2048,2\n"
    )

    data = load_data(test_file)

    assert isinstance(data, pd.DataFrame)
    assert data.shape == (2, 3)
    assert list(data.columns) == ["battery_power", "ram", "price_range"]
    assert data["price_range"].tolist() == [3, 2]


def test_load_data_file_not_found():
    """
    Test that load_data() raises FileNotFoundError
    when the file does not exist.
    """
    with pytest.raises(FileNotFoundError):
        load_data("data/non_existing_file.csv")


def test_load_data_invalid_file_type(tmp_path):
    """
    Test that load_data() raises ValueError
    when the file is not a CSV file.
    """
    test_file = tmp_path / "sample_data.txt"
    test_file.write_text("This is not a CSV file.")

    with pytest.raises(ValueError):
        load_data(test_file)


def test_load_data_empty_csv(tmp_path):
    """
    Test that load_data() raises ValueError
    when the CSV file is empty.
    """
    test_file = tmp_path / "empty.csv"
    test_file.write_text("")

    with pytest.raises(ValueError):
        load_data(test_file)
