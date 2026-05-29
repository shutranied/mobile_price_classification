import pandas as pd

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
