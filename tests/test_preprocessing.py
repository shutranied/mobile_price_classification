import pandas as pd

from preprocessing.preprocessing import (
    scale_train_test_features,
    split_features_target,
    split_train_test,
)


def test_split_features_target():
    """
    Test that split_features_target() separates features and target correctly.
    """
    data = pd.DataFrame(
        {
            "battery_power": [1500, 1200, 1800, 900],
            "ram": [4096, 2048, 3072, 1024],
            "price_range": [3, 2, 3, 1],
        }
    )

    X, y = split_features_target(data, "price_range")

    assert "price_range" not in X.columns
    assert list(X.columns) == ["battery_power", "ram"]
    assert y.tolist() == [3, 2, 3, 1]
    assert X.shape == (4, 2)


def test_split_train_test():
    """
    Test that split_train_test() correctly splits the dataset.
    """
    X = pd.DataFrame(
        {
            "battery_power": [500, 600, 700, 800, 1500, 1600, 1700, 1800],
            "ram": [512, 768, 1024, 1200, 3072, 3200, 3500, 4096],
        }
    )

    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    assert len(X_train) == 6
    assert len(X_test) == 2
    assert len(y_train) == 6
    assert len(y_test) == 2


def test_scale_train_test_features():
    """
    Test that scale_train_test_features() scales the training and testing data.
    """
    X_train = pd.DataFrame(
        {
            "battery_power": [500, 1000, 1500],
            "ram": [512, 2048, 4096],
        }
    )

    X_test = pd.DataFrame(
        {
            "battery_power": [750, 1250],
            "ram": [1024, 3072],
        }
    )

    X_train_scaled, X_test_scaled, scaler = scale_train_test_features(
        X_train,
        X_test,
    )

    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert scaler is not None

    assert X_train_scaled.min() >= 0
    assert X_train_scaled.max() <= 1
