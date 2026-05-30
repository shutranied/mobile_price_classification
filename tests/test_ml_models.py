import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.ml_models import (
    compare_models,
    load_model,
    predict_single_phone,
    save_model,
    train_logistic_regression,
    train_random_forest_model,
    train_svc_model,
)


def get_sample_data():
    """
    Create a small sample dataset for model testing.
    """
    X_train = [
        [500, 512],
        [800, 1024],
        [1500, 3072],
        [1800, 4096],
    ]

    y_train = [0, 0, 1, 1]

    X_test = [
        [600, 768],
        [1700, 3500],
    ]

    y_test = [0, 1]

    return X_train, X_test, y_train, y_test


def test_train_logistic_regression():
    """
    Test that Logistic Regression can be trained and used for prediction.
    """
    X_train, _, y_train, _ = get_sample_data()

    model = train_logistic_regression(X_train, y_train)
    predictions = model.predict(X_train)

    assert len(predictions) == len(y_train)


def test_train_svc_model():
    """
    Test that Support Vector Classifier can be trained and used for prediction.
    """
    X_train, _, y_train, _ = get_sample_data()

    model = train_svc_model(X_train, y_train)
    predictions = model.predict(X_train)

    assert len(predictions) == len(y_train)


def test_train_random_forest_model():
    """
    Test that Random Forest Classifier can be trained and used for prediction.
    """
    X_train, _, y_train, _ = get_sample_data()

    model = train_random_forest_model(X_train, y_train)
    predictions = model.predict(X_train)

    assert len(predictions) == len(y_train)


def test_compare_models_returns_best_model_and_results():
    """
    Test that compare_models() returns a best model, model name, and results.
    """
    X_train, X_test, y_train, y_test = get_sample_data()

    best_model, best_model_name, results = compare_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    assert best_model is not None
    assert best_model_name in results
    assert isinstance(results, dict)
    assert "Logistic Regression" in results
    assert "Support Vector Classifier" in results
    assert "Random Forest Classifier" in results


def test_save_and_load_model(tmp_path):
    """
    Test that a trained model can be saved and loaded successfully.
    """
    X_train, _, y_train, _ = get_sample_data()

    model = train_random_forest_model(X_train, y_train)
    model_path = tmp_path / "test_model.pkl"

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    assert model_path.exists()
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict")


def test_predict_single_phone_returns_label():
    """
    Test that predict_single_phone() returns a valid price range label.
    """
    X_train = pd.DataFrame(
        {
            "battery_power": [500, 800, 1500, 1800],
            "ram": [512, 1024, 3072, 4096],
        }
    )
    y_train = [0, 0, 1, 1]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = train_random_forest_model(X_train_scaled, y_train)

    sample_phone = {
        "battery_power": 1500,
        "ram": 3072,
    }

    predicted_label = predict_single_phone(
        model=model,
        scaler=scaler,
        feature_columns=["battery_power", "ram"],
        sample_phone=sample_phone,
    )

    assert isinstance(predicted_label, str)
    assert predicted_label in [
        "Cheap / Low Cost",
        "Medium Cost",
        "Expensive / High Cost",
        "Very Expensive",
    ]
