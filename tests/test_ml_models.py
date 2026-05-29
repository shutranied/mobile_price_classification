from pathlib import Path

from models.ml_models import (
    compare_models,
    load_model,
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
    model_path = Path(tmp_path) / "test_model.pkl"

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    assert model_path.exists()
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict")
