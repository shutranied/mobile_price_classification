"""
Machine learning model module for the Mobile Price Classification project.

This module contains helper functions for training classification models,
comparing model performance, making a single-phone prediction, and saving
or loading trained model objects.
"""

import pickle
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def train_logistic_regression(X_train, y_train, max_iter: int = 1000):
    """
    Train a Logistic Regression model.

    Parameters:
        X_train: Training feature data.
        y_train: Training target data.
        max_iter (int): Maximum number of iterations for model convergence.

    Returns:
        LogisticRegression: Trained Logistic Regression model.
    """
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def train_svc_model(X_train, y_train):
    """
    Train a Support Vector Classifier model.

    Parameters:
        X_train: Training feature data.
        y_train: Training target data.

    Returns:
        SVC: Trained Support Vector Classifier model.
    """
    model = SVC()
    model.fit(X_train, y_train)
    return model


def train_random_forest_model(
    X_train,
    y_train,
    n_estimators: int = 100,
    random_state: int = 42,
):
    """
    Train a Random Forest Classifier model.

    Parameters:
        X_train: Training feature data.
        y_train: Training target data.
        n_estimators (int): Number of trees in the random forest.
        random_state (int): Random seed for reproducibility.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def compare_models(
    X_train,
    X_test,
    y_train,
    y_test,
    random_state: int = 42,
) -> Tuple[Any, str, Dict[str, float]]:
    """
    Train and compare multiple machine learning models.

    The best model is selected based on accuracy score.

    Parameters:
        X_train: Training feature data.
        X_test: Testing feature data.
        y_train: Training target data.
        y_test: Testing target data.
        random_state (int): Random seed used for reproducible models.

    Returns:
        tuple: A tuple containing:
            - best_model: The model with the highest accuracy.
            - best_model_name (str): Name of the best model.
            - results (dict): Accuracy score for each model.
    """
    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train),
        "Support Vector Classifier": train_svc_model(X_train, y_train),
        "Random Forest Classifier": train_random_forest_model(
            X_train,
            y_train,
            random_state=random_state,
        ),
    }

    results = {}
    best_model = None
    best_model_name = ""
    best_accuracy = 0.0

    print("\n=== Model Comparison ===")

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results[model_name] = accuracy
        print(f"{model_name}: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    print("\n=== Best Model Selected ===")
    print(f"{best_model_name}: {best_accuracy:.4f}")

    return best_model, best_model_name, results


def predict_single_phone(
    model,
    scaler,
    feature_columns: Sequence[str],
    sample_phone: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Predict the price range for one mobile phone.

    The same scaler used during training is applied to the new input before
    prediction.

    Parameters:
        model: Trained machine learning model.
        scaler: Fitted scaler used during training.
        feature_columns (Sequence[str]): Feature columns used by the model.
        sample_phone (dict, optional): Mobile phone specifications.

    Returns:
        str: Predicted price range label.
    """
    if sample_phone is None:
        sample_phone = {
            "battery_power": 1500,
            "blue": 1,
            "clock_speed": 2.2,
            "dual_sim": 1,
            "fc": 8,
            "four_g": 1,
            "int_memory": 64,
            "m_dep": 0.5,
            "mobile_wt": 150,
            "n_cores": 8,
            "pc": 12,
            "px_height": 1200,
            "px_width": 1920,
            "ram": 4096,
            "sc_h": 15,
            "sc_w": 7,
            "talk_time": 20,
            "three_g": 1,
            "touch_screen": 1,
            "wifi": 1,
        }

    missing_features = set(feature_columns) - set(sample_phone.keys())

    if missing_features:
        raise ValueError(f"Missing phone specification features: {missing_features}")

    phone_df = pd.DataFrame([sample_phone])
    phone_df = phone_df[list(feature_columns)]

    phone_scaled = scaler.transform(phone_df)
    prediction = model.predict(phone_scaled)

    price_mapping = {
        0: "Cheap / Low Cost",
        1: "Medium Cost",
        2: "Expensive / High Cost",
        3: "Very Expensive",
    }

    predicted_label = price_mapping.get(prediction[0], "Unknown")

    print("\n=== Single Phone Prediction ===")
    print("Predicted Class:", prediction[0])
    print("Predicted Price Range:", predicted_label)

    return predicted_label


def save_model(model: Any, file_path: str) -> None:
    """
    Save a trained model or model package as a pickle file.

    Parameters:
        model: Model object or dictionary package to save.
        file_path (str): Path where the pickle file will be saved.

    Returns:
        None
    """
    with open(file_path, "wb") as file:
        pickle.dump(model, file)


def load_model(file_path: str) -> Any:
    """
    Load a saved pickle model or model package.

    Parameters:
        file_path (str): Path to the saved pickle file.

    Returns:
        Any: Loaded model object or model package.
    """
    with open(file_path, "rb") as file:
        model = pickle.load(file)

    return model
