"""
Evaluation module for the Mobile Price Classification project.

This module contains helper functions for evaluating trained machine learning
models, saving confusion matrix images, and saving classification reports.
"""

from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)


def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Tuple[float, Any]:
    """
    Evaluate a trained model using accuracy score and classification report.

    Parameters:
        model: Trained machine learning model.
        X_test: Testing feature data.
        y_test: Testing target data.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): Model accuracy score.
            - y_pred: Predicted target values.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:")
    print(f"{accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return accuracy, y_pred


def plot_confusion_matrix(
    model: Any,
    X_test: Any,
    y_test: Any,
    file_path: str = "outputs/confusion_matrix.png",
) -> None:
    """
    Save the confusion matrix for a trained model.

    Parameters:
        model: Trained machine learning model.
        X_test: Testing feature data.
        y_test: Testing target data.
        file_path (str): Path where the confusion matrix image will be saved.

    Returns:
        None
    """
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Confusion matrix saved to {output_path}")


def save_evaluation_report(
    model: Any,
    X_test: Any,
    y_test: Any,
    model_name: str,
    file_path: str = "outputs/model_report.txt",
) -> None:
    """
    Save model evaluation results into a text file.

    Parameters:
        model: Trained machine learning model.
        X_test: Testing feature data.
        y_test: Testing target data.
        model_name (str): Name of the evaluated model.
        file_path (str): Path where the report text file will be saved.

    Returns:
        None
    """
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("=== Mobile Price Classification Model Report ===\n\n")
        file.write(f"Best Model: {model_name}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write("Classification Report:\n")
        file.write(report)

    print(f"Evaluation report saved to {output_path}")