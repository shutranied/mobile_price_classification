import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using accuracy and classification report.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:")
    print(f"{accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy, y_pred


def plot_confusion_matrix(model, X_test, y_test, file_path="outputs/confusion_matrix.png"):
    """
    Save the confusion matrix for the trained model.
    """
    os.makedirs("outputs", exist_ok=True)

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

    print(f"Confusion matrix saved to {file_path}")


def save_evaluation_report(model, X_test, y_test, model_name, file_path="outputs/model_report.txt"):
    """
    Save model evaluation result into a text file.
    """
    os.makedirs("outputs", exist_ok=True)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    with open(file_path, "w") as file:
        file.write("=== Mobile Price Classification Model Report ===\n\n")
        file.write(f"Best Model: {model_name}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write("Classification Report:\n")
        file.write(report)

    print(f"Evaluation report saved to {file_path}")