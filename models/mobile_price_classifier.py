"""
Object-Oriented Programming module for the Mobile Price Classification project.

This module contains the MobilePriceClassifier class, which manages the full
machine learning workflow: data preparation, model training, model evaluation,
prediction, confusion matrix saving, evaluation report saving, and model saving.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


class MobilePriceClassifier:
    """
    Class for training and evaluating mobile price classification models.

    This class handles the complete machine learning workflow for predicting
    the price range category of a mobile phone based on its technical
    specifications.

    Attributes:
        test_size (float): Proportion of data used for testing.
        random_state (int): Random seed for reproducible results.
        model_path (str): File path for saving the best model package.
        scaler (MinMaxScaler): Scaler used to normalize feature values.
        models (dict): Dictionary of machine learning models.
        results (dict): Dictionary containing model accuracy scores.
        best_model: Best trained model based on accuracy.
        best_model_name (str): Name of the best model.
        best_accuracy (float): Accuracy score of the best model.
        feature_columns (list): List of feature column names used for training.
        price_labels (dict): Mapping of class values to price range labels.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        model_path: str = "models/mobile_price_model.pkl",
    ) -> None:
        """
        Initialize the MobilePriceClassifier object.

        Parameters:
            test_size (float): Proportion of the dataset used for testing.
            random_state (int): Random seed for reproducibility.
            model_path (str): Path where the trained model package is saved.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model_path = model_path

        self.scaler = MinMaxScaler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0.0
        self.feature_columns = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

        self.price_labels = {
            0: "Cheap / Low Cost",
            1: "Medium Cost",
            2: "Expensive / High Cost",
            3: "Very Expensive",
        }

    def prepare_data(self, data: pd.DataFrame, target_column: str) -> None:
        """
        Prepare the dataset for machine learning.

        This method separates the dataset into features and target values,
        splits the data into training and testing sets, and applies MinMaxScaler.
        The scaler is fitted only on the training data to avoid data leakage.

        Parameters:
            data (pandas.DataFrame): Full training dataset.
            target_column (str): Name of the target column.

        Raises:
            ValueError: If the target column does not exist in the dataset.

        Returns:
            None
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        self.feature_columns = list(X.columns)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("\nData prepared successfully.")
        print("Training data shape:", self.X_train.shape)
        print("Testing data shape:", self.X_test.shape)

    def train_models(self) -> None:
        """
        Train multiple machine learning classification models.

        The trained models are stored in the self.models dictionary.

        Raises:
            RuntimeError: If data has not been prepared before training.

        Returns:
            None
        """
        self._check_data_prepared()

        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Support Vector Classifier": SVC(),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
            ),
        }

        for model_name, model in self.models.items():
            model.fit(self.X_train_scaled, self.y_train)
            print(f"{model_name} trained successfully.")

    def evaluate_models(self) -> None:
        """
        Evaluate all trained models and select the best model.

        The best model is selected based on accuracy score.

        Raises:
            RuntimeError: If models have not been trained before evaluation.

        Returns:
            None
        """
        self._check_models_trained()

        print("\n=== Model Comparison Results ===")

        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)

            self.results[model_name] = accuracy

            print(f"\n{model_name}")
            print(f"Accuracy: {accuracy:.4f}")
            print(classification_report(self.y_test, y_pred))

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = model_name

        print("\n=== Best Model Selected ===")
        print(f"Best Model: {self.best_model_name}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")

    def predict_single_phone(self, phone_data: Dict[str, float]) -> str:
        """
        Predict the price range category of one mobile phone.

        Parameters:
            phone_data (dict): Dictionary containing mobile phone specifications.

        Raises:
            RuntimeError: If the best model is not available.
            ValueError: If required input features are missing.

        Returns:
            str: Predicted price range label.
        """
        self._check_best_model_available()

        missing_features = set(self.feature_columns) - set(phone_data.keys())

        if missing_features:
            raise ValueError(f"Missing phone specification features: {missing_features}")

        phone_df = pd.DataFrame([phone_data])
        phone_df = phone_df[self.feature_columns]

        phone_scaled = self.scaler.transform(phone_df)
        prediction = self.best_model.predict(phone_scaled)

        predicted_class = prediction[0]
        predicted_label = self.price_labels.get(predicted_class, "Unknown")

        print("\n=== Single Phone Price Prediction ===")
        print("Predicted Class:", predicted_class)
        print("Predicted Price Range:", predicted_label)

        return predicted_label

    def plot_best_confusion_matrix(
        self,
        file_path: str = "outputs/best_model_confusion_matrix.png",
    ) -> None:
        """
        Save the confusion matrix image for the best model.

        Parameters:
            file_path (str): Path where the confusion matrix image is saved.

        Raises:
            RuntimeError: If the best model is not available.

        Returns:
            None
        """
        self._check_best_model_available()

        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ConfusionMatrixDisplay.from_estimator(
            self.best_model,
            self.X_test_scaled,
            self.y_test,
        )

        plt.title(f"Confusion Matrix - {self.best_model_name}")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Confusion matrix saved to {output_path}")

    def save_evaluation_report(
        self,
        file_path: str = "outputs/model_report.txt",
    ) -> None:
        """
        Save the final model evaluation report into a text file.

        Parameters:
            file_path (str): Path where the evaluation report is saved.

        Raises:
            RuntimeError: If the best model is not available.

        Returns:
            None
        """
        self._check_best_model_available()

        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        y_pred = self.best_model.predict(self.X_test_scaled)
        report = classification_report(self.y_test, y_pred)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write("=== Mobile Price Classification Report ===\n\n")
            file.write(f"Best Model: {self.best_model_name}\n")
            file.write(f"Best Accuracy: {self.best_accuracy:.4f}\n\n")
            file.write("Price Range Meaning:\n")
            file.write("0 = Cheap / Low Cost\n")
            file.write("1 = Medium Cost\n")
            file.write("2 = Expensive / High Cost\n")
            file.write("3 = Very Expensive\n\n")
            file.write("Classification Report:\n")
            file.write(report)

        print(f"Evaluation report saved to {output_path}")

    def save_best_model(self) -> None:
        """
        Save the best trained model, scaler, feature columns, and price labels.

        The saved pickle package can be reused later for prediction.

        Raises:
            RuntimeError: If the best model is not available.

        Returns:
            None
        """
        self._check_best_model_available()

        model_path = Path(self.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_package = {
            "model": self.best_model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "price_labels": self.price_labels,
            "best_model_name": self.best_model_name,
            "best_accuracy": self.best_accuracy,
        }

        with open(model_path, "wb") as file:
            pickle.dump(model_package, file)

        print(f"Best model package saved to {model_path}")

    def _check_data_prepared(self) -> None:
        """
        Check whether the data has been prepared before model training.

        Raises:
            RuntimeError: If prepare_data() has not been called.

        Returns:
            None
        """
        if self.X_train_scaled is None or self.y_train is None:
            raise RuntimeError("Data has not been prepared. Run prepare_data() first.")

    def _check_models_trained(self) -> None:
        """
        Check whether models have been trained before evaluation.

        Raises:
            RuntimeError: If train_models() has not been called.

        Returns:
            None
        """
        if not self.models:
            raise RuntimeError("Models have not been trained. Run train_models() first.")

    def _check_best_model_available(self) -> None:
        """
        Check whether the best model is available before prediction or saving.

        Raises:
            RuntimeError: If evaluate_models() has not been called.

        Returns:
            None
        """
        if self.best_model is None:
            raise RuntimeError("Best model is not available. Run evaluate_models() first.")
