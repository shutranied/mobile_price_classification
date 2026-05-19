import os
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


class MobilePriceClassifier:
    """
    Object-Oriented Programming class for mobile price classification.
    This class handles preprocessing, model training, evaluation,
    prediction, and model saving.
    """

    def __init__(self, test_size=0.2, random_state=42, model_path="models/mobile_price_model.pkl"):
        self.test_size = test_size
        self.random_state = random_state
        self.model_path = model_path

        self.scaler = MinMaxScaler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
        self.feature_columns = None

        self.price_labels = {
            0: "Cheap / Low Cost",
            1: "Medium Cost",
            2: "Expensive / High Cost",
            3: "Very Expensive"
        }

    def prepare_data(self, data, target_column):
        """
        Split dataset into features and target, then split into training and testing sets.
        Scaling is fitted only on training data to avoid data leakage.
        """

        X = data.drop(target_column, axis=1)
        y = data[target_column]

        self.feature_columns = X.columns

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("\nData prepared successfully.")
        print("Training data shape:", self.X_train.shape)
        print("Testing data shape:", self.X_test.shape)

    def train_models(self):
        """
        Train multiple machine learning classification models.
        """

        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Support Vector Classifier": SVC(),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        }

        for model_name, model in self.models.items():
            model.fit(self.X_train_scaled, self.y_train)
            print(f"{model_name} trained successfully.")

    def evaluate_models(self):
        """
        Evaluate all trained models and select the best model based on accuracy.
        """

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

    def predict_single_phone(self, phone_data):
        """
        Predict the price class of one mobile phone.
        """

        import pandas as pd

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

    def plot_best_confusion_matrix(self, file_path="outputs/best_model_confusion_matrix.png"):
        """
        Save confusion matrix image for the best model.
        """

        os.makedirs("outputs", exist_ok=True)

        ConfusionMatrixDisplay.from_estimator(
            self.best_model,
            self.X_test_scaled,
            self.y_test
        )

        plt.title(f"Confusion Matrix - {self.best_model_name}")
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        print(f"Confusion matrix saved to {file_path}")

    def save_evaluation_report(self, file_path="outputs/model_report.txt"):
        """
        Save final model evaluation report into a text file.
        """

        os.makedirs("outputs", exist_ok=True)

        y_pred = self.best_model.predict(self.X_test_scaled)
        report = classification_report(self.y_test, y_pred)

        with open(file_path, "w") as file:
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

        print(f"Evaluation report saved to {file_path}")

    def save_best_model(self):
        """
        Save the best trained model and scaler.
        """

        model_package = {
            "model": self.best_model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "price_labels": self.price_labels
        }

        with open(self.model_path, "wb") as file:
            pickle.dump(model_package, file)

        print(f"Best model package saved to {self.model_path}")