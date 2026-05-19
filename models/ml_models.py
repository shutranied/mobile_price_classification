from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_svc_model(X_train, y_train):
    """
    Train a Support Vector Classifier model.
    """
    model = SVC()
    model.fit(X_train, y_train)
    return model


def train_random_forest_model(X_train, y_train):
    """
    Train a Random Forest Classifier model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def compare_models(X_train, X_test, y_train, y_test):
    """
    Train and compare multiple machine learning models.
    The best model is selected based on accuracy.
    """
    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train),
        "Support Vector Classifier": train_svc_model(X_train, y_train),
        "Random Forest Classifier": train_random_forest_model(X_train, y_train)
    }

    results = {}
    best_model = None
    best_model_name = None
    best_accuracy = 0

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


def predict_single_phone(model, scaler, feature_columns):
    """
    Predict price range for one sample mobile phone.
    The same scaler used during training is applied to the new input.
    """
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
        "wifi": 1
    }

    phone_df = pd.DataFrame([sample_phone])
    phone_df = phone_df[feature_columns]

    phone_scaled = scaler.transform(phone_df)
    prediction = model.predict(phone_scaled)

    price_mapping = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }

    predicted_label = price_mapping.get(prediction[0], "Unknown")

    print("\n=== Single Phone Prediction ===")
    print("Predicted Price Range:", predicted_label)

    return predicted_label


def save_model(model, file_path):
    """
    Save the trained model as a pickle file.
    """
    with open(file_path, "wb") as file:
        pickle.dump(model, file)


def load_model(file_path):
    """
    Load a saved pickle model.
    """
    with open(file_path, "rb") as file:
        model = pickle.load(file)
        return model