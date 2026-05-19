from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
