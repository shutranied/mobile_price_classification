from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def split_features_target(data, target_column):
    """
    Separate the dataset into features (X) and target variable (y).
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    Stratify is used to keep the class distribution balanced.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_train_test_features(X_train, X_test):
    """
    Fit MinMaxScaler on training data only, then transform both
    training and testing data.
    """
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler