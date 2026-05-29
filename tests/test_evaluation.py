import matplotlib

matplotlib.use("Agg")

from sklearn.dummy import DummyClassifier

from evaluation.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    save_evaluation_report,
)


def test_evaluate_model_returns_accuracy_and_predictions():
    """
    Test that evaluate_model() returns the correct accuracy
    and prediction values.
    """

    class FakeModel:
        def predict(self, X_test):
            return [0, 1, 1, 0]

    X_test = [[100], [200], [300], [400]]
    y_test = [0, 1, 0, 0]

    accuracy, y_pred = evaluate_model(FakeModel(), X_test, y_test)

    assert accuracy == 0.75
    assert list(y_pred) == [0, 1, 1, 0]


def test_save_evaluation_report_creates_file(tmp_path):
    """
    Test that save_evaluation_report() creates a report text file.
    """

    model = DummyClassifier(strategy="most_frequent")
    X_train = [[100], [200], [300], [400]]
    y_train = [0, 0, 1, 1]
    model.fit(X_train, y_train)

    X_test = [[150], [250]]
    y_test = [0, 1]

    report_path = tmp_path / "model_report.txt"

    save_evaluation_report(
        model,
        X_test,
        y_test,
        model_name="Dummy Classifier",
        file_path=report_path,
    )

    assert report_path.exists()

    report_content = report_path.read_text()

    assert "Mobile Price Classification Model Report" in report_content
    assert "Dummy Classifier" in report_content
    assert "Accuracy" in report_content
    assert "Classification Report" in report_content


def test_plot_confusion_matrix_creates_image_file(tmp_path):
    """
    Test that plot_confusion_matrix() creates a confusion matrix image file.
    """

    model = DummyClassifier(strategy="most_frequent")
    X_train = [[100], [200], [300], [400]]
    y_train = [0, 0, 1, 1]
    model.fit(X_train, y_train)

    X_test = [[150], [250]]
    y_test = [0, 1]

    image_path = tmp_path / "confusion_matrix.png"

    plot_confusion_matrix(
        model,
        X_test,
        y_test,
        file_path=image_path,
    )

    assert image_path.exists()
