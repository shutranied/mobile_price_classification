"""
Main execution script for the Mobile Price Classification project.

This file runs the complete machine learning workflow:
1. Load the training dataset.
2. Display basic dataset information.
3. Generate exploratory data visualizations.
4. Create the MobilePriceClassifier object.
5. Prepare the data for model training.
6. Train multiple classification models.
7. Evaluate the models and select the best model.
8. Save the confusion matrix, evaluation report, and trained model.
9. Predict the price range of a sample mobile phone.
"""

from config import TRAIN_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, MODEL_PATH
from data.data_loader import load_data
from utils.data_exploration import display_basic_info
from utils.visualization import (
    plot_class_balance,
    plot_ram_distribution,
    plot_battery_power_distribution,
    plot_three_g_support,
)
from models.mobile_price_classifier import MobilePriceClassifier


def main() -> None:
    """
    Run the full mobile price classification workflow.

    This function loads the dataset, performs basic exploration,
    generates visualizations, trains multiple machine learning models,
    evaluates model performance, saves the best model, and performs
    a sample prediction using the trained classifier.

    Returns:
        None
    """

    print("Loading training dataset...")
    df_train = load_data(TRAIN_DATA_PATH)

    print("\nDisplaying basic dataset information...")
    display_basic_info(df_train)

    print("\nGenerating visualizations...")
    plot_class_balance(df_train, TARGET_COLUMN)
    plot_ram_distribution(df_train, TARGET_COLUMN)
    plot_battery_power_distribution(df_train, TARGET_COLUMN)
    plot_three_g_support(df_train)

    print("\nCreating classifier object...")
    classifier = MobilePriceClassifier(
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        model_path=MODEL_PATH,
    )

    print("\nPreparing data...")
    classifier.prepare_data(df_train, TARGET_COLUMN)

    print("\nTraining models...")
    classifier.train_models()

    print("\nEvaluating models...")
    classifier.evaluate_models()

    print("\nSaving confusion matrix...")
    classifier.plot_best_confusion_matrix()

    print("\nSaving evaluation report...")
    classifier.save_evaluation_report()

    print("\nSaving best model...")
    classifier.save_best_model()

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

    print("\nPredicting price range for a sample mobile phone...")
    classifier.predict_single_phone(sample_phone)

    print("\nWorkflow completed successfully.")


if __name__ == "__main__":
    main()
