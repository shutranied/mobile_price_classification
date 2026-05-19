from config import TRAIN_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, MODEL_PATH
from data.data_loader import load_data
from utils.data_exploration import display_basic_info
from utils.visualization import (
    plot_class_balance,
    plot_ram_distribution,
    plot_battery_power_distribution,
    plot_three_g_support
)
from models.mobile_price_classifier import MobilePriceClassifier


def main():
    # Load dataset
    df_train = load_data(TRAIN_DATA_PATH)

    # Display basic dataset information
    display_basic_info(df_train)

    # Visualize selected features
    plot_class_balance(df_train, TARGET_COLUMN)
    plot_ram_distribution(df_train, TARGET_COLUMN)
    plot_battery_power_distribution(df_train, TARGET_COLUMN)
    plot_three_g_support(df_train)

    # Create OOP classifier object
    classifier = MobilePriceClassifier(
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        model_path=MODEL_PATH
    )

    # Prepare data
    classifier.prepare_data(df_train, TARGET_COLUMN)

    # Train multiple models
    classifier.train_models()

    # Evaluate and select best model
    classifier.evaluate_models()

    # Save confusion matrix
    classifier.plot_best_confusion_matrix()

    # Save evaluation report
    classifier.save_evaluation_report()

    # Save best model
    classifier.save_best_model()

    # Example single phone prediction
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

    classifier.predict_single_phone(sample_phone)


if __name__ == "__main__":
    main()