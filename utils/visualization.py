import os
import matplotlib.pyplot as plt
import seaborn as sns


OUTPUT_DIR = "outputs"


def create_output_folder():
    """
    Create output folder to save visualization images.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_class_balance(data, target_column="price_range"):
    """
    Plot and save the relative frequency of each price range class.
    """
    create_output_folder()

    data[target_column].value_counts(normalize=True).plot(kind="bar")
    plt.xlabel("Price Range Classes")
    plt.ylabel("Frequency")
    plt.title("Class Balance")
    plt.savefig(f"{OUTPUT_DIR}/class_balance.png")
    plt.close()


def plot_ram_distribution(data, target_column="price_range"):
    """
    Plot and save the distribution of RAM based on price range classes.
    """
    create_output_folder()

    sns.boxplot(x=target_column, y="ram", data=data)
    plt.xlabel("Price Range Classes")
    plt.ylabel("RAM")
    plt.title("Distribution of RAM by Price Range Class")
    plt.savefig(f"{OUTPUT_DIR}/ram_distribution.png")
    plt.close()


def plot_battery_power_distribution(data, target_column="price_range"):
    """
    Plot and save the distribution of battery power based on price range classes.
    """
    create_output_folder()

    sns.boxplot(x=target_column, y="battery_power", data=data)
    plt.xlabel("Price Range Classes")
    plt.ylabel("Battery Power")
    plt.title("Distribution of Battery Power by Price Range Class")
    plt.savefig(f"{OUTPUT_DIR}/battery_power_distribution.png")
    plt.close()


def plot_three_g_support(data):
    """
    Plot and save the percentage of phones that support 3G.
    """
    create_output_folder()

    labels = ["Not supported", "3G-supported"]
    values = data["three_g"].value_counts().sort_index().values

    fig, ax = plt.subplots()
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90
    )
    plt.title("3G Supported Phones")
    plt.savefig(f"{OUTPUT_DIR}/three_g_support.png")
    plt.close()