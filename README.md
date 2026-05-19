# Mobile Price Classification - Code Decomposition and OOP Project

## 1. Project Overview

This project is based on a publicly available Kaggle notebook titled **Mobile Price Classification**.

Original Kaggle Notebook:

```text
https://www.kaggle.com/code/abdelruhmanessam/mobile-price-classification
```

The purpose of this project is to restructure the original Kaggle notebook into a clean, organized, and modular Python project. The original notebook code has been decomposed into separate modules according to the main stages of a machine learning workflow, including data loading, data exploration, visualization, preprocessing, model training, model evaluation, model saving, and prediction.

In addition, Object-Oriented Programming (OOP) was added by creating a `MobilePriceClassifier` class. This class manages the complete machine learning workflow, including data preparation, model training, model comparison, evaluation, output saving, and predicting the price category of a new mobile phone.

---

## 2. Dataset

The dataset used in this project is the **Mobile Price Classification** dataset.

The main training file is:

```text
train.csv
```

The target variable is:

```text
price_range
```

The model predicts the price range category of a mobile phone based on technical specifications such as:

- RAM
- Battery power
- Internal memory
- Camera specifications
- Screen resolution
- 3G support
- 4G support
- Wi-Fi support
- Touch screen availability

The `price_range` target column has four classes:

| Class | Meaning |
|---|---|
| 0 | Cheap / Low Cost |
| 1 | Medium Cost |
| 2 | Expensive / High Cost |
| 3 | Very Expensive |

---

## 3. Project Folder Structure

```text
mobile_price_classification/
├── data/
│   ├── data_loader.py
│   ├── test.csv
│   └── train.csv
│
├── evaluation/
│   └── evaluation.py
│
├── models/
│   ├── ml_models.py
│   ├── mobile_price_classifier.py
│   └── mobile_price_model.pkl
│
├── outputs/
│   ├── battery_power_distribution.png
│   ├── best_model_confusion_matrix.png
│   ├── class_balance.png
│   ├── model_report.txt
│   ├── ram_distribution.png
│   └── three_g_support.png
│
├── preprocessing/
│   └── preprocessing.py
│
├── utils/
│   ├── data_exploration.py
│   └── visualization.py
│
├── .gitignore
├── config.py
├── main.py
├── README.md
└── requirements.txt
```

---

## 4. Module Explanation

### 4.1 `config.py`

This file stores important project settings such as:

- Training dataset path
- Testing dataset path
- Target column name
- Test size
- Random state
- Model save path

Example:

```python
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"
TARGET_COLUMN = "price_range"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_PATH = "models/mobile_price_model.pkl"
```

---

### 4.2 `data/data_loader.py`

This file contains the function for loading the dataset from a CSV file using pandas.

The function used is:

```python
load_data(file_path)
```

This function reads the CSV file and returns it as a pandas DataFrame.

---

### 4.3 `utils/data_exploration.py`

This file contains functions for basic data exploration.

The main function used is:

```python
display_basic_info(data)
```

It displays:

- Dataset information
- Descriptive statistics
- Number of unique values
- Missing values
- First five rows of the dataset

This step helps to understand the dataset before training the machine learning models.

---

### 4.4 `utils/visualization.py`

This file contains visualization functions used to generate and save charts.

The visualizations generated are:

- Class balance chart
- RAM distribution chart
- Battery power distribution chart
- 3G support distribution chart

The functions used are:

```python
plot_class_balance()
plot_ram_distribution()
plot_battery_power_distribution()
plot_three_g_support()
```

These visualizations are used for exploratory data analysis. They help to understand the distribution of important features in the dataset.

---

### 4.5 `preprocessing/preprocessing.py`

This file contains preprocessing functions for preparing the dataset before model training.

The preprocessing process includes:

1. Separating the input features and target variable
2. Splitting the dataset into training and testing sets
3. Scaling the feature values

In the updated version, the dataset is split before scaling. This helps to reduce data leakage because the scaler is fitted only on the training data and then applied to the testing data.

The main preprocessing functions are:

```python
split_features_target()
split_train_test()
scale_train_test_features()
```

---

### 4.6 `models/ml_models.py`

This file contains earlier model-related helper functions, including:

- Training Logistic Regression
- Training Support Vector Classifier
- Saving a model
- Loading a model

Although the latest workflow mainly uses the OOP class in `mobile_price_classifier.py`, this file is still part of the modular project structure.

---

### 4.7 `models/mobile_price_classifier.py`

This is the main Object-Oriented Programming file in the project.

It contains the class:

```python
MobilePriceClassifier
```

This class manages the full machine learning workflow.

The main methods inside the class are:

```python
prepare_data()
train_models()
evaluate_models()
predict_single_phone()
plot_best_confusion_matrix()
save_evaluation_report()
save_best_model()
```

The class trains and compares three machine learning classification models:

1. Logistic Regression
2. Support Vector Classifier
3. Random Forest Classifier

The best model is selected based on accuracy.

---

### 4.8 `evaluation/evaluation.py`

This file contains model evaluation functions.

The evaluation process includes:

- Accuracy score
- Classification report
- Confusion matrix display

The classification report shows important classification metrics such as precision, recall, F1-score, and support.

---

### 4.9 `main.py`

This is the main file that runs the complete machine learning workflow.

The workflow in `main.py` includes:

1. Loading the dataset
2. Displaying dataset information
3. Generating visualizations
4. Creating the `MobilePriceClassifier` object
5. Preparing the data
6. Training multiple models
7. Comparing model performance
8. Selecting the best model
9. Saving the confusion matrix
10. Saving the evaluation report
11. Saving the best model
12. Predicting the price range of a sample mobile phone

---

## 5. Object-Oriented Programming Implementation

Object-Oriented Programming was added to improve the structure and reusability of the project.

The main OOP class is:

```python
MobilePriceClassifier
```

This class groups the machine learning workflow into one object. Instead of writing all processes separately in `main.py`, the class organizes related functions into methods.

The OOP implementation makes the project:

- Easier to read
- Easier to maintain
- More reusable
- More organized
- More suitable for a real machine learning application

---

## 6. Machine Learning Models Used

Three classification models were used in this project:

| Model | Description |
|---|---|
| Logistic Regression | A classification model used to predict categorical outcomes |
| Support Vector Classifier | A classification model that separates classes using a decision boundary |
| Random Forest Classifier | An ensemble model that uses multiple decision trees |

The models were trained and compared based on accuracy.

Based on the terminal output, the model comparison result was:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 91.75% |
| Support Vector Classifier | 86.00% |
| Random Forest Classifier | 87.75% |

The best model selected was:

```text
Logistic Regression
Accuracy: 91.75%
```

This means Logistic Regression performed the best among the three models used in this project.

---

## 7. Model Evaluation Result

The best model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Support
- Confusion matrix

The final best model achieved:

```text
Best Model: Logistic Regression
Best Accuracy: 0.9175
```

This means the model achieved an accuracy of **91.75%** on the testing data.

The confusion matrix was saved as:

```text
outputs/best_model_confusion_matrix.png
```

The evaluation report was saved as:

```text
outputs/model_report.txt
```

---

## 8. Single Phone Prediction

After training and evaluating the models, the best model was used to predict the price category of one sample mobile phone.

The sample phone specification included features such as:

- Battery power
- Bluetooth support
- Clock speed
- Dual SIM support
- Front camera
- 4G support
- Internal memory
- Mobile weight
- Number of cores
- Primary camera
- Pixel height
- Pixel width
- RAM
- Screen height
- Screen width
- Talk time
- 3G support
- Touch screen
- Wi-Fi support

The prediction result was:

```text
Predicted Class: 3
Predicted Price Range: Very Expensive
```

This means the sample phone was classified as a **very expensive mobile phone** based on its specifications.

---

## 9. Output Files

The project generates several output files inside the `outputs` folder.

The visualization output files are:

```text
class_balance.png
ram_distribution.png
battery_power_distribution.png
three_g_support.png
```

The model evaluation output files are:

```text
best_model_confusion_matrix.png
model_report.txt
```

The trained model package is saved as:

```text
models/mobile_price_model.pkl
```

The saved model package contains:

- Best trained model
- Scaler
- Feature columns
- Price labels

---

## 10. How to Run the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

If using Python 3:

```bash
python3 main.py
```

---

## 11. Sample Terminal Output

Example output from the program:

```text
Data prepared successfully.
Training data shape: (1600, 20)
Testing data shape: (400, 20)

Logistic Regression trained successfully.
Support Vector Classifier trained successfully.
Random Forest Classifier trained successfully.

=== Model Comparison Results ===

Logistic Regression
Accuracy: 0.9175

Support Vector Classifier
Accuracy: 0.8600

Random Forest Classifier
Accuracy: 0.8775

=== Best Model Selected ===
Best Model: Logistic Regression
Best Accuracy: 0.9175

Confusion matrix saved to outputs/best_model_confusion_matrix.png
Evaluation report saved to outputs/model_report.txt
Best model package saved to models/mobile_price_model.pkl

=== Single Phone Price Prediction ===
Predicted Class: 3
Predicted Price Range: Very Expensive
```

---

## 12. Conclusion

This project shows how a Kaggle machine learning notebook can be decomposed into a structured Python project and improved using Object-Oriented Programming.

The updated system successfully performs:

- Data loading
- Data exploration
- Data visualization
- Data preprocessing
- Model training
- Model comparison
- Best model selection
- Model evaluation
- Model saving
- Single-phone price prediction

The best model was **Logistic Regression** with an accuracy of **91.75%**.

The final prediction example classified a sample mobile phone as:

```text
Class 3 = Very Expensive
```

Overall, this project demonstrates a complete machine learning classification workflow for predicting mobile phone price range based on technical specifications.
