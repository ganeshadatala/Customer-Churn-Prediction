# Customer Churn Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

## 1. Project Overview

Customer Churn Prediction is a Machine Learning project designed to predict whether a customer is likely to leave a telecommunications company.

The project analyzes customer information, identifies patterns associated with customer churn, and builds a predictive model using Logistic Regression. A Streamlit web application allows users to enter customer details and receive churn predictions.

**Project Goals:**

* Understand customer churn patterns.
* Perform data cleaning and exploratory data analysis.
* Build and evaluate a Machine Learning classification model.
* Develop an interactive web application for predictions.
* Support data-driven customer retention decisions.

## 2. Business Problem

Customer churn occurs when customers stop using a company's products or services.

High customer churn can reduce revenue and increase the cost of acquiring new customers. Businesses need to understand which customers may leave so they can improve customer satisfaction and retention.

**Business Questions:**

* Which customer groups have higher churn rates?
* How does customer tenure relate to churn?
* Does contract type influence customer churn?
* Which payment methods are associated with different churn patterns?
* Can Machine Learning help predict whether a customer may leave?

**Proposed Solution:**

Analyze historical customer data and train a classification model to predict customer churn. Businesses can use these predictions to identify customers who may need additional support or retention offers.

## 3. Dataset Description

**Dataset:** Telco Customer Churn

**File:** `Telco-Customer-Churn.csv`

The dataset contains 7,043 customer records and 21 columns before cleaning.

| Column             | Description                              |
| ------------------ | ---------------------------------------- |
| `customerID`       | Unique customer identifier               |
| `gender`           | Customer gender                          |
| `SeniorCitizen`    | Indicates senior citizen status          |
| `Partner`          | Whether the customer has a partner       |
| `Dependents`       | Whether the customer has dependents      |
| `tenure`           | Number of months the customer has stayed |
| `PhoneService`     | Whether phone service is enabled         |
| `MultipleLines`    | Whether multiple phone lines are used    |
| `InternetService`  | Type of internet service                 |
| `OnlineSecurity`   | Online security service status           |
| `OnlineBackup`     | Online backup service status             |
| `DeviceProtection` | Device protection service status         |
| `TechSupport`      | Technical support service status         |
| `StreamingTV`      | Streaming TV service status              |
| `StreamingMovies`  | Streaming movies service status          |
| `Contract`         | Customer contract type                   |
| `PaperlessBilling` | Whether paperless billing is enabled     |
| `PaymentMethod`    | Customer payment method                  |
| `MonthlyCharges`   | Monthly amount charged                   |
| `TotalCharges`     | Total amount charged                     |
| `Churn`            | Whether the customer left the company    |

**Target Variable:** `Churn`

* `Yes` = Customer left the company.
* `No` = Customer stayed with the company.

## 4. Technologies Used

| Technology          | Purpose                                       |
| ------------------- | --------------------------------------------- |
| Python              | Programming and model development             |
| Pandas              | Data manipulation and analysis                |
| NumPy               | Numerical operations                          |
| Matplotlib          | Data visualization                            |
| Seaborn             | Exploratory data visualization                |
| Scikit-learn        | Preprocessing, model training, and evaluation |
| Logistic Regression | Churn classification                          |
| Joblib              | Saving and loading the trained model          |
| SQL Server          | Database storage and SQL analysis             |
| Streamlit           | Interactive prediction application            |
| Jupyter Notebook    | Data exploration and experimentation          |
| VS Code             | Development environment                       |
| Git & GitHub        | Version control and project hosting           |

## 5. Data Cleaning

Data cleaning was performed to prepare the dataset for analysis and Machine Learning.

**Steps performed:**

1. Loaded the dataset using Pandas.
2. Inspected the dataset shape, columns, and data types.
3. Checked for missing values.
4. Converted `TotalCharges` from text to a numeric data type.
5. Identified blank values in `TotalCharges`.
6. Removed the `customerID` column from model features because it is an identifier rather than a meaningful predictive measurement.
7. Prepared the target variable `Churn` for classification.
8. Used preprocessing pipelines to handle missing numerical and categorical feature values.

For model preprocessing:

* Numerical missing values are handled using median imputation.
* Categorical missing values are handled using most-frequent imputation.
* Numerical features are standardized.
* Categorical features are converted using One-Hot Encoding.

## 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand customer characteristics and churn patterns.

**Analyses performed:**

* Overall customer churn rate.
* Gender versus churn.
* Contract type versus churn.
* Customer tenure versus churn.
* Payment method versus churn.
* Customer service and billing characteristics.

**Key business insights to investigate:**

* Whether month-to-month contracts have higher churn than longer-term contracts.
* Whether newer customers are more likely to leave.
* Whether churn differs across payment methods.
* Which customer characteristics may be useful for retention strategies.

*Actual findings should be added after checking the charts and results generated by the project.*

## 7. Machine Learning Workflow

The project follows these Machine Learning steps:

1. **Data Loading:** Load the Telco Customer Churn dataset.
2. **Data Cleaning:** Correct data types and prepare missing values.
3. **Feature Selection:** Separate input features from the target variable.
4. **Preprocessing:** Impute missing values, scale numerical features, and encode categorical features.
5. **Train-Test Split:** Split the data into training and testing sets using an 80:20 ratio.
6. **Model Training:** Train a Logistic Regression classifier.
7. **Model Evaluation:** Measure performance on the test dataset.
8. **Model Saving:** Save the trained pipeline using Joblib.
9. **Prediction:** Load the saved model and predict churn for customer inputs.
10. **Deployment Interface:** Use Streamlit to provide an interactive prediction interface.

**Algorithm:** Logistic Regression

**Train-Test Split:**

* Training data: 80%
* Testing data: 20%

**Reproducibility:** `random_state=42`

The preprocessing pipeline and classifier are combined so the same transformations can be applied during both training and prediction.

## 8. Model Evaluation Results

The model is evaluated using the following classification metrics:

| Metric                | Purpose                                                            |
| --------------------- | ------------------------------------------------------------------ |
| Accuracy              | Overall proportion of correct predictions                          |
| Precision             | Proportion of predicted churn cases that were actually churn cases |
| Recall                | Proportion of actual churn cases correctly identified              |
| F1-score              | Balance between precision and recall                               |
| Classification Report | Summary of precision, recall, F1-score, and support                |

**Model:** Logistic Regression

**Test Set:** 20% of the available data

### Results

Add your actual model results below after running the training script and checking `output/model_evaluation.txt`.

| Metric    | Result              |
| --------- | ------------------- |
| Accuracy  | Add measured result |
| Precision | Add measured result |
| Recall    | Add measured result |
| F1-score  | Add measured result |

*Note: No performance values are included here because the actual evaluation output must be verified before publication.*

## 9. How to Run the Project

### Prerequisites

Install Python and Git. VS Code is recommended.

### Step 1: Clone the Repository

```bash
git clone https://github.com/ganeshadatala/Customer-Churn-Prediction.git
```

### Step 2: Open the Project Directory

```bash
cd Customer-Churn-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit jupyter
```

### Step 5: Train the Model

Run the following command from the project root:

```bash
python src/train_model.py
```

This should train the model and save the pipeline if the script and dataset paths are configured correctly.

### Step 6: Run the Prediction Script

```bash
python src/predict.py
```

### Step 7: Start the Streamlit Application

```bash
streamlit run app.py
```

```markdown
![Customer Churn Analysis](output/churn_analysis.png)
```

## 8. Conclusion and Future Improvements

### Conclusion

This project demonstrates an end-to-end Machine Learning workflow for customer churn prediction.

It includes data cleaning, exploratory data analysis, feature preprocessing, Logistic Regression model training, model evaluation, and an interactive Streamlit application.

The project provides practical experience with Python, Pandas, Scikit-learn, SQL, and application development.

The predictions may help businesses identify customers who could be at risk of leaving, but the model should be evaluated and monitored before being used for real business decisions.

### Future Improvements

* Compare Logistic Regression with Random Forest and other classification algorithms.
* Perform hyperparameter tuning to improve model performance.
* Address class imbalance if needed.
* Add more detailed visualizations and a customer churn dashboard.
* Improve the Streamlit interface and input validation.
* Deploy the application to a cloud platform.
* Monitor model performance as new customer data becomes available.
* Explore explainability techniques to understand individual predictions.

---

## Author

**Ganesh Adatala**

GitHub: [ganeshadatala](https://github.com/ganeshadatala)

**Project:** Customer Churn Prediction Using Machine Learning
