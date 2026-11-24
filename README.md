# Telecom Customer Churn Prediction

## Project Overview

Telecom Customer Churn Prediction is a machine learning project that analyzes customer data to identify customers who are likely to discontinue their telecom services.

The project uses historical customer information, service subscriptions, contract details, and billing information to predict customer churn.

A machine learning model generates churn probabilities and classifies customers into three risk categories:

- **High Risk:** Customers with a churn probability of 70% or above.
- **Medium Risk:** Customers with a churn probability between 30% and 70%.
- **Low Risk:** Customers with a churn probability below 30%.

The project also includes an interactive Streamlit dashboard that allows users to explore customer churn patterns, enter customer details, generate predictions, and view business insights.
## Project Objectives

1. Analyze customer data to understand patterns associated with telecom customer churn.
2. Perform data cleaning and preprocessing to prepare the dataset for machine learning.
3. Identify important factors associated with customer churn.
4. Build and compare multiple machine learning classification models.
5. Improve model performance using class balancing, cross-validation, hyperparameter tuning, and threshold analysis.
6. Generate customer-level churn probabilities and risk categories.
7. Develop an interactive dashboard to display predictions and business insights.
8. Provide data-driven recommendations to help telecom companies improve customer retention.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Data Processing | SciPy |
| Model Persistence | Joblib |
| Development Environment | Jupyter Notebook, VS Code |
| Web Application | Streamlit |
| Version Control | Git, GitHub |

## Project Structure

```text
Telecom-Customer-Churn-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       ├── cleaned_telco_data.csv
│       ├── X_train_processed.npz
│       ├── X_test_processed.npz
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_building.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_model_improvement.ipynb
│   └── 06_business_insights_and_recommendations.ipynb
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── tuned_logistic_regression.pkl
│   └── preprocessing_objects.joblib
│
├── reports/
│   ├── customer_churn_risk_scores.csv
│   └── prediction_results.csv
│
├── src/
│
├── requirements.txt
└── README.md
```
## Project Workflow

The project follows these major stages:

1. **Data Understanding and Exploratory Data Analysis (EDA)**
   - Load and inspect the telecom customer dataset.
   - Analyze customer demographics, services, contracts, and billing information.
   - Identify patterns associated with customer churn.

2. **Data Preprocessing**
   - Handle missing values.
   - Separate features and target variables.
   - Split data into training and testing sets.
   - Apply StandardScaler and OneHotEncoder.

3. **Model Building**
   - Train Logistic Regression.
   - Train Decision Tree.
   - Train Random Forest.

4. **Model Evaluation**
   - Evaluate models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
   - Analyze confusion matrices and feature importance.

5. **Model Improvement**
   - Apply class balancing.
   - Perform cross-validation and hyperparameter tuning.
   - Experiment with different prediction thresholds.

6. **Customer Risk Scoring**
   - Generate churn probabilities.
   - Categorize customers into High, Medium, and Low risk groups.

7. **Business Insights**
   - Identify customer segments with higher churn rates.
   - Develop customer retention recommendations.

8. **Deployment**
   - Build an interactive Streamlit application.
   - Allow users to enter customer details and predict churn.
   - Save prediction results in a CSV file.

## Dataset Description

This project uses the **IBM Telco Customer Churn dataset**, which contains customer information from a fictional telecommunications company.

The dataset consists of:

- **7,043 customer records**
- **21 original columns**
- Customer demographic information
- Services subscribed by customers
- Contract and payment details
- Monthly and total charges
- Customer churn status

### Target Variable

The target variable is `Churn`.

| Value | Meaning |
|---|---|
| Yes | Customer discontinued the service |
| No | Customer continued using the service |

### Dataset Source

IBM Telco Customer Churn dataset – Kaggle.

The dataset is used for educational and machine learning purposes.

## Model Performance

Three machine learning classification algorithms were initially trained and evaluated.

### Baseline Model Comparison

| Evaluation Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---:|---:|---:|
| Accuracy | 80.55% | 72.25% | 78.85% |
| Precision | 65.72% | 47.77% | 63.10% |
| Recall | 55.88% | 48.66% | 48.93% |
| F1-Score | 60.40% | 48.21% | 55.12% |
| ROC-AUC | 84.20% | 64.69% | 82.01% |

### Model Improvement

Class balancing was applied to Logistic Regression to improve the detection of customers likely to churn.

| Evaluation Metric | Baseline Logistic Regression | Improved Logistic Regression |
|---|---:|---:|
| Accuracy | 80.55% | 73.81% |
| Precision | 65.72% | 50.43% |
| Recall | 55.88% | 78.34% |
| F1-Score | 60.40% | 61.36% |
| ROC-AUC | 84.20% | 84.15% |

The improved model achieved higher churn recall, helping identify more customers who were likely to leave.

The final model uses a 0.5 prediction threshold, with customer risk categories assigned using churn probability.

## Installation and Usage

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd Telecom-Customer-Churn-Prediction
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Application

```bash
streamlit run app/app.py
```

### 5. Access the Application

Once the application starts, open the local URL displayed in the terminal (usually `http://localhost:8501`).

### Application Features

- Interactive customer churn dashboard.
- Customer churn prediction using machine learning.
- Churn probability and risk classification.
- Customer churn insights and visualizations.
- Business recommendations for customer retention.
- Prediction history stored in CSV format.

## Key Business Insights

Exploratory data analysis revealed several customer segments with higher churn rates.

| Factor | Customer Segment | Churn Rate |
|---|---|---:|
| Contract | Month-to-month | 42.71% |
| Contract | One year | 11.27% |
| Contract | Two year | 2.83% |
| Payment Method | Electronic check | 45.29% |
| Internet Service | Fiber optic | 41.89% |
| Online Security | No | 41.77% |
| Tech Support | No | 41.64% |

### Business Recommendations

1. Encourage month-to-month customers to consider long-term contracts through suitable retention offers.
2. Identify electronic-check customers as a segment for further churn analysis.
3. Investigate customer satisfaction among fiber-optic subscribers.
4. Consider offering suitable online security and technical support packages.
5. Monitor customers with shorter tenure and higher monthly charges.
6. Use churn risk scores to prioritize customer retention campaigns.

**Note:** These findings represent associations in the dataset and do not establish causation.

## Future Enhancements

- Deploy the application on Streamlit Community Cloud.
- Experiment with advanced machine learning algorithms.
- Improve model performance through additional feature engineering.
- Integrate real-time customer data for churn prediction.
- Develop personalized customer retention strategies.

## Author

Developed as an end-to-end Machine Learning project to analyze telecom customer churn and support data-driven customer retention strategies.

### Dataset Source

IBM Telco Customer Churn dataset:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains 7,043 customer records and 21 columns.