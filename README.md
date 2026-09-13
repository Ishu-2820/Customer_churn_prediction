
# Customer Churn Prediction --- End-to-End Machine Learning

An end-to-end machine learning project for predicting telecom customer
churn and turning churn predictions into actionable customer-retention
insights.

The project covers data quality checks, exploratory data analysis,
feature engineering, preprocessing, model comparison, hyperparameter
tuning, probability-threshold optimization, model explainability with
SHAP, model serialization, and an interactive Streamlit application.

------------------------------------------------------------------------

## Business Problem

Customer churn can reduce recurring revenue and increase
customer-acquisition costs.

The objective of this project is to:

-   Identify customers who are more likely to churn.
-   Compare multiple classification models.
-   Improve the final model through hyperparameter tuning.
-   Select a probability threshold using validation data.
-   Explain predictions using SHAP.
-   Provide an interactive churn-risk prediction application.

------------------------------------------------------------------------

## Dataset

The notebook uses the **Telco Customer Churn** dataset.

-   **Initial records:** 7,043
-   **Initial features:** 21
-   **Target:** `Churn`
-   **Target classes:** `No`, `Yes`
-   **Initial churn rate:** approximately 26.54%

During data cleaning:

-   `TotalCharges` is converted from object to numeric.
-   11 rows with missing `TotalCharges` values are removed.
-   `customerID` is removed because it is an identifier rather than a
    predictive feature.
-   The cleaned modeling dataset contains **7,032 records and 20
    columns**.

------------------------------------------------------------------------

## Project Workflow

``` text
Raw Data
   ↓
Data Quality Checks
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Preprocessing
   ├── StandardScaler for numerical features
   └── OneHotEncoder for categorical features
   ↓
Baseline Model Comparison
   ├── Logistic Regression
   ├── Random Forest
   └── XGBoost
   ↓
XGBoost Hyperparameter Tuning
   ↓
Validation-Based Threshold Selection
   ↓
Final Test Evaluation
   ↓
SHAP Explainability
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

------------------------------------------------------------------------

## Exploratory Data Analysis

The notebook examines churn distribution and relationships between
customer characteristics and churn.

Examples include:

-   Churn distribution
-   Monthly charges vs. churn
-   Tenure-related churn patterns
-   Contract type vs. churn
-   Payment method vs. churn
-   Internet service vs. churn
-   Service usage and churn
-   High-value customer analysis

The analysis is used to understand patterns in the dataset before model
development.

------------------------------------------------------------------------

## Feature Engineering

The project creates business-oriented features to provide additional
information to the models.

### Engineered Features

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  `total_services`                    Count of selected services used by
                                      a customer

  `is_high_value_customer`            Indicates customers at or above the
                                      75th percentile of monthly charges

  `is_month_to_month`                 Indicates whether the customer has
                                      a month-to-month contract

  `has_partner_or_dependents`         Indicates whether the customer has
                                      a partner or dependents

  `avg_monthly_spend`                 Total charges divided by tenure,
                                      with a fallback for zero tenure
  -----------------------------------------------------------------------

The high-value customer threshold learned in the notebook is **89.8625**
in monthly charges.

------------------------------------------------------------------------

## Data Preprocessing

The project uses a `ColumnTransformer` to apply different preprocessing
strategies to numerical and categorical features.

### Numerical Features

`StandardScaler` is applied to numerical variables.

### Categorical Features

`OneHotEncoder` is used with:

-   `handle_unknown="ignore"`
-   `drop="first"`

After preprocessing, the feature matrix contains **35 features**.

------------------------------------------------------------------------

## Train/Test Strategy

The cleaned dataset is split using a stratified train/test split:

-   **Training set:** 80%
-   **Test set:** 20%
-   **Random state:** 42
-   **Stratification:** enabled

This produces:

-   Training: 5,625 records
-   Testing: 1,407 records

A separate validation split is created from the training data for tuning
and threshold selection, while the test set is kept untouched until
final evaluation.

------------------------------------------------------------------------

## Models Evaluated

The project compares four model stages:

1.  **Logistic Regression**
2.  **Random Forest**
3.  **XGBoost**
4.  **Tuned XGBoost**

Evaluation focuses on:

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   ROC-AUC

For churn prediction, recall and F1 score are particularly useful
because missing a customer who is likely to churn can reduce the value
of a retention strategy.

------------------------------------------------------------------------

## Model Tuning

`GridSearchCV` is used to tune XGBoost hyperparameters.

The search covers:

-   `n_estimators`
-   `max_depth`
-   `learning_rate`
-   `subsample`

The notebook selects the following best parameter combination:

``` text
learning_rate = 0.03
max_depth     = 3
n_estimators  = 200
subsample     = 0.8
```

The final XGBoost model uses the parameters returned by the grid search
rather than a separate manually selected parameter set.

------------------------------------------------------------------------

## Probability Threshold Optimization

Instead of relying only on the default probability threshold of 0.50,
the project evaluates precision and recall on the validation set and
selects the threshold that maximizes F1 score.

The selected validation threshold is:

``` text
0.349
```

Validation F1 at this threshold:

``` text
0.6289
```

This approach allows the classification decision to better reflect the
trade-off between identifying churners and limiting false positives.

------------------------------------------------------------------------

## Final Model Performance

The final tuned XGBoost model achieves the following test-set results:

  Metric         Score
  ----------- --------
  Accuracy      77.61%
  Precision     55.89%
  Recall        74.87%
  F1 Score      64.00%
  ROC-AUC       84.02%

### Model Comparison

  Model                   Accuracy   Precision   Recall       F1   ROC-AUC
  --------------------- ---------- ----------- -------- -------- ---------
  Logistic Regression       80.31%      64.83%   56.68%   60.49%    83.58%
  Random Forest             78.61%      51.13%   78.61%   61.96%    83.62%
  XGBoost                   78.39%      61.15%   51.34%   55.81%    82.92%
  Tuned XGBoost             77.61%      55.89%   74.87%   64.00%    84.02%

The tuned XGBoost model provides the highest ROC-AUC and F1 score among
the compared models, while the threshold optimization substantially
improves churn recall.

------------------------------------------------------------------------

## Classification Performance

For the final model, the notebook reports:

-   **No Churn:** precision 0.90, recall 0.79, F1 0.84
-   **Churn:** precision 0.56, recall 0.75, F1 0.64
-   **Overall accuracy:** 0.78

The model therefore identifies approximately three-quarters of the churn
cases in the held-out test set.

------------------------------------------------------------------------

## Explainability with SHAP

SHAP is used to make the final XGBoost model more interpretable.

The project includes SHAP-based feature-importance analysis to
understand:

-   Which variables influence churn predictions.
-   The direction and magnitude of feature contributions.
-   Why individual predictions may be classified as higher or lower
    risk.

This helps connect model output with business decision-making.

------------------------------------------------------------------------

## Key Business Insights

The analysis in the notebook identifies several dataset-level patterns:

-   Month-to-month customers have substantially higher churn risk.
-   Shorter-tenure customers are more likely to churn.
-   Electronic-check customers show higher churn in this dataset.
-   Fiber-optic customers show higher churn than DSL customers in this
    dataset.
-   Higher monthly charges are associated with greater churn risk.

**Important:** These are associations observed in the dataset and should
not be interpreted as causal relationships without additional analysis.

------------------------------------------------------------------------

## Business Recommendations

Based on the observed patterns and model output:

1.  Encourage month-to-month customers to consider longer-term
    contracts.
2.  Launch early-retention campaigns for newly acquired customers.
3.  Investigate pricing and service-quality concerns among high-charge
    customers.
4.  Encourage automatic payment methods.
5.  Prioritize high-risk customers for targeted retention outreach.
6.  Use SHAP explanations to support customer-level retention decisions.

------------------------------------------------------------------------

## Streamlit Deployment

The project includes an interactive Streamlit application.

The application allows users to enter customer information such as:

-   Demographics
-   Tenure
-   Phone and internet services
-   Security and support services
-   Contract type
-   Billing method
-   Payment method
-   Monthly charges
-   Total charges

The application then calculates:

-   Churn probability
-   Churn-risk classification
-   Recommended retention action

The app uses the serialized model, preprocessing transformer, learned
decision threshold, and high-value customer threshold.

------------------------------------------------------------------------

## Run the Project Locally

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd customer-churn-prediction
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## Project Structure

``` text
customer-churn-prediction/
│
├── app.py
├── requirements.txt
├── Telco-Customer-Churn.csv
├── customer_churn_xgboost.pkl
├── churn_preprocessor.pkl
├── churn_threshold.pkl
├── high_value_threshold.pkl
├── Customer_Churn_Prediction_End_to_End.ipynb
└── README.md
```

> Keep the filenames in this section synchronized with the files you
> actually upload to GitHub.

------------------------------------------------------------------------

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Scikit-learn
-   XGBoost
-   SHAP
-   Joblib
-   Streamlit

------------------------------------------------------------------------

## Skills Demonstrated

This project demonstrates practical experience with:

-   Data cleaning and validation
-   Exploratory data analysis
-   Feature engineering
-   Classification
-   Model comparison
-   Feature preprocessing pipelines
-   Hyperparameter tuning
-   Cross-validation
-   Threshold optimization
-   Model evaluation
-   Explainable AI
-   Model serialization
-   Streamlit deployment
-   Translating ML results into business recommendations

------------------------------------------------------------------------

## Limitations

-   The dataset represents a specific telecom customer population and
    may not generalize directly to other businesses.
-   The business insights are observational associations, not causal
    conclusions.
-   The model's performance depends on the available customer features
    and historical data.
-   A production system would require ongoing monitoring, retraining,
    data-quality checks, and business validation.

------------------------------------------------------------------------

## Future Improvements

Potential improvements include:

-   Add a live deployed application link.
-   Add screenshots or a short demo GIF.
-   Track model performance over time.
-   Add automated data validation.
-   Add probability calibration.
-   Evaluate additional models and cost-sensitive learning.
-   Add automated retraining pipelines.
-   Monitor model drift after deployment.
-   Add customer-level SHAP explanations directly to the Streamlit
    interface.

------------------------------------------------------------------------

## Author

**Ishu**

This project was developed as an end-to-end machine learning portfolio
project focused on customer churn prediction, model explainability, and
business-oriented decision support.
