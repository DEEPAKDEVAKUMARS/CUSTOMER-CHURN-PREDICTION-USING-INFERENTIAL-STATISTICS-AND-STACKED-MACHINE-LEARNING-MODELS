# CUSTOMER-CHURN-PREDICTION-USING-INFERENTIAL-STATISTICS-AND-STACKED-MACHINE-LEARNING-MODELS

Project Overview

This project aims to predict customer churn in the telecommunications industry using a combination of statistical analysis (ANOVA and Chi-square tests) and machine learning ensembles.
The model identifies customers who are most likely to discontinue services, allowing the company to take preventive actions and improve customer retention.

The dataset used is from the Telco Customer Churn dataset available on Kaggle

Objectives

To perform data preprocessing and handle categorical and numerical variables appropriately.

To apply ANOVA and Chi-Square tests for feature significance.

To build predictive models including:

CHAID-style Decision Tree

Logistic Regression

Random Forest

LightGBM

Stacked Ensemble (Meta-Learners: Logistic & LightGBM)

To evaluate the models based on ROC-AUC, Recall, F1-Score, and interpret the results.

Methodology

Data Preprocessing:
Missing values were handled, categorical columns were encoded, and numerical columns standardized.
The cleaned dataset was saved as churn_cleaned.csv.

Feature Significance Analysis:

ANOVA was used for numeric features.

Chi-Square test was applied for categorical features.

Only statistically significant predictors were retained for modeling.

Modeling:

CHAID Decision Tree

Logistic Regression

Random Forest

LightGBM

Stacked Meta-Learning Ensemble

Hyperparameter Optimization:
Optuna was used for fine-tuning Logistic and LightGBM meta-learners.

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Gains/Lift Analysis

Results Summary
Model	ROC-AUC	Accuracy	Recall	F1-Score
CHAID Decision Tree	0.8295	0.7197	0.8155	0.6070
Logistic Regression	0.8415	0.7381	0.7834	0.6136
Random Forest	0.8444	0.7551	0.7888	0.6310
Stacked (Meta-Logistic)	0.8453	0.7566	0.7914	0.6332
Stacked (Meta-LightGBM, Tuned)	0.8493	0.7991	0.5053	0.5719

Observation:
The tuned Meta-LightGBM ensemble achieved the best ROC-AUC of 0.8493, while Meta-Logistic offered a higher recall, ideal for churn-sensitive use cases.

Business Insight

The model reveals that contract type, tenure, and monthly charges are major churn determinants.
Customers with month-to-month contracts, high monthly bills, and no technical support are at the highest risk of leaving.
The predictions can be integrated into a CRM system to enable proactive retention strategies.

Tools and Technologies

Python 3.12

Scikit-Learn

LightGBM

Optuna

Pandas / NumPy

Matplotlib / Seaborn

Streamlit (optional deployment)

How to Run the Project
# Clone the repository
git clone https://github.com/<your-username>/Telco-Customer-Churn-Analysis.git
cd Telco-Customer-Churn-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Customer_Churn_Prediction.ipynb


For Streamlit deployment (optional):

streamlit run streamlit_app.py

📁 Repository Structure
Telco-Customer-Churn-Analysis/
│
├── data/
│   ├── cleaned/
│   │   └── churn_cleaned.csv
│   └── splits/
│       ├── train.csv
│       └── test.csv
│
├── models/
│   ├── random_forest.joblib
│   ├── meta_logistic_tuned.joblib
│   ├── meta_lgb_tuned.joblib
│   └── preprocessor.joblib
│
├── figures/
│   ├── anova_numeric_pvalues.png
│   ├── chi2_categorical_pvalues.png
│   ├── meta_ensemble_comparison.png
│   └── optuna_meta_lgb_param_importance.png
│
├── Customer_Churn_Prediction.ipynb
├── streamlit_app.py
└── README.md
