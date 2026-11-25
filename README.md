# ⚡ BCG PowerCo – Customer Churn Prediction

## 📌 Project Overview

This project focuses on identifying customers who are likely to leave PowerCo and understanding the factors contributing to customer churn.

The main objective is to analyse customer characteristics, electricity consumption, pricing information, and customer behaviour to identify churn patterns and develop machine learning models that can predict customers at risk of leaving.

The project covers four major tasks:

1. **Task 1 – Business Understanding:** Understand the business problem and identify the data required for churn analysis.
2. **Task 2 – Exploratory Data Analysis (EDA):** Analyse customer information, consumption patterns, pricing trends, and their relationship with churn.
3. **Task 3 – Feature Engineering:** Create meaningful features using customer behaviour, consumption, pricing, tenure, and financial information.
4. **Task 4 – Model Development:** Train, tune, compare, and evaluate multiple classification models.

A total of 14 experiment notebooks were developed to investigate different algorithms, preprocessing techniques, feature selection strategies, and classification thresholds.

---

## 📁 Project Structure

```text
BCGX-PowerCo-Data-Science-Project/
│
├── task1/
│   └── 1st task wirite mail to client.txt
│
├── task2/
│   ├── Task 2 - eda_starter.ipynb
│   ├── client_data (1).csv
│   ├── price_data (1).csv
│   └── merged_price_right_df.csv
│
├── task3/
│   ├── Task 3 - feature_engineering.ipynb
│   ├── clean_data_after_eda.csv
│   ├── data_for_predictions.csv
│   └── feature_eng_df.csv
│
├── task4/
│   ├── Task 4 - modeling_starter.ipynb
│   ├── data_for_predictions.csv
│   │
│   └── Experiment Notebooks/
│       ├── Exp_1_Baseline_Random_Forest.ipynb
│       ├── Exp_2_Hyperparameter_Tuned_Regularized_RF.ipynb
│       ├── Exp_3_Stronger_Regularization_Random_Forest.ipynb
│       ├── Exp_4_Hyperparameter_Tuned_XGBoost_Classifier.ipynb
│       ├── Exp_5_Strongly_Regularized_XGBoost_Classifier.ipynb
│       ├── Exp_6_LightGBM_Optuna_Optimization.ipynb
│       ├── Exp_7_SVM_Classifier_Validation_Based_Threshold_Optimization.ipynb
│       ├── Exp_8_2_Hyperparameter_Tuned_Regularized_Random_Forest.ipynb
│       ├── Exp_8_4_Hyperparameter_Tuned_XGBoost_Validation_Threshold.ipynb
│       ├── Exp_9_2_Hyperparameter_Tuned_RF_Feature_Specific_Outlier_Capping.ipynb
│       ├── Exp_9_4_Hyperparameter_Tuned_XGBoost_Outlier_Treatment.ipynb
│       ├── Exp_10_2_Hyperparameter_Tuned_RF_Targeted_Median_Imputation.ipynb
│       ├── Exp_11_2_Hyperparameter_Tuned_RF_Median_Imputation_Log_Transformations.ipynb
│       └── Exp_12_Random_Forest_Selected_Feature_Dataset_Extended_Log_Transformations.ipynb
│
├── .gitignore
└── README.md
```

---

## 🎯 Task 1 – Business Understanding

### Business Problem

PowerCo is facing customer churn, where customers are leaving the company and switching to other energy providers.

The objective is to identify why customers are leaving and investigate whether electricity pricing is a major factor behind customer churn.

### Data Requirements

The initial business understanding identified the following data requirements:

1. Customer ID and joining date.
2. Customer location and contract details.
3. Services currently used by customers.
4. Customer retention or churn status.
5. Customer payment and service charges.
6. Competitors' prices for similar services.
7. Customer complaints and resolution details.
8. Customer feedback and satisfaction ratings.
9. Payment history and late payments.
10. Reasons provided by customers for leaving.

### Proposed Analytical Approach

1. Understand the available data and identify relevant variables.
2. Clean the data by handling missing values, duplicates, incorrect values, and outliers.
3. Perform EDA and hypothesis testing to identify churn patterns and investigate pricing relationships.
4. Create useful features such as customer tenure, price differences, and complaint frequency where data is available.
5. Identify major churn indicators and provide business recommendations to improve customer retention.

**Objective:** Understand whether pricing is associated with customer churn or whether other customer-related factors also contribute to the problem.

---

## 📊 Task 2 – Exploratory Data Analysis (EDA)

### Dataset Overview

Two datasets were used for the analysis:

| Dataset | Rows | Columns | Description |
|---|---:|---:|---|
| Client Data | 14,606 | 26 | Customer-level information |
| Price Data | 193,002 | 8 | Monthly electricity pricing information |

The client dataset contains customer characteristics, consumption, subscribed power, margins, product information, and churn status.

The price dataset contains monthly variable and fixed electricity prices across off-peak, peak, and mid-peak periods.

### Target Variable – Churn

The target variable is `churn`.

- `0` – Retained customer
- `1` – Churned customer

| Customer Category | Count | Percentage |
|---|---:|---:|
| Retained | 13,187 | 90.28% |
| Churned | 1,419 | 9.72% |
| Total | 14,606 | 100% |

**Observation:** The dataset is imbalanced, with churned customers representing only 9.72% of the total population.

### Key EDA Findings

**1. Customer Behaviour and Tenure**

- Most customers have approximately 3–6 years of tenure.
- Retained customers generally have longer tenure than churned customers.
- Churn tends to decrease as customer tenure increases.
- Customers with shorter tenure show relatively higher churn rates.

**2. Consumption Patterns**

- Most consumption-related numerical features are positively skewed.
- Several features contain a large number of zero values.
- Consumption features show substantial overlap between retained and churned customers.
- No single consumption feature clearly separates churned customers from retained customers.

**3. Pricing Analysis**

- Churned customers generally have higher average prices across five of the six pricing components.
- Peak and mid-peak pricing show relatively noticeable differences between retained and churned customers.
- Higher pricing groups generally show higher churn rates for several pricing components.
- Average pricing features have weak linear correlations with churn.

**Conclusion:** Electricity pricing may be associated with customer churn, but pricing alone does not fully explain customer behaviour.

**4. Correlation Analysis**

- Gross and net power margins show an almost perfect positive relationship.
- Consumption-related features show strong relationships with each other.
- Tenure has a weak negative correlation with churn.
- Margin-related features show comparatively stronger positive correlations with churn.

**5. Missing Values and Outliers**

- The original client-level dataset contains no missing values.
- Several numerical features are highly right-skewed.
- Potential outliers are present in consumption, pricing, margin, and subscribed power variables.
- Zero values are not automatically treated as missing values because they can represent genuine customer behaviour.

### EDA Conclusion

Customer tenure, pricing patterns, sales channels, product modifications, and margin-related features show differences between retained and churned customers.

However, no single feature clearly explains customer churn. Therefore, multiple customer and pricing variables should be considered during feature engineering and model development.

---

## ⚙️ Task 3 – Feature Engineering

### Objective

Create meaningful variables that capture customer behaviour, consumption patterns, pricing differences, tenure, and financial characteristics.

### 1. Consumption-Based Features

Created features to capture changes and differences in customer consumption.

Examples:

- `cons_last_month_ratio`
- `cons_last_month_diff`
- `forecast_cons_diff`
- `forecast_cons_ratio`
- `forecast_year_ratio`

### 2. Financial and Margin-Based Features

Created features to understand customer profitability and margin relationships.

Examples:

- `margin_per_product`
- `margin_per_power`

Also removed `margin_gross_pow_ele` because it was almost identical to `margin_net_pow_ele` for most customers.

### 3. Price-Based Features

Created relative pricing features to compare electricity prices across different tariff periods.

Examples:

- `off_peak_peak_var_ratio`
- `off_peak_mid_peak_var_ratio`
- `off_peak_peak_fix_ratio`

### 4. Date-Based Features

Created date-related features using customer activation, product modification, contract end, and renewal dates.

Examples:

- `months_activ`
- `months_to_end`
- `months_modif_prod`
- `months_renewal`
- `activation_month`
- `renewal_month`
- `modification_month`

### 5. Zero-Consumption Indicators

Created binary indicators to identify customers with zero consumption.

- `zero_gas_consumption`
- `zero_last_month_consumption`
- `zero_forecast_consumption`

### 6. Price Trend Features

Calculated average prices during the first six months and last six months.

Examples:

- `first_6m_off_peak_var`
- `last_6m_off_peak_var`
- `first_6m_peak_var`
- `last_6m_peak_var`
- `off_peak_6m_change`
- `peak_6m_change`

### 7. Extended Feature Engineering

Further feature engineering was performed through different sections.

| Section | Focus |
|---|---|
| FE-Part A | Price Sensitivity Features |
| FE-Part B | Consumption Behaviour Features |
| FE-Part C | Customer Profitability and Business Value |
| FE-Part D | Customer Interaction Features |

**FE-Part A – Price Sensitivity Features**

Created features to measure price variability, price differences, and relative pricing across tariff periods.

**FE-Part B – Consumption Behaviour Features**

Created features such as:

- `consumption_per_power`
- `gas_consumption_share`

**FE-Part C – Customer Profitability and Business Value**

Created additional features to represent customer profitability and business value.

**FE-Part D – Customer Interaction Features**

Created interaction features using price trends, consumption, customer tenure, margins, subscribed power, and forecast discounts.

### Final Output

The final engineered dataset was saved as:

`feature_eng_df.csv`

This dataset was subsequently used for model development and experimentation.

---

## 🤖 Task 4 – Model Development and Experimentation

### Objective

Develop machine learning classification models to predict whether a customer is likely to churn.

- `0` – Retained
- `1` – Churned

### Data Preparation

The modeling workflow includes:

1. Separating independent features and target variable.
2. Excluding customer ID and raw date columns from model training.
3. Handling infinite values and missing values.
4. Identifying numerical and categorical features.
5. Applying appropriate preprocessing.
6. Splitting data using stratified sampling.
7. Handling class imbalance.
8. Training and evaluating classification models.

### Class Imbalance

Only 9.72% of customers churned, compared with 90.28% who were retained.

Therefore, accuracy alone was not considered sufficient for evaluating model performance.

### Model Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Correctness of predicted churn customers |
| Recall | Ability to identify actual churn customers |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish between classes |
| PR-AUC | Precision-recall performance across different thresholds |

PR-AUC was given particular importance because the dataset is imbalanced.

### Classification Threshold Optimization

The default classification threshold of 0.50 may not provide the best results for an imbalanced dataset.

Different experiments used probability-based threshold analysis to identify suitable thresholds.

The objective was to improve the balance between precision and recall, based on the business requirement.

---

## 🧪 Model Experiments

A total of 14 experiment notebooks were developed.

| Experiment | Model / Approach |
|---|---|
| Exp 1 | Baseline Random Forest |
| Exp 2 | Hyperparameter-Tuned Regularized Random Forest |
| Exp 3 | Stronger Regularization – Random Forest |
| Exp 4 | Hyperparameter-Tuned XGBoost |
| Exp 5 | Strongly Regularized XGBoost |
| Exp 6 | LightGBM with Optuna Optimization |
| Exp 7 | SVM with RBF Kernel |
| Exp 8(2) | Random Forest with Validation-Based Threshold Optimization |
| Exp 8(4) | XGBoost with Validation-Based Threshold Optimization |
| Exp 9(2) | Random Forest with Feature-Specific Outlier Capping |
| Exp 9(4) | XGBoost with Outlier Treatment |
| Exp 10(2) | Random Forest with Targeted Median Imputation |
| Exp 11(2) | Random Forest with Median Imputation and Log Transformations |
| Exp 12 | Random Forest with Selected Features and Extended Log Transformations |

### Experiment Improvements

The experiments progressively investigated:

1. Baseline model performance.
2. Hyperparameter optimization using RandomizedSearchCV.
3. Stronger model regularization.
4. Different classification algorithms.
5. Class imbalance handling.
6. Optuna-based hyperparameter optimization.
7. Validation-based threshold optimization.
8. Feature-specific outlier capping.
9. Targeted median imputation.
10. Log transformations to reduce feature skewness.
11. Selected-feature modeling and extended transformations.

### Hyperparameter Optimization

- RandomizedSearchCV was used for Random Forest and XGBoost experiments.
- Stratified 5-Fold Cross-Validation was used during hyperparameter tuning.
- PR-AUC was used as the primary optimization metric.
- Optuna was used for LightGBM optimization.

---

## 📈 Model Comparison – All Experiments

The following table compares the test performance of all 13 experiment configurations.

| Metric | Exp1 – Baseline RF | Exp2 – Regularized RF | Exp3 – Strong Reg. RF | Exp4 – XGBoost | Exp5 – Strong Reg. XGB | Exp6 – LightGBM | Exp7 – SVM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 84.02% | 87.34% | 84.94% | **89.39%** | 86.69% | 88.02% | 82.75% |
| Precision | 26.72% | 34.86% | 29.26% | **43.09%** | 31.83% | 36.48% | 23.68% |
| Recall | 36.97% | 34.86% | 38.73% | 28.52% | 32.39% | 31.34% | 34.86% |
| F1-score | 31.02% | **34.86%** | 33.33% | 34.32% | 32.11% | 33.71% | 28.21% |
| ROC-AUC | **70.76%** | 70.48% | 69.45% | 69.24% | 69.58% | 69.33% | 66.45% |
| PR-AUC | 32.77% | **35.75%** | 31.71% | 32.15% | 29.42% | 32.20% | 20.13% |
| CV PR-AUC | 25.43% | **30.55%** | 26.15% | 28.42% | 26.38% | 28.63% | 20.85% |

| Metric | Exp8(2) – RF | Exp9(2) – Capped RF | Exp9(4) – Capped XGB | Exp10(2) – RF + Median | Exp11(2) – RF + Log | Exp12 – Selected Feature RF |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 82.07% | 82.20% | 89.05% | 84.80% | 83.54% | 79.71% |
| Precision | 25.71% | 25.72% | 40.62% | 28.49% | 27.04% | 23.32% |
| Recall | 44.72% | 44.01% | 27.46% | 37.32% | 40.85% | **47.54%** |
| F1-score | 32.65% | 32.47% | 32.77% | 32.32% | 32.54% | 31.29% |
| ROC-AUC | 69.98% | 69.10% | 68.66% | 69.23% | 69.43% | 69.27% |
| PR-AUC | 34.26% | 32.43% | 32.57% | 32.75% | 32.23% | 30.88% |
| CV PR-AUC | 27.35% | 26.98% | 28.14% | 27.06% | 26.83% | N/A |

**Key observations:**

- Exp2 achieves the highest PR-AUC (35.75%) and F1-score (34.86%).
- Exp4 achieves the highest accuracy (89.39%) and precision (43.09%).
- Exp1 achieves the highest ROC-AUC (70.76%).
- Exp12 achieves the highest recall (47.54%).
- Exp12 has lower PR-AUC and F1-score than Exp2.
- Exp12 identifies more actual churners but generates considerably more false positives.
- Reducing the feature count from 104 to 61 did not improve overall predictive performance.

*Note: Exp12 does not have a reported CV PR-AUC. Its validation PR-AUC is 26.54%, which is not substituted for CV PR-AUC.*

---

## 🏆 Final Model Selection – Exp2 and Exp12

Two Random Forest models were selected for different business objectives.

- **Exp2 – Regularized Random Forest:** Selected as the best overall model.
- **Exp12 – Selected Feature Random Forest:** Selected as an alternative when identifying more potential churn customers is the priority.

### 1. Best Overall Model – Exp2 Regularized Random Forest

<badge color="success">Best overall model</badge>

Exp2 achieved the strongest overall results based on test PR-AUC and F1-score.

| Metric | Test Performance |
|---|---:|
| Accuracy | 87.34% |
| Precision | **34.86%** |
| Recall | 34.86% |
| F1-score | **34.86%** |
| ROC-AUC | 70.48% |
| PR-AUC | **35.75%** |

**Why Exp2 was selected:**

- Achieves the highest test PR-AUC of **35.75%** among all experiments.
- Achieves the highest F1-score of **34.86%**.
- Provides a better overall balance between identifying churners and limiting false churn alerts.
- Performs better than the other experiments on the two key metrics used for overall churn identification.
- Provides a more balanced choice when retention campaigns have costs and false-positive predictions need to be controlled.

**Business interpretation:**

Exp2 is suitable when PowerCo wants to identify potential churn customers while maintaining a reasonable balance between correctly identifying churners and avoiding unnecessary retention campaigns.

Therefore, Exp2 is the preferred model when overall predictive performance is the main objective.

### 2. Alternative Model – Exp12 Selected Feature Random Forest

Exp12 uses a reduced dataset containing **61 selected numerical features**, compared with the 104-feature dataset used in earlier experiments.

**Exp12 preprocessing and modeling:**

- Selected dataset shape: 14,606 rows and 63 columns.
- Model input features: 61 numerical features.
- Missing values: 0.
- Outlier capping applied to five selected numerical features.
- `log1p(x)` transformations applied to 30 numerical features.
- Separate 64:16:20 Fit-Validation-Test split.
- Selected classification threshold: **0.27**, based on validation F1-score.

**Exp12 Test Performance**

| Metric | Test Performance |
|---|---:|
| Accuracy | 79.71% |
| Precision | 23.32% |
| Recall | **47.54%** |
| F1-score | 31.29% |
| ROC-AUC | 69.27% |
| PR-AUC | 30.88% |

**Confusion Matrix – Threshold 0.27**

| | Predicted Retained | Predicted Churned |
|---|---:|---:|
| Actual Retained | 2,194 | 444 |
| Actual Churned | 149 | 135 |

At the selected threshold:

- Correctly identifies **135 out of 284 actual churners**.
- Misses 149 actual churners.
- Incorrectly flags 444 retained customers as churners.
- Identifies 47.54% of actual churn customers.

**Why Exp12 was selected as an alternative:**

- Achieves the highest recall (**47.54%**) among all experiments.
- Identifies more actual churners than Exp2.
- Uses fewer features, reducing the input feature count from 104 to 61.
- Contains no missing values after preprocessing.
- Uses outlier capping and log transformations to address extreme values and skewness.

However:

- Precision is only 23.32%.
- It generates 444 false positives.
- PR-AUC is 30.88%, lower than Exp2's 35.75%.
- F1-score is 31.29%, lower than Exp2's 34.86%.
- Significant overfitting remains.

**Business interpretation:**

Exp12 is useful when PowerCo wants to identify a larger proportion of customers who may leave, even if more retained customers are incorrectly flagged.

This approach may be appropriate when missing an actual churner is more costly than contacting some customers who would have stayed.

### 3. Why Choose Exp2 or Exp12?

| Business Requirement | Recommended Model | Reason |
|---|---|---|
| Best overall predictive performance | **Exp2** | Highest PR-AUC and F1-score |
| Balanced precision and recall | **Exp2** | Better balance between churn detection and false positives |
| Identify more potential churners | **Exp12** | Highest recall at 47.54% |
| Reduce unnecessary retention campaigns | **Exp2** | Higher precision than Exp12 |
| Prioritize missing fewer actual churners | **Exp12** | Identifies a larger proportion of actual churners |
| Use a reduced feature set | **Exp12** | Uses 61 selected features instead of 104 |

**Simple decision rule:**

- Choose **Exp2** when PowerCo wants a balanced model and needs to control unnecessary customer retention efforts.
- Choose **Exp12** when PowerCo prioritizes identifying as many potential churners as possible and can accommodate additional false-positive alerts.

The choice depends on the business cost of missing a churner versus the cost of contacting a customer who was not going to leave.

### 4. Is Model Performance Satisfactory?

The overall performance is **partially satisfactory**.

Exp2 achieves a PR-AUC of 35.75% and an F1-score of 34.86%. This is considerably better than the 9.72% churn baseline in terms of precision-recall performance.

However, churn identification remains challenging:

- Exp2 has a recall of 34.86%, meaning it misses a substantial proportion of actual churners.
- Exp12 improves recall to 47.54%, but its precision falls to 23.32%.
- False-positive predictions can result in unnecessary retention campaigns and additional business costs.

Overfitting is another concern.

Exp12 achieves a training PR-AUC of 99.74%, compared with a validation PR-AUC of only 26.54%. This substantial gap indicates that the model does not generalize as effectively to unseen customers as it performs on training data.

Further improvements are required before deploying either model in production.

### 5. Best Results Across All Experiments

| Evaluation Metric | Best Experiment | Result |
|---|---|---:|
| Accuracy | Exp4 – XGBoost | 89.39% |
| Precision | Exp4 – XGBoost | 43.09% |
| Recall | Exp12 – Selected Feature RF | 47.54% |
| F1-score | Exp2 – Regularized RF | 34.86% |
| ROC-AUC | Exp1 – Baseline RF | 70.76% |
| PR-AUC | Exp2 – Regularized RF | 35.75% |

No single experiment performs best across every metric. Model selection therefore depends on the business objective rather than accuracy alone.

---

## 📝 Final Conclusion

**Exp2 – Regularized Random Forest is selected as the best overall model for this project.**

It achieves the highest PR-AUC (**35.75%**) and F1-score (**34.86%**) among the experiments, providing the strongest overall balance between churn detection and false-positive predictions.

**Exp12 – Random Forest on Selected Feature Dataset is selected as an alternative model when maximizing churn recall is the priority.**

It identifies 47.54% of actual churners, the highest recall among all experiments, but generates more false-positive predictions and achieves lower overall PR-AUC and F1-score.

Although Exp4 achieves the highest precision and accuracy, Exp2 is preferred because PR-AUC and F1-score provide a more suitable basis for evaluating performance on this imbalanced churn dataset.

Overall:

- **Exp2 is the recommended model candidate for balanced churn prediction.**
- **Exp12 is the alternative for recall-focused churn identification.**

Neither model should yet be considered production-ready. Future improvements should focus on reducing overfitting, improving recall without substantially sacrificing precision, and validating performance on unseen customer data.

**Important evaluation note:** Cross-validation classification metrics were calculated using the default 0.50 threshold, whereas validation and test classification metrics use experiment-specific selected thresholds. These classification metrics should therefore be compared cautiously. PR-AUC is threshold-independent and is more suitable for comparing model discrimination across experiments.

---

## 🛠️ Technologies and Libraries Used

- **Python** – Core programming language.
- **Pandas** – Data manipulation and analysis.
- **NumPy** – Numerical operations.
- **Matplotlib** – Data visualization.
- **Seaborn** – Statistical visualization.
- **Scikit-learn** – Preprocessing, model building, cross-validation, and evaluation.
- **Random Forest** – Ensemble classification algorithm.
- **XGBoost** – Gradient boosting classification.
- **LightGBM** – Gradient boosting framework.
- **Optuna** – Hyperparameter optimization.
- **SVM** – Support Vector Machine classification.
- **Jupyter Notebook** – Development and experimentation environment.

---

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/bhushan-bowlekar/BCGX-PowerCo-Data-Science-Project.git
```

### 2. Navigate to the Project Directory

```bash
cd BCGX-PowerCo-Data-Science-Project
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel xgboost lightgbm optuna
```

### 6. Start Jupyter Notebook

```bash
jupyter notebook
```

### 7. Execute the Notebooks

Run the notebooks in the following order:

1. `task2/Task 2 - eda_starter.ipynb`
2. `task3/Task 3 - feature_engineering.ipynb`
3. `task4/Task 4 - modeling_starter.ipynb`
4. Individual experiment notebooks inside `task4/Experiment Notebooks/`

Ensure the required datasets are available in the expected working directories before executing the notebooks.

---

## 📌 Key Learning Outcomes

Through this project, I learned how to:

- Understand a business problem and translate it into a machine learning task.
- Analyse customer-level and monthly pricing datasets.
- Perform exploratory data analysis and statistical comparisons.
- Identify customer churn patterns and pricing relationships.
- Handle imbalanced classification datasets.
- Perform feature engineering using consumption, pricing, tenure, and financial variables.
- Identify and handle highly correlated features.
- Apply One-Hot Encoding and missing-value imputation.
- Build preprocessing and machine learning pipelines.
- Train Random Forest, XGBoost, LightGBM, and SVM classifiers.
- Perform hyperparameter optimization using RandomizedSearchCV and Optuna.
- Apply Stratified K-Fold Cross-Validation.
- Generate Out-of-Fold predictions.
- Optimize classification thresholds using F1-score.
- Evaluate models using Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
- Apply feature-specific outlier capping and log transformations.
- Compare multiple experiments and select models based on business objectives.

---

## 🔮 Future Improvements

- Perform deeper statistical testing of pricing differences between retained and churned customers.
- Investigate additional customer behaviour and interaction features.
- Improve model generalization and reduce overfitting.
- Explore additional ensemble and boosting approaches.
- Perform model explainability using SHAP.
- Develop a customer churn prediction application.
- Provide actionable customer retention recommendations based on model predictions.

---

## ⚠️ Disclaimer

This project is based on the BCG X Data Science job simulation and is intended for educational and portfolio purposes.

Please follow the applicable BCG X / Forage terms and data-sharing restrictions before publishing supplied datasets, task materials, or other simulation content.

The analysis and model results represent the outcomes of this project and should not be interpreted as proof of causation between individual customer features and churn.
