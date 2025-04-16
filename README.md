# Credit-Risk-Analysis
Project Description
The project focuses on developing a predictive model to assess credit risk for borrowers based on financial and demographic features. The dataset contains 150,000 records with 11 features, including credit utilization, age, income, debt ratios, and past delinquency metrics. The target variable, SeriousDlqin2yrs, is binary (0 for no delinquency, 1 for delinquency), and the dataset is imbalanced, with only a small proportion of borrowers labeled as delinquent.

The notebook implements the following key steps:

Data Preprocessing: Handles missing values, scales features, and addresses class imbalance.
Feature Engineering: Creates new features to enhance model performance.
Model Training: Trains multiple machine learning models (Random Forest, Gradient Boosting, XGBoost) and evaluates their performance.
Hyperparameter Tuning: Optimizes the Random Forest model using Randomized Search.
Visualization: Visualizes model performance using confusion matrices.
Prediction Function: Defines a function for making predictions on new data (though not fully implemented in the notebook).
The project leverages Python libraries such as pandas, numpy, scikit-learn, imblearn, xgboost, matplotlib, and seaborn, and is executed in a GPU-accelerated environment (Google Colab with T4 GPU).

Dataset Details
Source: The dataset (cs-training.csv) is likely from the Kaggle "Give Me Some Credit" competition, designed for credit risk modeling.
Size: 150,000 rows and 12 columns (including an index column Unnamed: 0).
Features:
SeriousDlqin2yrs: Target variable (0 = no delinquency, 1 = delinquency).
RevolvingUtilizationOfUnsecuredLines: Ratio of credit card balances to credit limits.
age: Age of the borrower.
NumberOfTime30-59DaysPastDueNotWorse: Number of times borrower was 30–59 days past due.
DebtRatio: Ratio of total debt to monthly income.
MonthlyIncome: Monthly income of the borrower.
NumberOfOpenCreditLinesAndLoans: Number of open credit lines and loans.
NumberOfTimes90DaysLate: Number of times borrower was 90+ days past due.
NumberRealEstateLoansOrLines: Number of real estate loans or lines.
NumberOfTime60-89DaysPastDueNotWorse: Number of times borrower was 60–89 days past due.
NumberOfDependents: Number of dependents.
Challenges:
Missing Values: MonthlyIncome (29,731 missing) and NumberOfDependents (3,924 missing).
Class Imbalance: The target variable is highly imbalanced, with a small percentage of delinquent cases (typically ~6–7% based on similar datasets).
Feature Scaling: Features have different scales (e.g., MonthlyIncome vs. age), requiring standardization.
Implementation Details
The notebook is structured into five main code cells, each addressing a specific part of the machine learning pipeline. Below is a step-by-step breakdown of the implementation:

Data Loading and Preprocessing:
Libraries: Imports pandas, numpy, scikit-learn, imblearn, matplotlib, and seaborn.
Data Loading: Loads the dataset using pd.read_csv('/content/cs-training.csv').
Exploratory Analysis:
Displays the first five rows, data types, and missing value counts.
Identifies missing values in MonthlyIncome and NumberOfDependents.
Cleaning:
Drops the redundant Unnamed: 0 column.
Imputes missing values in MonthlyIncome and NumberOfDependents with their respective medians.
Issue: Uses deprecated inplace=True with fillna, triggering FutureWarning. Should be updated to df['column'] = df['column'].fillna(median).
Feature and Target Definition:
Features (X): All columns except SeriousDlqin2yrs.
Target (y): SeriousDlqin2yrs.
Feature Scaling: Applies StandardScaler to standardize features.
Class Imbalance Handling: Uses SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset, generating synthetic samples for the minority class (delinquent borrowers).
Train-Test Split: Splits the resampled data into 80% training and 20% testing sets (random_state=42).
Output: Confirms preprocessing completion with no missing values.
Feature Engineering:
Converts resampled data back to a DataFrame for feature creation.
Creates four new features:
Debt_to_Income_Ratio: DebtRatio divided by MonthlyIncome (adds 1 to avoid division by zero).
Is_Late_Payer: Binary indicator (1 if any past-due counts > 0, else 0).
Has_Dependents: Binary indicator (1 if NumberOfDependents > 0, else 0).
Total_Past_Dues: Sum of all past-due counts (30–59, 60–89, and 90+ days).
Re-applies StandardScaler to the engineered features.
Performs a new train-test split on the final feature set.
Output: Confirms feature engineering completion.
Model Training and Evaluation:
Trains three models:
Random Forest: RandomForestClassifier(random_state=42).
Gradient Boosting: GradientBoostingClassifier(random_state=42).
XGBoost: XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42).
Defines an evaluate_model function to compute:
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Results:
Random Forest:
Accuracy: 93.95%
Precision: 94.88%
Recall: 92.94%
F1 Score: 93.90%
Confusion Matrix: [[26518, 1407], [1981, 26084]]
Gradient Boosting:
Accuracy: 86.29%
Precision: 87.38%
Recall: 84.90%
F1 Score: 86.12%
Confusion Matrix: [[24484, 3441], [4237, 23828]]
XGBoost:
Accuracy: 93.42%
Precision: 96.21%
Recall: 90.43%
F1 Score: 93.23%
Confusion Matrix: [[26926, 999], [2687, 25378]]
Issue: XGBoost raises a warning about the unused use_label_encoder parameter, which can be removed in newer versions.
Hyperparameter Tuning:
Performs RandomizedSearchCV on the Random Forest model to optimize:
n_estimators: [100, 200, 300]
max_depth: [10, 20, 30, None]
min_samples_split: [2, 5, 10]
min_samples_leaf: [1, 2, 4]
bootstrap: [True, False]
Uses 3-fold cross-validation, 10 iterations, and F1 score as the scoring metric.
Trains the best model (best_rf) and evaluates it:
Tuned Random Forest:
Accuracy: 94.50%
Precision: 95.64%
Recall: 93.27%
F1 Score: 94.44%
Confusion Matrix: [[26732, 1193], [1889, 26176]]
Issue: A joblib warning indicates a potential memory leak or timeout during parallel processing, which could be mitigated by adjusting n_jobs or resources.
Visualization:
Defines a plot_conf_matrix function to plot the confusion matrix using seaborn heatmap.
Visualizes the confusion matrix for the tuned Random Forest model.
Labels: "Good Credit (0)" and "Bad Credit (1)".
Issue: The notebook includes a minor rendering warning for the Unicode character 📉 in the plot title, which does not affect functionality.
Prediction Function (Incomplete):
Defines a predict_credit_risk function intended to make predictions on new data using a stack_model.
Issue: The stack_model is not defined in the notebook, indicating an incomplete implementation. The function expects input data as a NumPy array or DataFrame row but lacks example usage.
Achievements of the Project
The project achieves several key milestones in building a robust credit risk prediction system:

Effective Data Preprocessing:
Successfully handled missing values in MonthlyIncome and NumberOfDependents using median imputation.
Addressed class imbalance using SMOTE, ensuring balanced training data for better model performance on the minority class (delinquent borrowers).
Standardized features to ensure compatibility with machine learning algorithms.
Smart Feature Engineering:
Created four meaningful features (Debt_to_Income_Ratio, Is_Late_Payer, Has_Dependents, Total_Past_Dues) that capture financial behavior and risk factors, likely improving model interpretability and performance.
Avoided dropping original features, preserving information for model training.
High-Performing Models:
Trained and compared three state-of-the-art models (Random Forest, Gradient Boosting, XGBoost), with Random Forest and XGBoost achieving over 93% accuracy and F1 scores.
The tuned Random Forest model achieved the highest performance:
Accuracy: 94.50%
F1 Score: 94.44%
High precision (95.64%) and recall (93.27%), indicating strong performance in identifying both delinquent and non-delinquent borrowers.
Demonstrated the ability to handle imbalanced data effectively, with good recall for the minority class.
Hyperparameter Optimization:
Improved Random Forest performance through RandomizedSearchCV, increasing accuracy and F1 score compared to the default model.
Balanced computational efficiency with model improvement by limiting search to 10 iterations and 3-fold cross-validation.
Clear Visualization:
Provided a clear visualization of the tuned Random Forest’s confusion matrix, aiding in model evaluation and interpretability.
Correctly labeled axes and used a professional heatmap format for stakeholder communication.
Scalable Pipeline:
Built a modular pipeline with reusable functions (evaluate_model, plot_conf_matrix) that can be extended to other datasets or models.
Laid the foundation for real-world deployment with a prediction function (though incomplete).
Limitations and Areas for Improvement
Despite its achievements, the project has some limitations:

Incomplete Prediction Function:
The predict_credit_risk function references an undefined stack_model, indicating a missing stacking ensemble or implementation error.
No example usage or sample input data is provided, limiting practical applicability.
Deprecated Code:
The use of inplace=True in fillna triggers warnings and should be updated for compatibility with future Pandas versions (e.g., df['column'] = df['column'].fillna(median)).
The XGBoost use_label_encoder parameter is unnecessary and should be removed.
Limited Model Exploration:
Only three models are tested, and no ensemble methods (e.g., stacking or voting) are implemented, despite the reference to stack_model.
Hyperparameter tuning is limited to Random Forest; tuning XGBoost or Gradient Boosting could further improve performance.
Visualization Scope:
Only the confusion matrix is visualized. Additional plots (e.g., feature importance, ROC curves, precision-recall curves) would enhance model evaluation.
Feature Engineering Scope:
While the engineered features are meaningful, additional features (e.g., interaction terms, credit utilization thresholds) could be explored.
No feature selection is performed, which could reduce model complexity.
Computational Warnings:
The joblib warning during hyperparameter tuning suggests potential resource issues, which could be addressed by optimizing parallel processing or reducing the parameter grid.
Recommendations for Enhancement
To improve the project, consider the following:

Complete the Prediction Function:
Implement a stacking ensemble (e.g., combining Random Forest, XGBoost, and Gradient Boosting) and update the predict_credit_risk function.
Provide example input data and test the function for real-world applicability.
Update Deprecated Code:
Replace inplace=True with explicit assignments in fillna.
Remove the use_label_encoder parameter from XGBoost.
Expand Model Exploration:
Test additional models (e.g., Logistic Regression, LightGBM) or ensemble methods.
Tune hyperparameters for XGBoost and Gradient Boosting to maximize performance.
Enhance Visualizations:
Plot feature importance to identify key predictors.
Include ROC and precision-recall curves to evaluate model performance across thresholds.
Visualize class distribution before and after SMOTE.
Advanced Feature Engineering:
Explore interaction terms (e.g., age × DebtRatio) or categorical bins for continuous features.
Apply feature selection (e.g., Recursive Feature Elimination) to reduce dimensionality.
Address Computational Issues:
Optimize RandomizedSearchCV by reducing n_jobs or using a smaller parameter grid.
Consider using GridSearchCV for critical parameters or distributed computing for large datasets.
Conclusion
The Credit Risk Analysis project successfully builds a robust machine learning pipeline for predicting borrower delinquency using a real-world financial dataset. It achieves high performance (94.50% accuracy with tuned Random Forest), effectively handles class imbalance with SMOTE, and incorporates smart feature engineering to enhance model interpretability. The modular design and clear visualizations make it a strong foundation for credit risk modeling.

However, the project is incomplete due to the undefined stack_model and lacks advanced visualizations or model ensembling. By addressing these limitations and incorporating the recommended enhancements, the project can be transformed into a production-ready credit risk assessment tool suitable for financial institutions. The achievements demonstrate strong data science proficiency, while the identified gaps provide opportunities for further refinement.
