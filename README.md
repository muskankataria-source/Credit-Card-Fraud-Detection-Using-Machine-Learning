# Credit-Card-Fraud-Detection-Using-Machine-Learning
  Project Title: Credit Card Fraud Detection Using Machine Learning Algorithms

  Problem Statement: Credit card fraud is a growing problem with significant financial implications for both consumers and financial institutions. Fraudulent transactions 
  are often rare and difficult to detect in real time due to their similarity with legitimate ones. Manual review of transactions is impractical due to the volume of data 
  and the speed required for processing.
 
  Objective: To build an efficient and accurate machine learning model that can detect fraudulent credit card transactions from a large dataset. The model should be able 
  to: identify fraudulent transactions with high precision and recall, handle imbalanced data, minimize false positives and false negatives, perform well in a real-time or 
  near real-time environment.
 
  Dataset: Source: Kaggle – Credit Card Fraud Detection Dataset, Description: Total transactions: 284,807, Fraudulent transactions: 492 (approx. 0.172%), Features: 31 
  features including: Time and Amount, 28 anonymized features labeled 
  V1 to V28 (result of PCA transformation), Class (target variable: 1 = fraud, 0 = legit)
 
  Methodology: First step is Data Preprocessing which includes: Load and clean the dataset, handle missing values (if any), split the data into training and testing sets, 
  handle class imbalance.
  Second step is Exploratory Data Analysis (EDA) which includes: Visualize distribution of classes, analyze feature correlations, detect outliers and feature importance.
  Third step is Model Selection in which i test and compare multiple algorithms such as: Logistic Regression, Decision Tree, Random Forest, XGBoost, Support Vector Machine 
  (SVM), K-Nearest Neighbors (KNN)
  Fourth step is Model Evaluation in which i use appropriate metrics due to class imbalance: Confusion Matrix, Precision, Recall, F1-score, Precision-Recall curve, Matthews 
  Correlation Coefficient
 
  Tools & Technologies: Programming Language: Python ; Libraries: Pandas, NumPy (data handling), Matplotlib, Seaborn (visualization), Scikit-learn ; Environment: Jupyter 
  Notebook / Google Colab / VSCode
 
  Challenges: Severe class imbalance, high risk of overfitting, trade-off between false positives and false negatives

  Outcome: The model accuracy is high due to class imbalance so, we will have computed precision, recall and f1 score to get a more meaningful understanding. We observe:
  high frequency that means the model is good at not incorrectly marking legitimate transactions as fraud, high recall means the model is good at identifying fraudulent 
  transactions, F1-score balances these two, indicating a strong performance, MCC of 0.8632 confirms that the model’s predictions are well-aligned with the true outcomes, 
  even in the case of imbalanced data.



