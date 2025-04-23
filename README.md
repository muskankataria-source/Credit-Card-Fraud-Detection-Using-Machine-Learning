# Credit-Card-Fraud-Detection-Using-Machine-Learning
 1. Project Title: Credit Card Fraud Detection Using Machine Learning Algorithms

 2. Problem Statement: Credit card fraud is a growing problem with significant financial implications for both consumers and financial institutions. Fraudulent transactions 
    are often rare and difficult to detect in real time due to their similarity with legitimate ones. Manual review of transactions is impractical due to the volume of data 
    and the speed required for processing.
 
 3. Objective: To build an efficient and accurate machine learning model that can detect fraudulent credit card transactions from a large dataset. The model should be able 
    to:

    a. Identify fraudulent transactions with high precision and recall.
    b. Handle imbalanced data.
    c. Minimize false positives and false negatives.
    d. Perform well in a real-time or near real-time environment.
 
 5. Dataset: Source: Kaggle – Credit Card Fraud Detection Dataset
    Description:
    a. Total transactions: 284,807
    b. Fraudulent transactions: 492 (approx. 0.172%)
    c. Features: 31 features including:
    i. Time, Amount
   ii. 28 anonymized features labeled V1 to V28 (result of PCA transformation)
  iii. Class (target variable: 1 = fraud, 0 = legit)
 
 6. Methodology
 
 Step 1: Data Preprocessing
    a. Load and clean the dataset.
    b. Handle missing values (if any).
    c. Split the data into training and testing sets.
    d. Handle class imbalance
 
 Step 2: Exploratory Data Analysis (EDA)
    a. Visualize distribution of classes.
    b. Analyze feature correlations.
    c. Detect outliers and feature importance.
 
 Step 3: Model Selection
 Test and compare multiple algorithms, such as: Logistic Regression, Decision Tree, Random Forest, XGBoost, Support Vector Machine (SVM), K-Nearest Neighbors (KNN)
 Step 4: Model Evaluation
 Use appropriate metrics due to class imbalance: Confusion Matrix, Precision, Recall, F1-score, Precision-Recall curve, Matthews Correlation Coefficient
 
 7. Tools & Technologies
 Programming Language: Python
 Libraries: Pandas, NumPy (data handling), Matplotlib, Seaborn (visualization), Scikit-learn
 Environment: Jupyter Notebook / Google Colab / VSCode
 
 8. Challenges
    a. Severe class imbalance
    b. High risk of overfitting
    c. Trade-off between false positives and false negatives

 9. Outcome
 The model accuracy is high due to class imbalance so, we will have computed precision, recall and f1 score to get a more meaningful understanding. We observe:
    a. high frequency that means the model is good at not incorrectly marking legitimate transactions as fraud.
    b. high recall means the model is good at identifying fraudulent transactions.
    c. F1-score balances these two, indicating a strong performance.
    d. MCC of 0.8632 confirms that the model’s predictions are well-aligned with the true outcomes, even in the case of imbalanced data.



