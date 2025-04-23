# Credit-Card-Fraud-Detection-Using-Machine-Learning
 1. Project Title
Credit Card Fraud Detection Using Machine Learning Algorithms

2. Problem Statement
Credit card fraud is a growing problem with significant financial implications for both consumers and financial institutions. Fraudulent transactions are often rare and difficult to detect in real time due to their similarity with legitimate ones. Manual review of transactions is impractical due to the volume of data and the speed required for processing.

3. Objective
To build an efficient and accurate machine learning model that can detect fraudulent credit card transactions from a large dataset. The model should be able to:

a. Identify fraudulent transactions with high precision and recall.

b. Handle imbalanced data.

c. Minimize false positives and false negatives.

d. Perform well in a real-time or near real-time environment.

4. Dataset
Source: Kaggle – Credit Card Fraud Detection Dataset

Description:

Total transactions: 284,807

Fraudulent transactions: 492 (approx. 0.172%)

Features: 31 features including:

a. Time, Amount

b. 28 anonymized features labeled V1 to V28 (result of PCA transformation)

c. Class (target variable: 1 = fraud, 0 = legit)

5. Methodology
Step 1: Data Preprocessing
Load and clean the dataset.

Handle missing values (if any).

Split the data into training and testing sets.

Handle class imbalance

Step 2: Exploratory Data Analysis (EDA)
Visualize distribution of classes.

Analyze feature correlations.

Detect outliers and feature importance.

Step 3: Model Selection
Test and compare multiple algorithms, such as:

Logistic Regression

Decision Tree

Random Forest

XGBoost

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Neural Networks (optional)

Step 4: Model Evaluation
Use appropriate metrics due to class imbalance:

Confusion Matrix

Precision

Recall

F1-score

Precision-Recall curve

Matthews Correlation Coefficient

6. Tools & Technologies
Programming Language: Python

Libraries: Pandas, NumPy (data handling), Matplotlib, Seaborn (visualization), Scikit-learn

Environment: Jupyter Notebook / Google Colab / VSCode

7. Challenges
Severe class imbalance

High risk of overfitting

Trade-off between false positives and false negatives

8. Outcome
The model accuracy is high due to class imbalance so, we will have computed precision, recall and f1 score to get a more meaningful understanding. We observe:

a. high frequency that means the model is good at not incorrectly marking legitimate transactions as fraud.
b. high recall means the model is good at identifying fraudulent transactions.
c. F1-score balances these two, indicating a strong performance.
d. MCC of 0.8632 confirms that the model’s predictions are well-aligned with the true outcomes, even in the case of imbalanced data.



