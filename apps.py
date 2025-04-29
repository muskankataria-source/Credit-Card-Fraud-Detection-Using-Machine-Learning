import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
import streamlit as st

# Load the data
data = pd.read_csv("creditcard.csv")

fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))

x = data.drop(['Class'], axis = 1)
y = data["Class"]
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.2, random_state = 42)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train, y_train)

# predictions
yPred = rfc.predict(x_test)

# Evaluation
accuracy = accuracy_score(y_test, yPred)
precision = precision_score(y_test, yPred)
recall = recall_score(y_test, yPred)
f1 = f1_score(y_test, yPred)
mcc = matthews_corrcoef(y_test, yPred)

# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = rfc.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")

