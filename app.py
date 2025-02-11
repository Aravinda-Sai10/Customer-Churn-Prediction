import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Datasetpython -m streamlit run app.py

@st.cache_data
def load_data():
    df = pd.read_csv("Churn_Modelling.csv")
    return df

df = load_data()

# Preprocess Data
def preprocess_data(df):
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)  # Drop unnecessary columns
    
    # Encode categorical variables
    label_enc = LabelEncoder()
    df["Gender"] = label_enc.fit_transform(df["Gender"])
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    # Split into X and y
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("Customer Churn Prediction App 📊")

st.sidebar.header("Enter Customer Details")

# User Inputs
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, value=600)
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance", min_value=0.0, max_value=300000.0, value=50000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=0)
has_cr_card = st.sidebar.radio("Has Credit Card?", [0, 1])
is_active = st.sidebar.radio("Is Active Member?", [0, 1])
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])

# Convert Inputs to Model Format
gender = 1 if gender == "Male" else 0
geo_france = 1 if geography == "France" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

user_input = np.array([[credit_score, age, tenure, balance, num_products, has_cr_card, is_active, estimated_salary, gender, geo_germany, geo_spain]])
user_input = scaler.transform(user_input)  # Scale input data

# Make Prediction
prediction = model.predict(user_input)
result = "Customer will Churn ❌" if prediction[0] == 1 else "Customer will Stay ✅"

# Display Results
st.subheader("Prediction Result:")
st.write(result)
st.write(f"**Model Accuracy:** {accuracy:.2%}")
