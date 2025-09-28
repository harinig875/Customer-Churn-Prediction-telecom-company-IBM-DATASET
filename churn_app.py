# churn_app.py
import streamlit as st
import pandas as pd
import joblib

# Load model + scaler + feature names
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("📊 Customer Churn Prediction App")
st.write("Fill in customer details:")

# Input fields (main dataset features)
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", 0, 100, 1)
phone = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protect = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
stream_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 100.0)

if st.button("Predict"):
    # Build input dataframe
    input_dict = {
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "MultipleLines": [phone],
        "InternetService": [internet],
        "OnlineSecurity": [online_sec],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protect],
        "TechSupport": [tech_support],
        "StreamingTV": [stream_tv],
        "StreamingMovies": [stream_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
    }

    input_df = pd.DataFrame(input_dict)

    # Map binary columns
    binary_map = {"Yes": 1, "No": 0, "Female": 0, "Male": 1}
    for col in ["gender", "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"]:
        input_df[col] = input_df[col].map(binary_map)

    # Convert categorical with get_dummies
    input_df = pd.get_dummies(input_df)

    # Align with training features
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer likely to churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer likely to stay (Probability: {prob:.2f})")
