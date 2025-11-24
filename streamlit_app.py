import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# ============================
#  Load Model Safely
# ============================

@st.cache_resource
def load_model():
    possible_paths = [
        "churn_pipeline.pkl",
        "./churn_pipeline.pkl",
        "models/churn_pipeline.pkl",
        "./models/churn_pipeline.pkl"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return joblib.load(path)

    st.error("❌ Model file NOT FOUND.\n\nExpected one of these paths:\n\n" + "\n".join(possible_paths))
    st.stop()

model = load_model()

# ============================
#   Streamlit UI
# ============================

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict whether they are likely to churn.")

# Sidebar inputs
st.sidebar.header("Customer Details")

tenure = st.sidebar.number_input("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 500.0)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
)

if st.sidebar.button("Predict Churn"):
    # Create input dataframe
    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "gender": [gender],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone_service],
        "InternetService": [internet_service],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
    })

    # Predict
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction Result")
    
    if prediction == 1:
        st.error(f"⚠ High Risk of Churn — Probability: {proba:.2f}")
    else:
        st.success(f"✔ Customer is Not Likely to Churn — Probability: {proba:.2f}")
