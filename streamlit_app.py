import streamlit as st
import joblib
import os
import pandas as pd

# ------------------------------
# MODEL LOADER (Robust Version)
# ------------------------------
@st.cache_resource
def load_model():
    possible_paths = [
        "churn_pipeline.pkl",
        "models/churn_pipeline.pkl"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                st.error(f"❌ Error loading model from {path}: {e}")
                return None

    st.error("❌ Model file NOT FOUND.\n\nExpected one of these paths:\n- churn_pipeline.pkl\n- models/churn_pipeline.pkl")
    return None


# Load the model
model = load_model()

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer information below:")

# ------------------------------
# SIDEBAR INPUT FEATURES
# ------------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

# ------------------------------
# PREDICTION
# ------------------------------
if st.button("Predict Churn"):
    if model is None:
        st.error("❌ Model not loaded. Please check your file path.")
    else:
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "InternetService": internet_service,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.success(f"### 🔍 Churn Prediction: **{'Yes' if prediction==1 else 'No'}**")
            st.info(f"### 📈 Probability of Churn: **{probability:.2f}**")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
