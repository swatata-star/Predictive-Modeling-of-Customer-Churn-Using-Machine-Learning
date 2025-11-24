import streamlit as st
import pandas as pd
import joblib
from src.preprocess import preprocess_data

st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.title("📊 Customer Churn Prediction")
st.write("Upload customer details to predict churn (Yes/No).")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/churn_pipeline.pkl")

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", df.head())

    # Preprocess
    processed = preprocess_data(df)

    # Predict
    predictions = model.predict(processed)
    df["Churn_Prediction"] = predictions

    st.write("### Prediction Results", df)

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
