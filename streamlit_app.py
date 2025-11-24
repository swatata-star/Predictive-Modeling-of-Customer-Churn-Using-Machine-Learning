import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.title("📊 Customer Churn Prediction App")

@st.cache_resource
def load_model():
    try:
        return joblib.load("churn_pipeline.pkl")
    except Exception as e:
        st.error("❌ Could not load model file. Make sure 'churn_pipeline.pkl' is in the same folder as streamlit_app.py")
        st.stop()

model = load_model()

st.write("Upload a CSV file to get churn predictions.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df.head())

    # Predict
    try:
        predictions = model.predict(df)
        df["Churn_Prediction"] = predictions

        st.write("### Results")
        st.dataframe(df)

        st.download_button(
            "Download Predictions",
            df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv"
        )

    except Exception as e:
        st.error("Prediction failed. Your input CSV must match model training schema.")
        st.write(e)
