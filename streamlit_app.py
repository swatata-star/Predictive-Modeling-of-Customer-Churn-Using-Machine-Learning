import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction App (Upload Excel)")

# Load model
try:
    model = joblib.load("churn_pipeline.pkl")
except:
    st.error("Model file 'churn_pipeline.pkl' not found in repo.")
    st.stop()

uploaded = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded:
    # Read file
    if uploaded.name.endswith("csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Choose filter column
    col = st.selectbox("Select a column to filter rows", df.columns)
    unique_vals = df[col].unique()
    val = st.selectbox("Select value", unique_vals)

    filtered_df = df[df[col] == val]
    st.subheader("Filtered Rows")
    st.dataframe(filtered_df)

    if st.button("Predict churn for filtered rows"):
        try:
            # Prepare features by dropping unwanted columns
            drop_cols = ["customerID", "Churn"]
            X = filtered_df.drop(columns=[c for c in drop_cols if c in filtered_df.columns], errors='ignore')

            # Predict
            preds = model.predict(X)(filtered_df)
            st.write("Predictions (0 = retain, 1 = churn):")
            st.dataframe(pd.DataFrame({"Prediction": preds}))
        except Exception as e:
            st.error("Prediction failed. Check feature names and model compatibility.")
            st.exception(e)
