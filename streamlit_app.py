# ====== Robust prediction wrapper - drop into streamlit_app.py ======
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import os

# Load model
MODEL_PATHS = ["churn_pipeline.pkl", "models/churn_pipeline.pkl"]
model = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        model = joblib.load(p)
        break

if model is None:
    st.error("Model file not found: upload churn_pipeline.pkl to the repo.")
    st.stop()

# If you saved feature_names separately, try loading them
FEATURE_NAMES_PATHS = ["feature_names.pkl", "models/feature_names.pkl"]
feature_names_from_file = None
for p in FEATURE_NAMES_PATHS:
    if os.path.exists(p):
        try:
            feature_names_from_file = joblib.load(p)
        except Exception:
            feature_names_from_file = None
        break

def get_expected_feature_names(model):
    """
    Try multiple ways to obtain the feature names the model expects:
    1. feature_names_from_file (if saved separately)
    2. model.feature_names_in_  (scikit-learn >=0.24 on fitted estimators)
    3. If pipeline with a preprocessor/ColumnTransformer, try get_feature_names_out
    4. Fallback: None
    """
    if feature_names_from_file:
        return list(feature_names_from_file)

    # 1) direct attribute
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass

    # 2) pipeline -> try to extract from last transformer or named_steps
    # Many pipelines keep a 'named_steps' dict
    try:
        # If model is a Pipeline, look for a preprocessing step using get_feature_names_out
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
    except Exception:
        Pipeline = None
        ColumnTransformer = None

    try:
        # If pipeline, try to find a column transformer with get_feature_names_out
        if hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():
                # ColumnTransformer
                if ColumnTransformer and isinstance(step, ColumnTransformer):
                    try:
                        cols = step.get_feature_names_out()
                        return list(cols)
                    except Exception:
                        pass
                # Any transformer offering get_feature_names_out
                if hasattr(step, "get_feature_names_out"):
                    try:
                        cols = step.get_feature_names_out()
                        return list(cols)
                    except Exception:
                        pass
        # If pipeline, maybe final estimator has feature_names_in_
        if hasattr(model, "steps"):
            for name, step in model.steps:
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
    except Exception:
        pass

    # 3) last resort: check model's attributes
    try:
        if hasattr(model, "get_feature_names_out"):
            return list(model.get_feature_names_out())
    except Exception:
        pass

    return None

expected_cols = get_expected_feature_names(model)

# Suppose you gather user inputs into an ordered dict input_data (keys are column names used in your form)
# Example:
# input_data = {
#   "gender": "Male",
#   "SeniorCitizen": 0,
#   "tenure": 12,
#   "MonthlyCharges": 70.5,
#   ...
# }

# Convert that input dict to dataframe (replace with your code that builds input_data)
# For Streamlit: build the dict from st.selectbox / st.number_input etc.
# Example for illustration only:
# input_data = {k: v for k, v in user_inputs.items()}

# Then:
def prepare_input_dataframe(input_data: dict, expected_columns):
    """
    - Ensure dataframe columns exactly match expected_columns.
    - If expected_columns is None: return a single-row df from input_data
      (the model might accept it but you risk mismatch).
    - Fill missing columns with zeros or sensible defaults.
    """
    df_input = pd.DataFrame([input_data])

    # If expected column list is available -> reindex to that exact order
    if expected_columns is not None:
        # Convert columns in input to same dtype if needed (e.g., numeric columns)
        # Fill missing columns with 0 (numeric) or "" (string) or the most sensible default.
        # Here we use 0 for numeric-like and empty string for object-like columns:
        # We'll just fill with 0 to be safe; you can customize for your features.
        df_input = df_input.reindex(columns=expected_columns, fill_value=0)
    else:
        # No expected columns known -> keep user-provided order
        pass

    return df_input

# ===== Example usage in your predict button callback =====
if st.button("Predict churn"):
    # Build input_data from your streamlit inputs — replace this with your real input collection
    # For example:
    # input_data = {
    #   "gender": gender_value,
    #   "SeniorCitizen": senior_value,
    #   ... etc
    # }
    try:
        # assume input_data exists here from your UI
        input_data  # noqa
    except NameError:
        st.error("input_data not found — ensure you build it from the form inputs.")
        st.stop()

    df_for_pred = prepare_input_dataframe(input_data, expected_cols)

    # Some models expect numeric types — coerce where possible
    # If your pipeline contains preprocessing (ColumnTransformer), this may not be necessary.
    try:
        # Attempt prediction
        prediction = model.predict(df_for_pred)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_for_pred)[:, 1]  # probability of churn class 1
        # If prediction is array-like with one element
        pred_value = int(prediction[0]) if hasattr(prediction, "__len__") else int(prediction)
        st.write("Prediction (0 = retained, 1 = churn):", pred_value)
        if proba is not None:
            st.write("Predicted churn probability (class=1):", float(proba[0]))
    except ValueError as e:
        # Capture feature-name mismatch and show helpful diagnostic in logs
        st.error("Prediction failed — feature name mismatch or wrong input shape.")
        st.exception(e)
        # Helpful debugging info for logs:
        st.write("Expected columns (from model):", expected_cols)
        st.write("Input columns you provided:", list(pd.DataFrame([input_data]).columns))
        st.stop()
    except Exception as e:
        st.error("Prediction failed with unexpected error.")
        st.exception(e)
        st.stop()

# ===== end of block =====
