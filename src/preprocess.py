import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode binary columns
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "Female": 1, "Male": 0})

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [col for col in cat_cols if col not in ["customerID"]]

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df
