import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from preprocess import preprocess_data
from utils import save_model

# 1. Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Preprocess
X = preprocess_data(df.drop("Churn", axis=1))
y = df["Churn"].map({"Yes": 1, "No": 0})

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

pipeline.fit(X_train, y_train)

# 5. Evaluate
preds = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# 6. Save model
save_model(pipeline, "models/churn_pipeline.pkl")

print("Model saved successfully!")
