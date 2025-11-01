# ==============================
# Customer Churn Prediction - Final Version
# Based on your notebook + bug fixes
# ==============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("data/churn.csv")
print("✅ Data Loaded Successfully:", df.shape)
print(df.info())

# ==============================
# 2. Data Cleaning
# ==============================

# Convert TotalCharges to numeric, handle missing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

# Drop customerID (not useful)
df.drop("customerID", axis=1, inplace=True)

# Replace "No internet service" and "No phone service" with "No"
cols_replace = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"
]
for col in cols_replace:
    df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

# Encode binary categorical columns (Yes/No)
binary_cols = [col for col in df.columns if df[col].nunique() == 2]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# Encode remaining categorical columns safely
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Fill any remaining NaN just in case
print("🚨 Missing values before fix:", df.isna().sum().sum())
df = df.fillna(0)
print("✅ Missing values after fix:", df.isna().sum().sum())

# ==============================
# 3. Split Dataset
# ==============================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4. Train Models
# ==============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42
    ),
}

os.makedirs("models", exist_ok=True)
results = {}

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"✅ {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    joblib.dump(model, f"models/{name.replace(' ', '_')}.pkl")

# Save feature names for app usage
joblib.dump(list(X.columns), "models/features.pkl")

# ==============================
# 5. Visualization
# ==============================
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ==============================
# 6. Save Results
# ==============================
results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
results_df.to_csv("models/model_performance.csv")
print("\n📊 Model Training Completed. Results saved to models/model_performance.csv")
