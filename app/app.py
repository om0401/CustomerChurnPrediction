import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ==============================
# Page config
# ==============================
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.title("📈 Customer Churn Prediction")
st.markdown("Interactive app that predicts customer churn using multiple trained models.")

# ==============================
# Locate models directory
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder containing this file (app/)
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))  # ../models

if not os.path.isdir(MODELS_DIR):
    st.error(f"Models folder not found at expected location: {MODELS_DIR}\n\nMake sure your repository has a `models` folder at project root containing the .pkl files.")
    st.stop()

# ==============================
# Expected filenames
# ==============================
expected_files = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Random Forest": "Random_Forest.pkl",
    "Gradient Boosting": "Gradient_Boosting.pkl",
    "XGBoost": "XGBoost.pkl",
    "features": "features.pkl"
}

# Verify files exist
missing = [fname for fname in expected_files.values() if not os.path.isfile(os.path.join(MODELS_DIR, fname))]
if missing:
    st.error("Missing model files in models directory: " + ", ".join(missing))
    st.stop()

# ==============================
# Load models and features
# ==============================
models = {}
try:
    for display_name, fname in expected_files.items():
        if display_name == "features":
            continue
        models[display_name] = joblib.load(os.path.join(MODELS_DIR, fname))
    features = joblib.load(os.path.join(MODELS_DIR, expected_files["features"]))
except Exception as e:
    st.error("Error loading models/features: " + str(e))
    st.stop()

# ==============================
# Sidebar inputs (user-friendly)
# ==============================
st.sidebar.header("🧾 Input Customer Details")
st.sidebar.markdown("Provide customer details below and click **Predict Churn**.")

input_data = {}

# Demographics
input_data["gender"] = st.sidebar.selectbox("Gender", [1, 0], index=1, format_func=lambda x: "Male" if x == 1 else "Female")
input_data["SeniorCitizen"] = st.sidebar.selectbox("Senior Citizen", [0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
input_data["Partner"] = st.sidebar.selectbox("Partner", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
input_data["Dependents"] = st.sidebar.selectbox("Dependents", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")

# Account
input_data["tenure"] = st.sidebar.slider("Tenure (months)", min_value=0, max_value=72, value=12)

# Phone
input_data["PhoneService"] = st.sidebar.selectbox("Phone Service", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
input_data["MultipleLines"] = st.sidebar.selectbox("Multiple Lines", [0, 1, 2], index=0, format_func=lambda x: ["No", "Yes", "No phone service"][x])

# Internet & related
input_data["InternetService"] = st.sidebar.selectbox("Internet Service", [0, 1, 2], index=0, format_func=lambda x: ["DSL", "Fiber optic", "No"][x])
input_data["OnlineSecurity"] = st.sidebar.selectbox("Online Security", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
input_data["OnlineBackup"] = st.sidebar.selectbox("Online Backup", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
input_data["DeviceProtection"] = st.sidebar.selectbox("Device Protection", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
input_data["TechSupport"] = st.sidebar.selectbox("Tech Support", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
input_data["StreamingTV"] = st.sidebar.selectbox("Streaming TV", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
input_data["StreamingMovies"] = st.sidebar.selectbox("Streaming Movies", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")

# Contract & billing
input_data["Contract"] = st.sidebar.selectbox("Contract Type", [0, 1, 2], index=0, format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
input_data["PaperlessBilling"] = st.sidebar.selectbox("Paperless Billing", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
input_data["PaymentMethod"] = st.sidebar.selectbox("Payment Method", [0, 1, 2, 3], index=0, format_func=lambda x: ["Electronic check", "Mailed check", "Bank transfer (auto)", "Credit card (auto)"][x])

# Charges
input_data["MonthlyCharges"] = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=10000.0, value=70.0, step=1.0)
input_data["TotalCharges"] = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=100000.0, value=2000.0, step=10.0)

st.sidebar.markdown("---")
st.sidebar.info("When ready click **Predict Churn**")

# ==============================
# Build DataFrame from inputs
# ==============================
user_df = pd.DataFrame([input_data])

# Align with training features
aligned = {feat: user_df.loc[0, feat] if feat in user_df.columns else 0.0 for feat in features}
input_df = pd.DataFrame([aligned], columns=features).astype(float)

# ==============================
# Predict when user clicks
# ==============================
if st.sidebar.button("🔮 Predict Churn"):
    results = {}
    probs = {}
    for name, model in models.items():
        try:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
            results[name] = "Churn" if int(pred) == 1 else "No Churn"
            probs[name] = float(prob) if prob is not None else None
        except Exception as e:
            st.error(f"Model {name} failed to predict: {e}")
            st.stop()

    # Show table
    res_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Prediction": list(results.values()),
        "Churn Probability": [round(p, 4) if p is not None else None for p in probs.values()]
    }).set_index("Model")

    st.subheader("📊 Model Predictions")
    st.dataframe(res_df)

    # Probability bar chart
    if any(p is not None for p in probs.values()):
        prob_df = pd.DataFrame({
            "Model": list(probs.keys()),
            "Churn Probability": [p if p is not None else 0 for p in probs.values()]
        })
        fig = px.bar(prob_df, x="Model", y="Churn Probability", color="Model", text_auto=True, title="Churn Probability by Model")
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# Model performance display
# ==============================
st.divider()
st.subheader("📈 Model Performance (Test Accuracy)")
perf_path = os.path.join(MODELS_DIR, "model_performance.csv")

if os.path.isfile(perf_path):
    perf = pd.read_csv(perf_path, index_col=0)
    st.dataframe(perf)
    fig_perf = px.bar(perf.reset_index(), x='index', y='Accuracy', color='Accuracy', text_auto=True)
    fig_perf.update_layout(xaxis_title="Model", yaxis_title="Accuracy")
    st.plotly_chart(fig_perf, use_container_width=True)
else:
    st.info("Model performance file not found. Train models with main.py to generate it.")
