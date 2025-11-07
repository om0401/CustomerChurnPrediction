import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

st.title("📈 Customer Churn Prediction Dashboard")
st.markdown("Use the left panel to explore the dataset, visualize churn trends, or predict new customer churns.")

# ==============================
# Sidebar Navigation
# ==============================
st.sidebar.title("🔍 Navigation Panel")
section = st.sidebar.radio(
    "Choose Section:",
    ["📊 Data Info", "📈 Visualization", "🔮 Prediction"],
    index=2
)

# ==============================
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "churn.csv"))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))

# ==============================
# Load Models and Features
# ==============================
expected_files = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Random Forest": "Random_Forest.pkl",
    "Gradient Boosting": "Gradient_Boosting.pkl",
    "XGBoost": "XGBoost.pkl",
    "features": "features.pkl"
}

models = {}
try:
    for display_name, fname in expected_files.items():
        if display_name == "features":
            continue
        model_path = os.path.join(MODELS_DIR, fname)
        if os.path.isfile(model_path):
            models[display_name] = joblib.load(model_path)
    features = joblib.load(os.path.join(MODELS_DIR, expected_files["features"]))
except Exception as e:
    st.error("Error loading models: " + str(e))
    st.stop()

# ==============================
# SECTION 1: DATA INFO
# ==============================
if section == "📊 Data Info":
    st.header("📋 Dataset Information")

    if os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.dataframe(df.head(), use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])

        st.markdown("### 🔢 Data Summary")
        st.dataframe(df.describe(include='all').T, use_container_width=True)

        st.markdown("### 🧩 Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0], use_container_width=True)
    else:
        st.error("Dataset not found at: `data/churn.csv`")

# ==============================
# SECTION 2: VISUALIZATION
# ==============================
elif section == "📈 Visualization":
    st.header("📊 Data Visualization")

    if os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

        st.markdown("### 🎯 Churn Distribution")
        fig1 = px.pie(df, names='Churn', title="Churn vs Non-Churn Customers")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### 💰 Monthly Charges Distribution")
        fig2 = px.histogram(df, x='MonthlyCharges', color='Churn', nbins=30)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### ⏳ Tenure vs Churn")
        fig3 = px.box(df, x='Churn', y='tenure', color='Churn', title="Tenure vs Churn")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### 📦 Contract Type vs Churn")
        if "Contract" in df.columns:
            fig4 = px.histogram(df, x='Contract', color='Churn', barmode='group')
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.error("Dataset not found for visualization!")

# ==============================
# SECTION 3: PREDICTION
# ==============================
elif section == "🔮 Prediction":
    st.header("🧾 Predict Customer Churn")

    st.markdown("Provide customer details below and click **Predict Churn**")

    input_data = {}

    # Demographics
    st.subheader("👤 Demographics")
    col1, col2 = st.columns(2)
    input_data["gender"] = col1.selectbox("Gender", [1, 0], index=1, format_func=lambda x: "Male" if x == 1 else "Female")
    input_data["SeniorCitizen"] = col2.selectbox("Senior Citizen", [0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["Partner"] = col1.selectbox("Partner", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["Dependents"] = col2.selectbox("Dependents", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")

    st.subheader("📞 Services & Account")
    input_data["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
    input_data["PhoneService"] = st.selectbox("Phone Service", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["MultipleLines"] = st.selectbox("Multiple Lines", [0, 1, 2], index=0, format_func=lambda x: ["No", "Yes", "No phone service"][x])
    input_data["InternetService"] = st.selectbox("Internet Service", [0, 1, 2], index=0, format_func=lambda x: ["DSL", "Fiber optic", "No"][x])

    st.subheader("💻 Online Services")
    input_data["OnlineSecurity"] = st.selectbox("Online Security", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["OnlineBackup"] = st.selectbox("Online Backup", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["DeviceProtection"] = st.selectbox("Device Protection", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["TechSupport"] = st.selectbox("Tech Support", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["StreamingTV"] = st.selectbox("Streaming TV", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["StreamingMovies"] = st.selectbox("Streaming Movies", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")

    st.subheader("💳 Contract & Billing")
    input_data["Contract"] = st.selectbox("Contract Type", [0, 1, 2], index=0, format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
    input_data["PaperlessBilling"] = st.selectbox("Paperless Billing", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["PaymentMethod"] = st.selectbox("Payment Method", [0, 1, 2, 3], index=0, format_func=lambda x: ["Electronic check", "Mailed check", "Bank transfer (auto)", "Credit card (auto)"][x])

    st.subheader("💰 Charges")
    input_data["MonthlyCharges"] = st.number_input("Monthly Charges ($)", 0.0, 10000.0, 70.0, 1.0)
    input_data["TotalCharges"] = st.number_input("Total Charges ($)", 0.0, 100000.0, 2000.0, 10.0)

    # Align data
    user_df = pd.DataFrame([input_data])
    aligned = {feat: user_df.loc[0, feat] if feat in user_df.columns else 0.0 for feat in features}
    input_df = pd.DataFrame([aligned], columns=features).astype(float)

    if st.button("🔮 Predict Churn"):
        results, probs = {}, {}
        for name, model in models.items():
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
            results[name] = "Churn" if int(pred) == 1 else "No Churn"
            probs[name] = float(prob) if prob is not None else None

        res_df = pd.DataFrame({
            "Model": results.keys(),
            "Prediction": results.values(),
            "Churn Probability": [round(p, 4) if p is not None else None for p in probs.values()]
        })
        st.subheader("📊 Model Predictions")
        st.dataframe(res_df, use_container_width=True)

        fig = px.bar(res_df, x="Model", y="Churn Probability", color="Model", text_auto=True,
                     title="Churn Probability by Model")
        st.plotly_chart(fig, use_container_width=True)
