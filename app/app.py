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
st.markdown("An interactive Streamlit app for exploring customer churn data, visualization, and making real-time predictions.")

# ==============================
# Sidebar Navigation
# ==============================
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to Section:", ["📊 Data Info", "📈 Visualization", "🔮 Prediction"])

# ==============================
# Load Models and Features
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))

expected_files = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Random Forest": "Random_Forest.pkl",
    "Gradient Boosting": "Gradient_Boosting.pkl",
    "XGBoost": "XGBoost.pkl",
    "features": "features.pkl"
}

models = {}
for display_name, fname in expected_files.items():
    if display_name == "features":
        continue
    path = os.path.join(MODELS_DIR, fname)
    if os.path.isfile(path):
        models[display_name] = joblib.load(path)
features = joblib.load(os.path.join(MODELS_DIR, expected_files["features"]))

# ==============================
# Section 1: Data Info
# ==============================
if page == "📊 Data Info":
    st.subheader("📋 Dataset Overview")
    data_path = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "churn.csv"))

    if os.path.isfile(data_path):
        df = pd.read_csv(data_path)
        st.dataframe(df.head(), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])

        st.markdown("### 🔢 Data Summary")
        st.dataframe(df.describe(include='all').transpose(), use_container_width=True)

    else:
        st.error("Dataset not found! Please ensure 'data/churn.csv' exists.")

# ==============================
# Section 2: Visualization
# ==============================
elif page == "📈 Visualization":
    st.subheader("📊 Data Visualization")

    data_path = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "churn.csv"))
    if os.path.isfile(data_path):
        df = pd.read_csv(data_path)

        st.markdown("### 🎯 Churn Distribution")
        fig1 = px.pie(df, names='Churn', title="Churn vs Non-Churn Customers")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### 💰 Monthly Charges Distribution")
        fig2 = px.histogram(df, x='MonthlyCharges', color='Churn', nbins=30)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### ⏳ Tenure vs Churn")
        fig3 = px.box(df, x='Churn', y='tenure', color='Churn', title="Tenure vs Churn")
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.error("Dataset not found for visualization!")

# ==============================
# Section 3: Prediction
# ==============================
elif page == "🔮 Prediction":
    st.subheader("🧾 Input Customer Details")

    input_data = {}

    # Demographics
    input_data["gender"] = st.selectbox("Gender", [1, 0], index=1, format_func=lambda x: "Male" if x == 1 else "Female")
    input_data["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["Partner"] = st.selectbox("Partner", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["Dependents"] = st.selectbox("Dependents", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")

    input_data["tenure"] = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    input_data["PhoneService"] = st.selectbox("Phone Service", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["MultipleLines"] = st.selectbox("Multiple Lines", [0, 1, 2], index=0, format_func=lambda x: ["No", "Yes", "No phone service"][x])
    input_data["InternetService"] = st.selectbox("Internet Service", [0, 1, 2], index=0, format_func=lambda x: ["DSL", "Fiber optic", "No"][x])

    # Online features
    input_data["OnlineSecurity"] = st.selectbox("Online Security", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["OnlineBackup"] = st.selectbox("Online Backup", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["DeviceProtection"] = st.selectbox("Device Protection", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["TechSupport"] = st.selectbox("Tech Support", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["StreamingTV"] = st.selectbox("Streaming TV", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["StreamingMovies"] = st.selectbox("Streaming Movies", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")

    # Contract & billing
    input_data["Contract"] = st.selectbox("Contract Type", [0, 1, 2], index=0, format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
    input_data["PaperlessBilling"] = st.selectbox("Paperless Billing", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["PaymentMethod"] = st.selectbox("Payment Method", [0, 1, 2, 3], index=0, format_func=lambda x: ["Electronic check", "Mailed check", "Bank transfer (auto)", "Credit card (auto)"][x])

    # Charges
    input_data["MonthlyCharges"] = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=10000.0, value=70.0, step=1.0)
    input_data["TotalCharges"] = st.number_input("Total Charges ($)", min_value=0.0, max_value=100000.0, value=2000.0, step=10.0)

    st.markdown("---")

    user_df = pd.DataFrame([input_data])
    aligned = {feat: user_df.loc[0, feat] if feat in user_df.columns else 0.0 for feat in features}
    input_df = pd.DataFrame([aligned], columns=features).astype(float)

    if st.button("🔮 Predict Churn"):
        results = {}
        probs = {}
        for name, model in models.items():
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
            results[name] = "Churn" if int(pred) == 1 else "No Churn"
            probs[name] = float(prob) if prob is not None else None

        res_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Prediction": list(results.values()),
            "Churn Probability": [round(p, 4) if p is not None else None for p in probs.values()]
        }).set_index("Model")

        st.subheader("📊 Model Predictions")
        st.dataframe(res_df)

        prob_df = pd.DataFrame({
            "Model": list(probs.keys()),
            "Churn Probability": [p if p is not None else 0 for p in probs.values()]
        })
        fig = px.bar(prob_df, x="Model", y="Churn Probability", color="Model", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
