import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.title("📈 Customer Churn Prediction Dashboard")

st.markdown("""
Welcome to the **Customer Churn Prediction App**.  
Use the left panel to explore data, visualize trends, or predict new customer churns.
""")

# ==============================
# Sidebar Navigation
# ==============================
st.sidebar.title("🔍 Navigation Panel")
section = st.sidebar.radio("Choose Section:", ["📊 Data Info", "📈 Visualization", "🔮 Prediction"], index=2)

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
for display_name, fname in expected_files.items():
    if display_name == "features":
        continue
    path = os.path.join(MODELS_DIR, fname)
    if os.path.isfile(path):
        models[display_name] = joblib.load(path)

features = joblib.load(os.path.join(MODELS_DIR, expected_files["features"]))

# ==============================================================
# SECTION 1: DATA INFO
# ==============================================================
if section == "📊 Data Info":
    st.header("📋 Dataset Overview")

    if os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

        st.dataframe(df.head(), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        churn_ratio = round((df['Churn'].value_counts(normalize=True).get("Yes", 0) * 100), 2)
        col3.metric("Churn Ratio (%)", churn_ratio)

        st.subheader("🔢 Data Summary (describe)")
        st.dataframe(df.describe(include='all').T, use_container_width=True)

        st.subheader("🧾 Description of Columns")
        col_desc = {
            "customerID": "Unique ID for each customer",
            "gender": "Whether the customer is male or female",
            "SeniorCitizen": "Whether the customer is a senior citizen (1 or 0)",
            "Partner": "Whether the customer has a partner (Yes, No)",
            "Dependents": "Whether the customer has dependents (Yes, No)",
            "tenure": "Number of months the customer has stayed with the company",
            "PhoneService": "Whether the customer has a phone service (Yes, No)",
            "MultipleLines": "Whether the customer has multiple lines (Yes, No, No phone service)",
            "InternetService": "Internet service provider (DSL, Fiber optic, No)",
            "OnlineSecurity": "Whether the customer has online security (Yes, No, No internet service)",
            "OnlineBackup": "Whether the customer has online backup (Yes, No, No internet service)",
            "DeviceProtection": "Whether the customer has device protection (Yes, No, No internet service)",
            "TechSupport": "Whether the customer has tech support (Yes, No, No internet service)",
            "StreamingTV": "Whether the customer streams TV (Yes, No, No internet service)",
            "StreamingMovies": "Whether the customer streams movies (Yes, No, No internet service)",
            "Contract": "Type of contract (Month-to-month, One year, Two year)",
            "PaperlessBilling": "Whether the customer uses paperless billing (Yes, No)",
            "PaymentMethod": "Payment method (Electronic check, Mailed check, Bank transfer, Credit card)",
            "MonthlyCharges": "Monthly amount charged to the customer",
            "TotalCharges": "Total amount charged",
            "Churn": "Whether the customer churned (Yes, No)"
        }
        desc_df = pd.DataFrame(list(col_desc.items()), columns=["Column", "Description"])
        st.table(desc_df)
    else:
        st.error("❌ Dataset not found. Please ensure 'data/churn.csv' exists.")

# ==============================================================
# SECTION 2: VISUALIZATION
# ==============================================================
elif section == "📈 Visualization":
    st.header("📊 Data Visualizations")

    if os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

        st.markdown("### 🎯 Churn Distribution")
        fig1 = px.pie(df, names='Churn', title="Churn vs Non-Churn Customers")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### 🧍 Gender vs Churn")
        fig2 = px.histogram(df, x='gender', color='Churn', barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 💰 Monthly Charges Distribution by Churn")
        fig3 = px.histogram(df, x='MonthlyCharges', color='Churn', nbins=30)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### ⏳ Tenure vs Monthly Charges (Colored by Churn)")
        fig4 = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn')
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("### 📦 Contract Type vs Churn")
        fig5 = px.histogram(df, x='Contract', color='Churn', barmode='group')
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("### 🔥 Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig_corr)
    else:
        st.error("❌ Dataset not found for visualization.")

# ==============================================================
# SECTION 3: PREDICTION
# ==============================================================
elif section == "🔮 Prediction":
    st.header("🧾 Predict Customer Churn")

    st.markdown("Provide customer details below and click **Predict Churn**")

    input_data = {}

    # --- Input Fields ---
    col1, col2 = st.columns(2)
    input_data["gender"] = col1.selectbox("Gender", [1, 0], index=1, format_func=lambda x: "Male" if x == 1 else "Female")
    input_data["SeniorCitizen"] = col2.selectbox("Senior Citizen", [0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["Partner"] = col1.selectbox("Partner", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["Dependents"] = col2.selectbox("Dependents", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
    input_data["PhoneService"] = st.selectbox("Phone Service", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["MultipleLines"] = st.selectbox("Multiple Lines", [0, 1, 2], index=0, format_func=lambda x: ["No", "Yes", "No phone service"][x])
    input_data["InternetService"] = st.selectbox("Internet Service", [0, 1, 2], index=0, format_func=lambda x: ["DSL", "Fiber optic", "No"][x])
    input_data["OnlineSecurity"] = st.selectbox("Online Security", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["OnlineBackup"] = st.selectbox("Online Backup", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["DeviceProtection"] = st.selectbox("Device Protection", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["TechSupport"] = st.selectbox("Tech Support", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["StreamingTV"] = st.selectbox("Streaming TV", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["StreamingMovies"] = st.selectbox("Streaming Movies", [1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["Contract"] = st.selectbox("Contract Type", [0, 1, 2], index=0, format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
    input_data["PaperlessBilling"] = st.selectbox("Paperless Billing", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    input_data["PaymentMethod"] = st.selectbox("Payment Method", [0, 1, 2, 3], index=0, format_func=lambda x: ["Electronic check", "Mailed check", "Bank transfer (auto)", "Credit card (auto)"][x])
    input_data["MonthlyCharges"] = st.number_input("Monthly Charges ($)", 0.0, 10000.0, 70.0, 1.0)
    input_data["TotalCharges"] = st.number_input("Total Charges ($)", 0.0, 100000.0, 2000.0, 10.0)

    # --- Align with features ---
    user_df = pd.DataFrame([input_data])
    aligned = {feat: user_df.loc[0, feat] if feat in user_df.columns else 0.0 for feat in features}
    input_df = pd.DataFrame([aligned], columns=features).astype(float)

    # --- Predict ---
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

        # Probability chart
        fig = px.bar(res_df, x="Model", y="Churn Probability", color="Model", text_auto=True, title="Current Prediction Probabilities")
        st.plotly_chart(fig, use_container_width=True)

        # Load model performance
        perf_path = os.path.join(MODELS_DIR, "model_performance.csv")
        if os.path.isfile(perf_path):
            st.subheader("📈 Trained Model Accuracy (from test set)")
            perf = pd.read_csv(perf_path, index_col=0)
            st.dataframe(perf)
            fig_perf = px.bar(perf.reset_index(), x='index', y='Accuracy', color='Accuracy', text_auto=True)
            fig_perf.update_layout(xaxis_title="Model", yaxis_title="Accuracy")
            st.plotly_chart(fig_perf, use_container_width=True)
