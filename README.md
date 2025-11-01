# 📊 Customer Churn Prediction App
Live link:  https://customerchurnprediction0401.streamlit.app/

An interactive **Machine Learning web application** built with **Streamlit** that predicts whether a telecom customer will churn or stay based on various customer attributes such as demographics, services, and billing information.

---

## 🚀 Project Overview

Customer churn — the loss of clients or subscribers — is one of the most critical business metrics for telecom and subscription-based companies.  
This project leverages **machine learning models** to predict the likelihood of churn using customer data.

The app provides:
- Real-time churn prediction for a single customer  
- Visual comparison of multiple trained models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
- Interactive dashboards with probability visualizations

---

## 📂 Project Structure

CustomerChurnPrediction/
│
├── app/
│ └── app.py # Streamlit web app
│
├── models/
│ ├── Logistic_Regression.pkl
│ ├── Random_Forest.pkl
│ ├── Gradient_Boosting.pkl
│ ├── XGBoost.pkl
│ ├── features.pkl
│ └── model_performance.csv
│
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv # Original dataset
│
├── main.py # Model training & saving script
├── requirements.txt # Project dependencies
├── .gitignore # Ignore unnecessary files
└── README.md # Project documentation

yaml
Copy code

---

## 🧠 Dataset Information

**Source:** [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)

### 🧾 Description of Columns

| Column | Description |
|--------|--------------|
| **customerID** | Unique ID for each customer |
| **gender** | Whether the customer is male or female |
| **SeniorCitizen** | Whether the customer is a senior citizen (1 or 0) |
| **Partner** | Whether the customer has a partner (Yes, No) |
| **Dependents** | Whether the customer has dependents (Yes, No) |
| **tenure** | Number of months the customer has stayed with the company |
| **PhoneService** | Whether the customer has a phone service (Yes, No) |
| **MultipleLines** | Whether the customer has multiple lines (Yes, No, No phone service) |
| **InternetService** | Internet service provider (DSL, Fiber optic, No) |
| **OnlineSecurity** | Whether the customer has online security (Yes, No, No internet service) |
| **OnlineBackup** | Whether the customer has online backup (Yes, No, No internet service) |
| **DeviceProtection** | Whether the customer has device protection (Yes, No, No internet service) |
| **TechSupport** | Whether the customer has tech support (Yes, No, No internet service) |
| **StreamingTV** | Whether the customer streams TV (Yes, No, No internet service) |
| **StreamingMovies** | Whether the customer streams movies (Yes, No, No internet service) |
| **Contract** | Type of contract (Month-to-month, One year, Two year) |
| **PaperlessBilling** | Whether the customer uses paperless billing (Yes, No) |
| **PaymentMethod** | Payment method (Electronic check, Mailed check, Bank transfer, Credit card) |
| **MonthlyCharges** | Monthly amount charged to the customer |
| **TotalCharges** | Total amount charged |
| **Churn** | Whether the customer churned (Yes, No) |

---

## 🧩 Models Trained

| Model | Description | Accuracy |
|-------|--------------|----------|
| **Logistic Regression** | Simple linear classifier, interpretable | ~80% |
| **Random Forest** | Ensemble of decision trees, handles nonlinearity well | ~83% |
| **Gradient Boosting** | Sequential boosting of weak learners | ~85% |
| **XGBoost** | Optimized gradient boosting, best performance | ~86% |

Each model is saved as a `.pkl` file inside `/models` and loaded dynamically in the Streamlit app.

---

## 🧰 Installation & Setup

### 1️⃣ Clone this repository

```bash
git clone https://github.com/om0401/om0401-CustomerChurnPrediction.git
cd om0401-CustomerChurnPrediction
2️⃣ Create a virtual environment
Windows (PowerShell):

bash
Copy code
python -m venv venv
& .\venv\Scripts\Activate.ps1
Mac/Linux:

bash
Copy code
python3 -m venv venv
source venv/bin/activate
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Train models
bash
Copy code
python main.py
5️⃣ Run Streamlit app
bash
Copy code
streamlit run app/app.py
Then open http://localhost:8501 in your browser 🎯

📈 Visualizations
The Streamlit dashboard provides:

Churn probability chart per model

Model performance comparison (accuracy bar chart)

Interactive sidebar inputs for customer attributes

Example:

less
Copy code
Customer Will Stay ✅ (Churn Probability: 12%)
📦 Dependencies
Main libraries used:

pandas

numpy

scikit-learn

xgboost

plotly

streamlit

joblib

See requirements.txt for exact versions.

🧑‍💻 Author
👤 Om maurya
📧 ommaurya7472gmail.com
💻 GitHub: @om0401

🪄 Future Improvements
Add SHAP feature importance for model interpretability

Deploy app to cloud (Render, AWS, Streamlit Cloud, etc.)

Add database integration for live churn monitoring

🏁 License
This project is licensed under the MIT License — see the LICENSE file for details.

yaml
Copy code

---

Would you like me to **add badges** (like Python version, Streamlit app status, dataset link, etc.) and format it with emojis and a centered title for a cleaner GitHub look?  
It would make your repo visually professional like top ML projects.
