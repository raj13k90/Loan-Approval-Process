
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Loan Approval Comparison App", layout="wide")

st.title("🏦 Loan Approval Prediction & Model Comparison")

uploaded_file = st.sidebar.file_uploader("📁 Upload loan dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    required_cols = ['loan_amount', 'annual_income', 'credit_score', 'interest_rate', 'loan_status']
    if not set(required_cols).issubset(df.columns):
        st.error(f"Dataset must include: {', '.join(required_cols)}")
    else:
        df.fillna(df.median(numeric_only=True), inplace=True)
        X = df[['loan_amount', 'annual_income', 'credit_score', 'interest_rate']]
        y = df['loan_status']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            auc = roc_auc_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            results[name] = {
                "model": model,
                "y_pred": y_pred,
                "auc": auc,
                "accuracy": acc,
                "report": classification_report(y_test, y_pred, output_dict=True)
            }

        st.subheader("📈 Model Performance Comparison")

        scores_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy": [results[m]["accuracy"] for m in results],
            "AUC-ROC": [results[m]["auc"] for m in results]
        })

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(scores_df.set_index("Model"))
        with col2:
            st.bar_chart(scores_df.set_index("Model"))

        st.subheader("🔍 Predict with Your Inputs")
        loan_amount = st.number_input("Loan Amount", 500, 50000, 15000)
        annual_income = st.number_input("Annual Income", 10000, 200000, 60000)
        credit_score = st.slider("Credit Score", 300, 850, 700)
        interest_rate = st.slider("Interest Rate (%)", 1.0, 25.0, 10.0)

        input_data = scaler.transform([[loan_amount, annual_income, credit_score, interest_rate]])

        st.write("### Predictions from All Models:")
        for name in models:
            model = results[name]["model"]
            pred = model.predict(input_data)[0]
            decision = "✅ Approved" if pred == 1 else "❌ Denied"
            st.write(f"**{name}**: {decision}")

        st.write("### 🔍 Detailed Classification Reports")
        for name in models:
            st.markdown(f"**{name}**")
            st.code(classification_report(y_test, results[name]["y_pred"]))
else:
    st.info("📥 Please upload your dataset to begin.")
