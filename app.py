
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Loan Model Rank App", layout="wide")
st.title("🏦 Loan Approval Predictor with Ranked Models")

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    required_cols = ['loan_amount', 'annual_income', 'credit_score', 'interest_rate', 'loan_status']
    if not set(required_cols).issubset(df.columns):
        st.error(f"Dataset must include: {', '.join(required_cols)}")
    else:
        df.dropna(subset=required_cols, inplace=True)
        X = df[['loan_amount', 'annual_income', 'credit_score', 'interest_rate']]
        y = df['loan_status']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        models = {
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression()
        }

        results = []
        predictions = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            predictions[name] = model
            results.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 2),
                "Precision": round(precision_score(y_test, y_pred), 2),
                "Recall": round(recall_score(y_test, y_pred), 2),
                "F1-Score": round(f1_score(y_test, y_pred), 2),
                "ROC-AUC": round(roc_auc_score(y_test, y_pred), 2)
            })

        result_df = pd.DataFrame(results)
        st.subheader("📌 Key Insights from Model Performance")
        result_df['Rank'] = [1, 2, 3]  # Based on provided order
        result_df = result_df.sort_values(by="Rank")
        st.dataframe(result_df.drop(columns='Rank').style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

        st.subheader("📊 Accuracy & ROC-AUC Comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=result_df.melt(id_vars='Model', value_vars=['Accuracy', 'ROC-AUC']), x='Model', y='value', hue='variable', ax=ax)
        plt.title("Model Accuracy and AUC")
        st.pyplot(fig)

        st.subheader("🔮 Make a Prediction")
        loan_amount = st.number_input("Loan Amount", 500, 50000, 15000)
        annual_income = st.number_input("Annual Income", 10000, 200000, 60000)
        credit_score = st.slider("Credit Score", 300, 850, 700)
        interest_rate = st.slider("Interest Rate (%)", 1.0, 25.0, 10.0)

        input_data = scaler.transform([[loan_amount, annual_income, credit_score, interest_rate]])

        st.markdown("### Predictions from Ranked Models:")
        for name in result_df['Model']:
            pred = predictions[name].predict(input_data)[0]
            result = "✅ Approved" if pred == 1 else "❌ Denied"
            st.write(f"**{name}**: {result}")
else:
    st.info("📥 Please upload your loan dataset to begin.")
