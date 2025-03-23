{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b30a61a8-1d39-4384-906c-da9892a6ca7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, roc_auc_score\n",
    "import pickle\n",
    "\n",
    "st.set_page_config(page_title=\"Loan Approval Predictor\", layout=\"wide\")\n",
    "\n",
    "# Sidebar settings\n",
    "st.sidebar.image(\"https://cdn-icons-png.flaticon.com/512/2721/2721272.png\", width=80)\n",
    "st.sidebar.title(\"Loan Prediction Settings\")\n",
    "uploaded_file = st.sidebar.file_uploader(\"📁 Upload CSV Dataset\", type=[\"csv\"])\n",
    "\n",
    "# App title and description\n",
    "st.title(\"🏦 Loan Approval Prediction Web App\")\n",
    "st.markdown(\"This application allows you to upload a loan dataset, train a machine learning model, and predict loan approvals.\")\n",
    "\n",
    "if uploaded_file:\n",
    "    df = pd.read_csv(uploaded_file)\n",
    "    st.subheader(\"📊 Dataset Preview\")\n",
    "    st.dataframe(df.head(), use_container_width=True)\n",
    "\n",
    "    required_cols = ['loan_amount', 'annual_income', 'credit_score', 'interest_rate', 'loan_status']\n",
    "    if not set(required_cols).issubset(df.columns):\n",
    "        st.error(f\"Dataset must include: {', '.join(required_cols)}\")\n",
    "    else:\n",
    "        df.fillna(df.median(numeric_only=True), inplace=True)\n",
    "\n",
    "        X = df[['loan_amount', 'annual_income', 'credit_score', 'interest_rate']]\n",
    "        y = df['loan_status']\n",
    "\n",
    "        scaler = StandardScaler()\n",
    "        X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
    "\n",
    "        model_name = st.sidebar.selectbox(\"🤖 Choose Model\", [\"Gradient Boosting\", \"Logistic Regression\"])\n",
    "        model = GradientBoostingClassifier() if model_name == \"Gradient Boosting\" else LogisticRegression()\n",
    "\n",
    "        model.fit(X_train, y_train)\n",
    "        y_pred = model.predict(X_test)\n",
    "\n",
    "        with open(\"loan_model.pkl\", \"wb\") as f:\n",
    "            pickle.dump(model, f)\n",
    "\n",
    "        st.subheader(\"✅ Model Evaluation\")\n",
    "        col1, col2 = st.columns(2)\n",
    "        with col1:\n",
    "            st.markdown(\"**Classification Report**\")\n",
    "            st.code(classification_report(y_test, y_pred), language='text')\n",
    "        with col2:\n",
    "            auc = roc_auc_score(y_test, y_pred)\n",
    "            st.metric(label=\"AUC-ROC Score\", value=f\"{auc:.2f}\")\n",
    "\n",
    "        st.subheader(\"🔍 Try a Prediction\")\n",
    "        col1, col2 = st.columns(2)\n",
    "        with col1:\n",
    "            loan_amount = st.number_input(\"Loan Amount\", 500, 50000, 15000)\n",
    "            annual_income = st.number_input(\"Annual Income\", 10000, 200000, 60000)\n",
    "        with col2:\n",
    "            credit_score = st.slider(\"Credit Score\", 300, 850, 700)\n",
    "            interest_rate = st.slider(\"Interest Rate (%)\", 1.0, 25.0, 10.0)\n",
    "\n",
    "        if st.button(\"🔮 Predict Loan Approval\"):\n",
    "            input_data = scaler.transform([[loan_amount, annual_income, credit_score, interest_rate]])\n",
    "            prediction = model.predict(input_data)\n",
    "            if prediction[0] == 1:\n",
    "                st.success(\"✅ Loan Approved\")\n",
    "            else:\n",
    "                st.error(\"❌ Loan Denied\")\n",
    "else:\n",
    "    st.info(\"📥 Please upload a dataset to get started.\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
