import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------- PAGE SETTINGS ----------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"  # better for mobile
)

# Stylish Title
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>💳 Fraud Detection System</h1>", unsafe_allow_html=True)
st.write("### A smart ML-powered system to detect fraudulent transactions.")

# ---------- LOAD MODEL ----------
rf = joblib.load("rf_fraud_model.joblib")
scaler = joblib.load("scaler.joblib")

# ---------- LOAD AND PREPARE TEST DATA ----------
file_path = r'C:\Users\u\Desktop\india.csv'
df = pd.read_csv(file_path)

target_col = 'is_fraudulent'
df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')
df['hour'] = df['transaction_time'].dt.hour.fillna(0).astype(int)
df['month'] = df['transaction_time'].dt.month.fillna(0).astype(int)
df['day'] = df['transaction_time'].dt.day.fillna(0).astype(int)
df.drop(columns=['transaction_time'], inplace=True)
df.fillna(0, inplace=True)

df = pd.get_dummies(df, columns=['card_type','location','purchase_category'], drop_first=True)

model_columns = [
    'transaction_id', 'customer_id', 'merchant_id', 'amount', 'customer_age',
    'hour', 'month', 'day',
    'card_type_Rupay', 'card_type_Visa',
    'location_Bangalore', 'location_Chennai', 'location_Delhi',
    'location_Hyderabad','location_Jaipur', 'location_Kolkata',
    'location_Mumbai', 'location_Pune', 'location_Surat',
    'purchase_category_POS'
]

for col in model_columns:
    if col not in df.columns:
        df[col] = 0

X = df[model_columns]
y = df[target_col]

X_scaled = scaler.transform(X)

y_pred = rf.predict(X_scaled)
y_prob = rf.predict_proba(X_scaled)[:, 1]

accuracy = accuracy_score(y, y_pred)
roc = roc_auc_score(y, y_prob)

st.success(f"✔ Model Accuracy: **{round(accuracy*100,2)}%**")
st.info(f"ROC-AUC Score: **{round(roc,4)}**")


# ---------- USER INPUT FORM ----------
st.write("---")
st.subheader("Enter Transaction Details")

with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        transaction_id = st.number_input("Transaction ID", step=1)
        customer_id = st.number_input("Customer ID", step=1)
        merchant_id = st.number_input("Merchant ID", step=1)
        amount = st.number_input("Amount", step=0.01)
        customer_age = st.number_input("Customer Age", step=1)

    with col2:
        hour = st.slider("Hour (0-23)", 0, 23, 12)
        month = st.slider("Month (1-12)", 1, 12, 6)
        day = st.slider("Day (1-31)", 1, 31, 15)
        card_type = st.selectbox("Card Type", ["Rupay", "Visa", "MasterCard"])
        location = st.selectbox("Location", ["Bangalore","Chennai","Delhi","Hyderabad","Jaipur","Kolkata","Mumbai","Pune","Surat"])
        purchase_category = st.selectbox("Purchase Category", ["POS", "Digital"])

    submitted = st.form_submit_button("🔍 Predict Fraud")


# ---------- PREDICTION ----------
if submitted:
    new_df = pd.DataFrame({
        'transaction_id':[transaction_id],
        'customer_id':[customer_id],
        'merchant_id':[merchant_id],
        'amount':[amount],
        'customer_age':[customer_age],
        'hour':[hour],
        'month':[month],
        'day':[day],
        'card_type':[card_type],
        'location':[location],
        'purchase_category':[purchase_category]
    })

    new_df = pd.get_dummies(new_df, columns=['card_type','location','purchase_category'])
    for col in model_columns:
        if col not in new_df.columns:
            new_df[col] = 0

    new_df = new_df[model_columns]
    new_scaled = scaler.transform(new_df)

    pred = rf.predict(new_scaled)[0]
    prob = rf.predict_proba(new_scaled)[0][1]

    st.write("---")
    if pred == 1:
        st.error(f"🚨 **Fraud Detected** – Probability: {round(prob,4)}")
    else:
        st.success(f"✅ **Not Fraud** – Probability: {round(prob,4)}")

    st.progress(float(prob))

