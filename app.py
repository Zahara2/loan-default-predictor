import streamlit as st
import joblib
import pandas as pd

# Load model & threshold
model = joblib.load("loan_default_model.pkl")
threshold = joblib.load("threshold.pkl")

st.title("Loan Default Prediction")

age = st.number_input("Age", 18, 100, 35)
income = st.number_input("Income", 10000, 200000, 60000)
loan_amount = st.number_input("Loan Amount", 1000, 100000, 20000)
loan_term = st.selectbox("Loan Term", [12, 24, 36, 48, 60])
interest_rate = st.slider("Interest Rate (%)", 1.0, 30.0, 12.5)
credit_score = st.slider("Credit Score", 300, 850, 650)
employment_years = st.number_input("Employment Years", 0, 40, 5)
debt_to_income = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.35)
previous_defaults = st.number_input("Previous Defaults", 0, 10, 0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "interest_rate": interest_rate,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income,
        "previous_defaults": previous_defaults
    }])

    prob = model.predict_proba(input_df)[0][1]
    decision = "Reject Loan" if prob >= threshold else "Approve Loan"

    st.metric("Default Probability", f"{prob:.2%}")
    st.write("Decision:", decision)
