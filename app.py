# ================= IMPORT LIBRARIES ================= #
import streamlit as st
import pandas as pd
import joblib


# ================= PAGE CONFIGURATION ================= #
# Sets page title and layout for a clean, centered UI
st.set_page_config(
    page_title="Loan Default Prediction",
    layout="centered"
)


# ================= LOAD SAVED FILES ================= #
# Load trained ML model, encoders, and feature order
model = joblib.load("best_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_names = joblib.load("feature_names.pkl")


# ================= UI HEADER ================= #
st.title("🏦 Loan Default Prediction System")
st.write("Predict whether a loan applicant will **DEFAULT (1)** or **NOT DEFAULT (0)**")


# ================= USER INPUT FIELDS ================= #
# Numeric inputs
age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Income", 1000, 1_000_000, 50000)
credit_score = st.number_input("Credit Score", 300, 900, 650)
dti = st.number_input("DTI Ratio", 0.0, 1.0, 0.30)

loan_amount = st.number_input("Loan Amount", 1000, 2_000_000, 200000)
interest_rate = st.number_input("Interest Rate (%)", 1.0, 30.0, 10.0)
loan_term = st.number_input("Loan Term (months)", 6, 360, 120)

months_employed = st.number_input("Months Employed", 0, 500, 60)
num_credit_lines = st.number_input("Number of Credit Lines", 0, 50, 5)


# Categorical inputs
credit_history = st.selectbox("Credit History", ["Good", "Bad"])
education = st.selectbox("Education", encoders["Education"].classes_)
employment = st.selectbox("Employment Type", encoders["EmploymentType"].classes_)
marital = st.selectbox("Marital Status", encoders["MaritalStatus"].classes_)
loan_purpose = st.selectbox("Loan Purpose", encoders["LoanPurpose"].classes_)

has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])


# ================= DATA ENCODING ================= #
# Convert Yes/No and Good/Bad values into binary format
credit_history = 1 if credit_history == "Good" else 0
has_mortgage = 1 if has_mortgage == "Yes" else 0
has_dependents = 1 if has_dependents == "Yes" else 0
has_cosigner = 1 if has_cosigner == "Yes" else 0

# Encode categorical variables using saved encoders
education = encoders["Education"].transform([education])[0]
employment = encoders["EmploymentType"].transform([employment])[0]
marital = encoders["MaritalStatus"].transform([marital])[0]
loan_purpose = encoders["LoanPurpose"].transform([loan_purpose])[0]


# ================= PREDICTION LOGIC ================= #
# Run prediction only when the button is clicked
if st.button("Predict"):

    # Create input DataFrame with one row (same format as training data)
    input_data = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti,
        "Education": education,
        "EmploymentType": employment,
        "MaritalStatus": marital,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }])

    # Ensure feature order matches training data
    input_data = input_data[feature_names]

    # Predict loan default (0 or 1)
    prediction = model.predict(input_data)[0]

    # Display result with clear meaning
    if prediction == 1:
        st.error("❌ Prediction: DEFAULT (1)\n\nPerson will **NOT repay** the loan")
    else:
        st.success("✅ Prediction: NOT DEFAULT (0)\n\nPerson **WILL repay** the loan")
