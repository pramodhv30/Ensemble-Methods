# Ensemble-Methods
# Loan Default Prediction using Ensemble Learning

## 📌 Project Overview
This project predicts whether a loan applicant will default on a loan using Ensemble Learning techniques.

The system implements:
- Random Forest (Bagging)
- XGBoost (Boosting)
- Stacking Classifier (Meta Learning)

The best-performing model is automatically selected based on testing accuracy.

---

## 📊 Dataset Information
- Dataset: Loan Default
- Total Records: 255,347
- Total Features: 18
- Target Variable: Default
    - 1 → Will Default
    - 0 → Will Repay

Feature categories include:
- Demographics (Age, Education, Marital Status, Dependents)
- Financial Profile (Income, Credit Score, DTI Ratio)
- Loan Details (Loan Amount, Interest Rate, Loan Term)
- Employment & Stability
- Risk Mitigation (Mortgage, Co-signer)

---

## ⚙️ Models Implemented
1. Random Forest Classifier
2. XGBoost Classifier
3. Stacking Classifier (RF + XGB + Logistic Regression)

The training pipeline:
- Removes identifier columns
- Encodes categorical variables using LabelEncoder
- Splits data into 80% training and 20% testing (Stratified)
- Evaluates training and testing accuracy
- Selects the best model automatically
- Saves trained model and encoders for deployment

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies
pip install -r requirements.txt

### 2️⃣ Train the model
python train_model.py

### 3️⃣ Run Streamlit Application
streamlit run app.py

---

## 📈 Key Features
- Ensemble Learning implementation (Bagging, Boosting, Stacking)
- Automatic best model selection
- Model persistence using joblib
- Financial risk prediction use case
- Deployable via Streamlit UI

---

## 🏦 Applications
- Credit Risk Assessment
- Loan Approval Systems
- Financial Risk Modeling
- Fraud Detection

---

## 👨‍💻 Technologies Used
- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
