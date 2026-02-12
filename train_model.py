# ================= IMPORT LIBRARIES ================= #
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier


# ================= LOAD DATA ================= #

# Load the loan dataset
df = pd.read_csv("Loan_default.csv")

# Remove LoanID since it is only an identifier, not a feature
if "LoanID" in df.columns:
    df = df.drop("LoanID", axis=1)

# Separate target variable and input features
# Default = 1 → Will NOT repay loan
# Default = 0 → Will repay loan
y = df["Default"]
X = df.drop("Default", axis=1)


# ================= ENCODE CATEGORICAL FEATURES ================= #

# Identify categorical columns
cat_cols = X.select_dtypes(include="object").columns
encoders = {}

# Convert categorical text data into numerical form
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le  # Save encoder for deployment


# Save feature order to avoid mismatch during prediction
joblib.dump(list(X.columns), "feature_names.pkl")


# ================= TRAIN–TEST SPLIT ================= #

# Split data into training (80%) and testing (20%) sets
# Stratify ensures class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ================= DEFINE MODELS ================= #

# Bagging model: Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Boosting model: XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# Stacking model: Combination of RF and XGBoost
stack = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("xgb", xgb)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    n_jobs=-1
)

# Store all models in a dictionary
models = {
    "RandomForest": rf,
    "XGBoost": xgb,
    "Stacking": stack
}


# ================= TRAIN & EVALUATE MODELS ================= #

best_model = None
best_test_acc = 0.0
best_train_acc = 0.0

print("\n========= MODEL PERFORMANCE =========\n")

# Train each model and evaluate accuracy
for name, model in models.items():
    model.fit(X_train, y_train)

    # Accuracy on training data
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    # Accuracy on unseen testing data
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"{name}")
    print(f"  Training Accuracy: {train_acc * 100:.2f} %")
    print(f"  Testing  Accuracy: {test_acc * 100:.2f} %\n")

    # Select the model with highest testing accuracy
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_train_acc = train_acc
        best_model = model


# ================= FINAL RESULTS ================= #

print("====================================")
print("BEST MODEL SELECTED")
print(f"Training Accuracy: {best_train_acc * 100:.2f} %")
print(f"Testing  Accuracy: {best_test_acc * 100:.2f} %")
print("====================================\n")


# ================= SAVE TRAINED FILES ================= #

# Save the best-performing model
joblib.dump(best_model, "best_model.pkl")

# Save encoders for consistent preprocessing in UI
joblib.dump(encoders, "encoders.pkl")

print("✅ best_model.pkl, encoders.pkl, feature_names.pkl saved")
