import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("data/loan_data.csv")

print("=" * 70)
print("DATA OVERVIEW")
print("=" * 70)
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

# =========================================================
# 2. BASIC DATA CHECKS
# =========================================================
print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(df.isnull().sum())

print("\n" + "=" * 70)
print("DEFAULT DISTRIBUTION")
print("=" * 70)
print(df["default"].value_counts())
print("\nDefault Rate:", round(df["default"].mean() * 100, 2), "%")

# =========================================================
# 3. SIMPLE VISUALIZATIONS
# =========================================================
plt.figure(figsize=(6, 4))
df["default"].value_counts().plot(kind="bar")
plt.title("Default vs Non-Default Count")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df[df["default"] == 0]["fico_score"], bins=30, alpha=0.6, label="No Default")
plt.hist(df[df["default"] == 1]["fico_score"], bins=30, alpha=0.6, label="Default")
plt.title("FICO Score Distribution by Default Status")
plt.xlabel("FICO Score")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["income"], df["loan_amt_outstanding"], alpha=0.4, c=df["default"])
plt.title("Income vs Loan Amount Outstanding")
plt.xlabel("Income")
plt.ylabel("Loan Amount Outstanding")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================
# 4. FEATURE SELECTION
# =========================================================
X = df.drop(columns=["customer_id", "default"])
y = df["default"]
feature_names = X.columns.tolist()

# =========================================================
# 5. TRAIN-TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)
print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# =========================================================
# 6. TRAIN MULTIPLE MODELS
# =========================================================
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
}

results = []
best_model = None
best_auc = 0
best_model_name = None

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(auc, 4)
    })

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

# =========================================================
# 7. MODEL COMPARISON
# =========================================================
results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(results_df)

print("\nBest Model:", best_model_name)
print("Best ROC-AUC:", round(best_auc, 4))

# =========================================================
# 8. BEST MODEL EVALUATION
# =========================================================
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print(f"BEST MODEL CLASSIFICATION REPORT: {best_model_name}")
print("=" * 70)
print(classification_report(y_test, y_pred_best, zero_division=0))

cm = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix:\n", cm)

RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title(f"ROC Curve - {best_model_name}")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================
# 9. FEATURE IMPORTANCE / COEFFICIENTS
# =========================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE / INTERPRETATION")
print("=" * 70)

if best_model_name == "Logistic Regression":
    # IMPORTANT FIX: extract logistic regression from pipeline
    logistic_model = best_model.named_steps["model"]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": logistic_model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

elif best_model_name in ["Decision Tree", "Random Forest"]:
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df.iloc[::-1, 0], importance_df.iloc[::-1, 1])
plt.title(f"Feature Importance - {best_model_name}")
plt.xlabel("Importance / Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# =========================================================
# 10. INPUT VALIDATION
# =========================================================
def validate_input(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score
):
    if credit_lines_outstanding < 0 or credit_lines_outstanding > 20:
        raise ValueError("credit_lines_outstanding should be between 0 and 20")

    if loan_amt_outstanding < 0:
        raise ValueError("loan_amt_outstanding cannot be negative")

    if total_debt_outstanding < 0:
        raise ValueError("total_debt_outstanding cannot be negative")

    if income <= 0:
        raise ValueError("income must be greater than 0")

    if years_employed < 0 or years_employed > 50:
        raise ValueError("years_employed should be between 0 and 50")

    if fico_score < 300 or fico_score > 850:
        raise ValueError("fico_score should be between 300 and 850")

# =========================================================
# 11. FUNCTION TO PREDICT PD
# =========================================================
def predict_default_probability(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score
):
    validate_input(
        credit_lines_outstanding,
        loan_amt_outstanding,
        total_debt_outstanding,
        income,
        years_employed,
        fico_score
    )

    input_data = pd.DataFrame([{
        "credit_lines_outstanding": credit_lines_outstanding,
        "loan_amt_outstanding": loan_amt_outstanding,
        "total_debt_outstanding": total_debt_outstanding,
        "income": income,
        "years_employed": years_employed,
        "fico_score": fico_score
    }])

    pd_value = best_model.predict_proba(input_data)[0][1]
    return round(float(pd_value), 4)

# =========================================================
# 12. RISK BUCKET FUNCTION
# =========================================================
def assign_risk_bucket(pd_value):
    if pd_value < 0.20:
        return "Low Risk"
    elif pd_value < 0.50:
        return "Medium Risk"
    else:
        return "High Risk"

# =========================================================
# 13. FUNCTION TO CALCULATE EXPECTED LOSS
# =========================================================
def calculate_expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score,
    recovery_rate=0.10
):
    pd_value = predict_default_probability(
        credit_lines_outstanding,
        loan_amt_outstanding,
        total_debt_outstanding,
        income,
        years_employed,
        fico_score
    )

    lgd = 1 - recovery_rate
    ead = loan_amt_outstanding
    expected_loss = pd_value * lgd * ead
    risk_bucket = assign_risk_bucket(pd_value)

    return {
        "Probability_of_Default": round(pd_value, 4),
        "Risk_Bucket": risk_bucket,
        "Loss_Given_Default": round(lgd, 2),
        "Exposure_at_Default": round(ead, 2),
        "Expected_Loss": round(expected_loss, 2)
    }

# =========================================================
# 14. TEST CASES
# =========================================================
print("\n" + "=" * 70)
print("SAMPLE BORROWER TESTS")
print("=" * 70)

sample_1 = calculate_expected_loss(
    credit_lines_outstanding=1,
    loan_amt_outstanding=3000,
    total_debt_outstanding=5000,
    income=70000,
    years_employed=8,
    fico_score=760
)

sample_2 = calculate_expected_loss(
    credit_lines_outstanding=5,
    loan_amt_outstanding=12000,
    total_debt_outstanding=20000,
    income=35000,
    years_employed=2,
    fico_score=580
)

sample_3 = calculate_expected_loss(
    credit_lines_outstanding=3,
    loan_amt_outstanding=7000,
    total_debt_outstanding=11000,
    income=50000,
    years_employed=4,
    fico_score=660
)

print("\nBorrower 1:")
print(sample_1)

print("\nBorrower 2:")
print(sample_2)

print("\nBorrower 3:")
print(sample_3)

# =========================================================
# 15. USER INPUT TEST
# =========================================================
print("\n" + "=" * 70)
print("CUSTOM BORROWER PREDICTION")
print("=" * 70)

try:
    credit_lines = int(input("Enter credit lines outstanding (0-20): "))
    loan_amt = float(input("Enter loan amount outstanding: "))
    total_debt = float(input("Enter total debt outstanding: "))
    income = float(input("Enter income: "))
    years_employed = float(input("Enter years employed: "))
    fico_score = int(input("Enter fico score (300-850): "))

    result = calculate_expected_loss(
        credit_lines_outstanding=credit_lines,
        loan_amt_outstanding=loan_amt,
        total_debt_outstanding=total_debt,
        income=income,
        years_employed=years_employed,
        fico_score=fico_score
    )

    print("\nPrediction Result:")
    print(result)

except ValueError as e:
    print("\nInput Error:", e)