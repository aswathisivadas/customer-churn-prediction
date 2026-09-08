import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_churn_dataset-testing-master.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------- 1. LOAD & EXPLORE --------------------
df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nChurn distribution:")
print(df["Churn"].value_counts())

# -------------------- 2. CLEANING --------------------
# Remove duplicate rows.
df = df.drop_duplicates().copy()

# CustomerID is an identifier, not a useful predictive feature.
X = df.drop(columns=["Churn", "CustomerID"])
y = df["Churn"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Numeric: median imputation + scaling.
# Categorical: most-frequent imputation + one-hot encoding.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

# -------------------- 3. TRAIN / TEST SPLIT --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------- 4. MODELS --------------------
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        random_state=42,
        n_jobs=-1
    ))
])

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

# -------------------- 5. TRAIN & EVALUATE --------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"\n===== {name} =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
    plt.close()

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# Select the model with the highest F1 score.
best_name = results_df.loc[results_df["F1 Score"].idxmax(), "Model"]
best_model = models[best_name]
print("\nSelected Model:", best_name)

joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")

# -------------------- 6. DATA VISUALIZATIONS --------------------
# Visualization 1: Churn vs Contract Length
contract_churn = pd.crosstab(
    df["Contract Length"], df["Churn"], normalize="index"
) * 100
contract_churn.plot(kind="bar", figsize=(8, 5))
plt.title("Churn Rate by Contract Length")
plt.xlabel("Contract Length")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.legend(["Not Churn", "Churn"], title="Customer Status")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_vs_contract_length.png")
plt.close()

# Visualization 2: Churn vs Customer Tenure
df.boxplot(column="Tenure", by="Churn", figsize=(7, 5))
plt.title("Customer Tenure vs Churn")
plt.suptitle("")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Tenure")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_vs_tenure.png")
plt.close()

# -------------------- 7. PREDICTION FUNCTION --------------------
def predict_customer_churn(
    age, gender, tenure, usage_frequency, support_calls,
    payment_delay, subscription_type, contract_length,
    total_spend, last_interaction
):
    """Return churn prediction and probability for one customer."""
    customer = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "Usage Frequency": usage_frequency,
        "Support Calls": support_calls,
        "Payment Delay": payment_delay,
        "Subscription Type": subscription_type,
        "Contract Length": contract_length,
        "Total Spend": total_spend,
        "Last Interaction": last_interaction
    }])

    prediction = int(best_model.predict(customer)[0])
    probability = float(best_model.predict_proba(customer)[0, 1])

    label = "Likely to Churn" if prediction == 1 else "Not Likely to Churn"
    return label, probability


# Example prediction generated by the trained model (not hardcoded).
example = predict_customer_churn(
    age=35,
    gender="Female",
    tenure=12,
    usage_frequency=15,
    support_calls=5,
    payment_delay=10,
    subscription_type="Standard",
    contract_length="Monthly",
    total_spend=500,
    last_interaction=10
)
print("\nExample Prediction:")
print("Prediction:", example[0])
print("Churn Probability:", f"{example[1]:.2%}")
