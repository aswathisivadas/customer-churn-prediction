# Customer Churn Prediction - AI/ML Practical Test

## Objective
Build a machine-learning system that predicts whether a customer is likely to churn.

## Dataset
The provided customer churn CSV is stored in `data/`.

Target variable:
- `Churn` — 1 = churn, 0 = not churn

Identifier:
- `CustomerID` is removed before training because it is only an identifier.

## Workflow
1. Load and explore the dataset
2. Check missing values and duplicates
3. Separate features and target
4. Encode categorical variables using One-Hot Encoding
5. Impute missing values
6. Scale numerical features
7. Split data into 80% training and 20% testing
8. Train Logistic Regression and Random Forest
9. Evaluate using Accuracy, Precision, Recall, F1 Score and Confusion Matrix
10. Select the model with the highest F1 score
11. Generate churn prediction and probability for a new customer

## Models
- Logistic Regression
- Random Forest

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python customer_churn_prediction.py
```

The script creates:
- model comparison CSV
- confusion matrix images
- churn analysis visualizations
- trained best model (`best_model.joblib`)

## Important observations from the provided dataset
- Dataset contains 64,374 rows and 12 columns.
- There are no missing values in the provided data.
- There are no duplicate rows.
- `Churn` is the target variable.
- The churn classes are reasonably balanced: 33,881 non-churn and 30,493 churn customers.
- Monthly-contract customers have a higher observed churn rate than annual and quarterly customers.
- Female customers show a higher observed churn rate than male customers in this dataset.

## Example result from the provided dataset
Using the script settings (`random_state=42`, 80/20 stratified split), the models produced approximately:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.70% | 81.37% | 82.34% | 81.85% |
| Random Forest | 99.86% | 99.95% | 99.75% | 99.85% |

Random Forest is selected by highest F1 score for this run.

## Files
- `customer_churn_prediction.py` - complete ML program
- `data/customer_churn_dataset-testing-master.csv` - provided dataset
- `requirements.txt` - Python dependencies
- `outputs/` - generated charts, metrics and trained model
