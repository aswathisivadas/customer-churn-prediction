# Customer Churn Prediction

## Objective

This project predicts whether a customer is likely to churn using customer and service usage information.

## Dataset

The dataset contains customer information such as age, gender, tenure, usage frequency, support calls, payment delay, subscription type, contract length, total spend and last interaction.

The target variable is `Churn`.

- 0 = Not Churn
- 1 = Churn

## Steps Performed

1. Loaded and explored the dataset
2. Checked for missing values and duplicate records
3. Removed CustomerID from the features
4. Encoded categorical variables
5. Scaled numerical features
6. Split the dataset into training and testing data
7. Trained two classification models
8. Compared the models using evaluation metrics
9. Created visualizations related to customer churn
10. Created a function to predict churn for a new customer

## Machine Learning Models

- Logistic Regression
- Random Forest

## Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The better-performing model was selected for making predictions.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
