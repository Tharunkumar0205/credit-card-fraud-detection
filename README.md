# Credit Card Fraud Detection

This project demonstrates an end-to-end machine learning workflow for detecting fraudulent credit card transactions using a real-world, highly imbalanced dataset.

## Dataset
- Source: Kaggle Credit Card Fraud Detection (ULB)
- Transactions: ~283K (after deduplication)
- Fraud rate: ~0.17%

## Key Steps
- Data validation and cleaning
- Duplicate transaction removal
- Class imbalance analysis
- Exploratory Data Analysis (EDA)
- Baseline model preparation (next phase)

## EDA Highlights
- No missing values across features
- 1,081 duplicate transactions removed
- Extreme class imbalance confirmed
- PCA-transformed features already standardized
- `Time` and `Amount` require scaling

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Jupyter Notebook

## Next Steps
- Baseline Logistic Regression model
- Proper evaluation using ROC-AUC and PR-AUC
- Threshold tuning
- API deployment using FastAPI
