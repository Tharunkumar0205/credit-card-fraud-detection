# 🚨 Credit Card Fraud Detection System

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions using a highly imbalanced real-world dataset.

This project covers:

- Data preprocessing & validation
- Exploratory Data Analysis (EDA)
- Model training & evaluation
- Probability calibration (Platt Scaling)
- Threshold optimization
- Production API deployment using FastAPI

---

## 📊 Dataset

- Source: Kaggle – ULB Credit Card Fraud Detection
- Total Transactions: 283,726
- Fraud Cases: 492 (~0.17%)
- Features: 30 numerical features (PCA-transformed except `Time` and `Amount`)

This dataset represents a realistic fraud detection problem with extreme class imbalance.

---

## 🔎 Project Workflow

### 1️⃣ Data Cleaning & Validation

- Removed 1,081 duplicate transactions
- Verified zero missing values
- Confirmed severe class imbalance
- Validated feature consistency

---

### 2️⃣ Exploratory Data Analysis (EDA)

- Fraud vs non-fraud distribution analysis
- Transaction amount comparison
- Time-based fraud pattern analysis
- Correlation inspection
- Outlier analysis

Key Observations:

- Fraud transactions show different amount behavior
- PCA features are already standardized
- `Time` and `Amount` required scaling

---

### 3️⃣ Model Training

Models Evaluated:

- Logistic Regression (baseline)
- XGBoost (final selected model)

Evaluation Metrics:

- ROC-AUC
- PR-AUC (Primary metric for imbalance)
- Precision
- Recall
- F1 Score
- Confusion Matrix at optimized threshold

---

### 4️⃣ Probability Calibration

Applied Platt Scaling to calibrate model probabilities.

Why?
Raw model probabilities were overconfident. Calibration improves reliability for real-world decision-making.

Saved artifacts:

- `models/xgb_fraud_model_raw.joblib`
- `models/platt_calibrator.joblib`
- `models/threshold.json`
- `models/feature_columns.json`

---

### 5️⃣ FastAPI Deployment

The trained model is deployed as a REST API.

### How to Run

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

uvicorn app:app --reload

### API usage:

curl -X POST "http://127.0.0.1:8000/predict-csv" \
 -H "accept: application/json" \
 -H "Content-Type: multipart/form-data" \
 -F "file=@data/creditcard_clean.csv;type=text/csv"

### Endpoint
