from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import io
import json
import joblib

app = FastAPI(title="Fraud Detection API")

# Load artifacts once
try:
    xgb_raw = joblib.load("models/xgb_fraud_model_raw.joblib")
    platt   = joblib.load("models/platt_calibrator.joblib")
    best_t  = json.load(open("models/threshold.json"))["threshold"]
    cols    = json.load(open("models/feature_columns.json"))["columns"]
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")


@app.post("/predict-csv")
async def predict_csv_api(file: UploadFile = File(...)):

    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file.")

    # Validate required columns
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing_cols}"
        )

    # Enforce correct order
    X = df[cols]

    # Predictions
    raw = xgb_raw.predict_proba(X)[:, 1]
    cal = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
    pred = (cal >= best_t).astype(int)

    # Build result
    res = df.copy()
    res["fraud_proba_raw"] = raw
    res["fraud_proba_cal"] = cal
    res["fraud_pred"] = pred

    # Stream CSV
    buffer = io.StringIO()
    res.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        io.BytesIO(buffer.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=predictions.csv"
        }
    )
