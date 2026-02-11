from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import pandas as pd
import numpy as np
import io, json, joblib

app = FastAPI()

xgb_raw = joblib.load("models/xgb_fraud_model_raw.joblib")
platt   = joblib.load("models/platt_calibrator.joblib")
best_t  = json.load(open("models/threshold.json"))["threshold"]
cols    = json.load(open("models/feature_columns.json"))["columns"]

@app.post(
    "/predict-csv",
    responses={200: {"content": {"text/csv": {}}}},
)
async def predict_csv_api(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    X = df[cols]

    raw = xgb_raw.predict_proba(X)[:, 1]
    cal = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
    pred = (cal >= best_t).astype(int)

    res = df.copy()
    res["fraud_proba_raw"] = raw
    res["fraud_proba_cal"] = cal
    res["fraud_pred"] = pred

    csv_bytes = res.to_csv(index=False).encode("utf-8")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
