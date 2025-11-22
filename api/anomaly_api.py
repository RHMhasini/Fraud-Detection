from fastapi import FastAPI
import sqlite3
import pandas as pd
import joblib
from datetime import datetime

app = FastAPI(title="Anomaly Detector API")

rf_model = joblib.load("models/RF_best_model.pkl")
xgb_model = joblib.load("models/xgb_fraud_model.pkl")
MODEL_FEATURES = rf_model.feature_names_in_.tolist()

def get_db():
    return sqlite3.connect("database/fraud.db")

@app.post("/score/{transaction_id}")
def score(transaction_id: int):
    conn = get_db()

    df = pd.read_sql(f"SELECT * FROM processed_transactions LIMIT 1 OFFSET {transaction_id-1}", conn)
    conn.close()

    X = df[MODEL_FEATURES]

    rf = float(rf_model.predict_proba(X)[0][1])
    xgb = float(xgb_model.predict_proba(X)[0][1])
    ensemble = (rf + xgb) / 2

    conn = get_db()
    conn.execute(
        "INSERT INTO model_scores (transaction_id, rf_score, xgb_score, ensemble_score, timestamp) VALUES (?, ?, ?, ?, ?)",
        (transaction_id, rf, xgb, ensemble, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    return {"rf": rf, "xgb": xgb, "ensemble": ensemble}
