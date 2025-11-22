from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime

app = FastAPI(title="Transaction Ingest & Preprocessing API")

# Database connection
def get_db():
    return sqlite3.connect("database/fraud.db")

# Incoming raw model
class RawTransaction(BaseModel):
    signup_time: str
    purchase_time: str
    purchase_value: float
    age: int
    device_id: str
    ip_address: str
    user_id: str
    source: str
    browser: str
    sex: str

def preprocess(tx: RawTransaction):
    df = pd.DataFrame([tx.dict()])

    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df["time_to_purchase"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()

    df["device_id_count"] = 1
    df["ip_address_count"] = 1

    df["hashed_user_id"] = df["user_id"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
    df["hashed_device_id"] = df["device_id"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
    df["hashed_ip_address"] = df["ip_address"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

    df["class"] = 0  # unknown during ingestion

    df = pd.get_dummies(df, columns=["source", "browser", "sex"], drop_first=False)

    return df

@app.post("/ingest")
def ingest(tx: RawTransaction):
    df = preprocess(tx)

    conn = get_db()
    df.to_sql("processed_transactions", conn, if_exists="append", index=False)
    conn.close()

    return {"message": "Transaction processed & stored", "columns": list(df.columns)}
