from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Optional
import hashlib

app = FastAPI(title="Fraud Detection API - Main Orchestrator")

# -------------------
# Logging Setup
# -------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "api.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------
# Resolve project paths
# -------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "fraud.db"

# -------------------
# Service URLs (assuming all services run on different ports or same port with different paths)
# In production, these would be environment variables or service discovery
# -------------------
INGEST_API_URL = "http://127.0.0.1:8001"
ANOMALY_API_URL = "http://127.0.0.1:8002"
ML_SCORING_API_URL = "http://127.0.0.1:8003"
VERIFIER_API_URL = "http://127.0.0.1:8004"
ALERT_API_URL = "http://127.0.0.1:8005"

# For development, we can run all services on the same port with different paths
# Or use a single service that handles all operations
USE_MONOLITHIC = True  # Set to False to use separate services

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# -------------------
# Create tables if not exist
# -------------------
with get_connection() as conn:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        signup_time TEXT,
        purchase_time TEXT,
        purchase_value REAL,
        age REAL,
        time_to_purchase REAL,
        device_id_count REAL,
        ip_address_count REAL,
        hashed_user_id TEXT,
        hashed_device_id TEXT,
        hashed_ip_address TEXT,
        source_Ads INTEGER,
        source_Direct INTEGER,
        source_SEO INTEGER,
        browser_Chrome INTEGER,
        browser_FireFox INTEGER,
        browser_IE INTEGER,
        browser_Opera INTEGER,
        browser_Safari INTEGER,
        sex_F INTEGER,
        sex_M INTEGER,
        created_at TEXT
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS ml_scores (
        score_id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id INTEGER,
        RF_score REAL,
        XGB_score REAL,
        ensemble_score REAL,
        created_at TEXT
    )
    """)
    conn.commit()

# -------------------
# Pydantic models
# -------------------
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

class Transaction(BaseModel):
    signup_time: str
    purchase_time: str
    purchase_value: float
    age: float
    device_id_count: float
    ip_address_count: float
    hashed_user_id: str
    hashed_device_id: str
    hashed_ip_address: str
    source_Ads: int
    source_Direct: int
    source_SEO: int
    browser_Chrome: int
    browser_FireFox: int
    browser_IE: int
    browser_Opera: int
    browser_Safari: int
    sex_F: int
    sex_M: int

# -------------------
# Helper Functions
# -------------------
def call_service(url: str, method: str = "GET", data: Optional[dict] = None, timeout: int = 5):
    """Call a microservice API"""
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 200:
            return response.json()
        else:
            logging.warning(f"Service call failed: {url}, status={response.status_code}")
            return None
    except Exception as e:
        logging.warning(f"Service call error: {url}, error={str(e)}")
        return None

# -------------------
# Main Transaction Flow Endpoint
# -------------------
@app.post("/transaction")
def process_transaction(tx: RawTransaction):
    """
    Main endpoint that orchestrates the fraud detection flow:
    1. Ingest transaction
    2. Detect anomalies
    3. Calculate ML scores
    4. Verify transaction
    5. Generate alerts if needed
    """
    try:
        # Step 1: Ingest transaction
        logging.info(f"Processing transaction for user: {tx.user_id}")
        
        if USE_MONOLITHIC:
            # Use local ingestion logic
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from ingest_api import calculate_counts_from_db
            import pandas as pd
            
            # Preprocess
            df = pd.DataFrame([tx.dict()])
            df["signup_time"] = pd.to_datetime(df["signup_time"])
            df["purchase_time"] = pd.to_datetime(df["purchase_time"])
            df["time_to_purchase"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
            
            device_id_count, ip_address_count = calculate_counts_from_db(tx.device_id, tx.ip_address)
            df["device_id_count"] = device_id_count
            df["ip_address_count"] = ip_address_count
            
            df["hashed_user_id"] = df["user_id"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
            df["hashed_device_id"] = df["device_id"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
            df["hashed_ip_address"] = df["ip_address"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
            df = pd.get_dummies(df, columns=["source", "browser", "sex"], drop_first=False)
            
            # Store transaction
            row = df.iloc[0]
            with get_connection() as conn:
                conn.execute("""
                    INSERT INTO transactions (
                        signup_time, purchase_time, purchase_value, age, time_to_purchase,
                        device_id_count, ip_address_count, hashed_user_id, hashed_device_id, hashed_ip_address,
                        source_Ads, source_Direct, source_SEO,
                        browser_Chrome, browser_FireFox, browser_IE, browser_Opera, browser_Safari,
                        sex_F, sex_M, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row.get('signup_time', tx.signup_time)),
                    str(row.get('purchase_time', tx.purchase_time)),
                    float(row.get('purchase_value', tx.purchase_value)),
                    float(row.get('age', tx.age)),
                    float(row.get('time_to_purchase', 0)),
                    float(row.get('device_id_count', 1)),
                    float(row.get('ip_address_count', 1)),
                    str(row.get('hashed_user_id', '')),
                    str(row.get('hashed_device_id', '')),
                    str(row.get('hashed_ip_address', '')),
                    int(row.get('source_Ads', 0)),
                    int(row.get('source_Direct', 0)),
                    int(row.get('source_SEO', 0)),
                    int(row.get('browser_Chrome', 0)),
                    int(row.get('browser_FireFox', 0)),
                    int(row.get('browser_IE', 0)),
                    int(row.get('browser_Opera', 0)),
                    int(row.get('browser_Safari', 0)),
                    int(row.get('sex_F', 0)),
                    int(row.get('sex_M', 0)),
                    datetime.now().isoformat()
                ))
                transaction_id = conn.lastrowid
                conn.commit()
        else:
            # Call ingest service
            ingest_result = call_service(f"{INGEST_API_URL}/ingest", "POST", tx.dict())
            if not ingest_result:
                raise HTTPException(status_code=500, detail="Failed to ingest transaction")
            transaction_id = ingest_result.get("transaction_id")
        
        logging.info(f"Transaction ingested: ID {transaction_id}")
        
        # Step 2: Detect anomalies
        if USE_MONOLITHIC:
            from anomaly_api import detect_anomalies
            # Get transaction data from database
            with get_connection() as conn:
                df = pd.read_sql("SELECT * FROM transactions WHERE transaction_id = ?", conn, params=[transaction_id])
                if df.empty:
                    raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")
                tx_data = df.iloc[0].to_dict()
            anomaly_result = detect_anomalies(tx_data)
        else:
            anomaly_result = call_service(f"{ANOMALY_API_URL}/detect/{transaction_id}", "POST")
        
        logging.info(f"Anomaly detection completed for transaction {transaction_id}: {anomaly_result}")
        
        # Step 3: Calculate ML scores
        if USE_MONOLITHIC:
            from ml_scoring_api import score_transaction
            ml_result = score_transaction(transaction_id)
        else:
            ml_result = call_service(f"{ML_SCORING_API_URL}/score/{transaction_id}", "POST")
        
        if not ml_result:
            raise HTTPException(status_code=500, detail="Failed to calculate ML scores")
        
        logging.info(f"ML scores calculated for transaction {transaction_id}: {ml_result}")
        
        # Step 4: Verify transaction
        if USE_MONOLITHIC:
            from verifier_api import verify_transaction
            verify_result = verify_transaction(transaction_id)
        else:
            verify_result = call_service(f"{VERIFIER_API_URL}/verify/{transaction_id}", "POST")
        
        if not verify_result:
            verify_result = {
                "transaction_id": transaction_id,
                "ensemble_score": ml_result.get("ensemble_score"),
                "status": "unknown",
                "explanation": None
            }
        
        logging.info(f"Transaction verified: {verify_result}")
        
        # Step 5: Generate alerts if high-risk
        alert_result = None
        if verify_result.get("status") in ["flag", "deny"]:
            if USE_MONOLITHIC:
                from alert_api import generate_alerts
                alert_result = generate_alerts()
            else:
                alert_result = call_service(f"{ALERT_API_URL}/generate_alerts", "POST")
            logging.info(f"Alerts generated: {alert_result}")
        
        return {
            "transaction_id": transaction_id,
            "anomaly_detection": anomaly_result,
            "ml_scores": ml_result,
            "verification": verify_result,
            "alerts": alert_result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error processing transaction: {str(e)}"
        logging.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# -------------------
# Data Retrieval Endpoints
# -------------------
@app.get("/transactions")
def get_transactions():
    """Get all transactions from database"""
    try:
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM transactions ORDER BY transaction_id DESC").fetchall()
            result = []
            for row in rows:
                d = dict(row)
                for k, v in d.items():
                    if isinstance(v, bytes):
                        try:
                            d[k] = v.decode('utf-8')
                        except UnicodeDecodeError:
                            d[k] = v.decode('latin-1', errors='ignore')
                result.append(d)
            return result
    except Exception as e:
        logging.error(f"Error fetching transactions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/ml_scores")
def get_ml_scores():
    """Get all ML scores from database"""
    try:
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM ml_scores ORDER BY transaction_id DESC").fetchall()
            result = []
            for row in rows:
                d = dict(row)
                for k, v in d.items():
                    if isinstance(v, bytes):
                        try:
                            d[k] = v.decode('utf-8')
                        except UnicodeDecodeError:
                            d[k] = v.decode('latin-1', errors='ignore')
                result.append(d)
            return result
    except Exception as e:
        logging.error(f"Error fetching ML scores: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/verify/{transaction_id}")
def verify_transaction_endpoint(transaction_id: int):
    """Verify a specific transaction"""
    try:
        if USE_MONOLITHIC:
            from verifier_api import verify_transaction
            return verify_transaction(transaction_id)
        else:
            result = call_service(f"{VERIFIER_API_URL}/verify/{transaction_id}", "POST")
            if not result:
                raise HTTPException(status_code=404, detail="Transaction not found")
            return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error verifying transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/verify_last")
def verify_last_transaction():
    """Verify the last transaction"""
    try:
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM ml_scores ORDER BY transaction_id DESC LIMIT 1").fetchall()
            if not rows:
                raise HTTPException(status_code=404, detail="No transactions found")
            transaction_id = dict(rows[0]).get("transaction_id")
            return verify_transaction_endpoint(transaction_id)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error verifying last transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/alerts")
def get_alerts():
    """Get all alerts"""
    try:
        if USE_MONOLITHIC:
            from alert_api import get_alerts
            return get_alerts()
        else:
            result = call_service(f"{ALERT_API_URL}/alerts", "GET")
            return result if result else []
    except Exception as e:
        logging.error(f"Error fetching alerts: {str(e)}")
        return []

@app.get("/health")
def health():
    return {"status": "healthy", "service": "main_orchestrator"}
