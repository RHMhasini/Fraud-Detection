from fastapi import FastAPI, HTTPException
import sqlite3
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional

app = FastAPI(title="Anomaly Detector API")

# Logging Setup
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "anomaly_api.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "fraud.db"

# Data API endpoint (to get historical data)
DATA_API_URL = "http://127.0.0.1:8001"  # Assuming data API runs on different port

def get_db():
    return sqlite3.connect(DB_PATH)

def get_transaction_from_api(transaction_id: int) -> Optional[dict]:
    """Get transaction data from data API"""
    try:
        # Try to get from main API first
        response = requests.get(f"http://127.0.0.1:8000/transactions", timeout=2)
        if response.status_code == 200:
            transactions = response.json()
            for tx in transactions:
                if tx.get('transaction_id') == transaction_id:
                    return tx
        return None
    except Exception as e:
        logging.warning(f"Could not fetch transaction from API: {e}")
        # Fallback to database
        conn = get_db()
        try:
            df = pd.read_sql(
                "SELECT * FROM transactions WHERE transaction_id = ?",
                conn, params=[transaction_id]
            )
            if df.empty:
                return None
            return df.iloc[0].to_dict()
        finally:
            conn.close()

def detect_anomalies(transaction_data: dict) -> dict:
    """Detect anomalies in transaction data"""
    anomalies = []
    anomaly_score = 0.0
    
    # Check for high purchase value
    purchase_value = transaction_data.get('purchase_value', 0)
    if purchase_value > 100000:  # Threshold for high value
        anomalies.append("High purchase value detected")
        anomaly_score += 0.3
    
    # Check for unusual device/IP counts
    device_count = transaction_data.get('device_id_count', 1)
    ip_count = transaction_data.get('ip_address_count', 1)
    
    if device_count > 10:
        anomalies.append(f"Unusual device count: {device_count}")
        anomaly_score += 0.2
    
    if ip_count > 10:
        anomalies.append(f"Unusual IP count: {ip_count}")
        anomaly_score += 0.2
    
    # Check for very short time to purchase (potential fraud)
    time_to_purchase = transaction_data.get('time_to_purchase', 0)
    if time_to_purchase < 60:  # Less than 1 minute
        anomalies.append("Very short time between signup and purchase")
        anomaly_score += 0.3
    
    # Check for unusual age
    age = transaction_data.get('age', 0)
    if age < 18 or age > 100:
        anomalies.append(f"Unusual age: {age}")
        anomaly_score += 0.1
    
    # Normalize anomaly score to 0-1 range
    anomaly_score = min(anomaly_score, 1.0)
    
    return {
        "has_anomalies": len(anomalies) > 0,
        "anomaly_score": anomaly_score,
        "anomalies": anomalies,
        "anomaly_count": len(anomalies)
    }

@app.post("/detect/{transaction_id}")
def detect_anomaly(transaction_id: int):
    """Detect anomalies for a specific transaction"""
    try:
        # Get transaction data from API
        transaction_data = get_transaction_from_api(transaction_id)
        
        if not transaction_data:
            # Fallback: try database directly
            conn = get_db()
            try:
                df = pd.read_sql(
                    "SELECT * FROM transactions WHERE transaction_id = ?",
                    conn, params=[transaction_id]
                )
                if df.empty:
                    raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")
                transaction_data = df.iloc[0].to_dict()
            finally:
                conn.close()
        
        # Detect anomalies
        result = detect_anomalies(transaction_data)
        
        # Store anomaly detection result
        conn = get_db()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_detections (
                    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id INTEGER,
                    has_anomalies INTEGER,
                    anomaly_score REAL,
                    anomaly_count INTEGER,
                    anomalies TEXT,
                    created_at TEXT
                )
            """)
            
            import json
            conn.execute("""
                INSERT INTO anomaly_detections 
                (transaction_id, has_anomalies, anomaly_score, anomaly_count, anomalies, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                transaction_id,
                1 if result['has_anomalies'] else 0,
                result['anomaly_score'],
                result['anomaly_count'],
                json.dumps(result['anomalies']),
                datetime.now().isoformat()
            ))
            conn.commit()
        finally:
            conn.close()
        
        logging.info(f"Anomaly detection completed for transaction {transaction_id}: {result}")
        
        return {
            "transaction_id": transaction_id,
            **result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting anomalies: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "anomaly_api"}
