from fastapi import FastAPI, HTTPException
import sqlite3
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import logging

app = FastAPI(title="Alert Agent API")

# Logging Setup
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "alert_api.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "fraud.db"

# Threshold for high-risk transactions
THRESHOLD_DENY = 0.6

def get_db():
    return sqlite3.connect(DB_PATH)

def get_high_risk_transactions_from_api() -> list:
    """Get high-risk transactions from main API"""
    try:
        # Get transactions and scores from API
        trans_response = requests.get("http://127.0.0.1:8000/transactions", timeout=5)
        scores_response = requests.get("http://127.0.0.1:8000/ml_scores", timeout=5)
        
        if trans_response.status_code != 200 or scores_response.status_code != 200:
            return []
        
        transactions = trans_response.json()
        scores = scores_response.json()
        
        if not transactions or not scores:
            return []
        
        # Merge and filter high-risk
        trans_df = pd.DataFrame(transactions)
        scores_df = pd.DataFrame(scores)
        
        merged_df = trans_df.merge(
            scores_df[["transaction_id", "ensemble_score"]],
            on="transaction_id",
            how="inner"
        )
        
        high_risk = merged_df[merged_df["ensemble_score"] >= THRESHOLD_DENY].copy()
        return high_risk.to_dict(orient="records")
    
    except Exception as e:
        logging.warning(f"Could not fetch high-risk transactions from API: {e}")
        return []

def get_verification_from_api(transaction_id: int) -> dict:
    """Get verification result from verifier API"""
    try:
        response = requests.post(f"http://127.0.0.1:8004/verify/{transaction_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logging.warning(f"Could not fetch verification from API: {e}")
        return None

@app.post("/generate_alerts")
def generate_alerts():
    """Generate alerts for high-risk transactions"""
    try:
        # Get high-risk transactions from API
        high_risk_transactions = get_high_risk_transactions_from_api()
        
        if not high_risk_transactions:
            return {
                "alert_count": 0,
                "alerts": [],
                "message": "No high-risk transactions found"
            }
        
        alerts = []
        for tx in high_risk_transactions:
            transaction_id = tx.get('transaction_id')
            ensemble_score = tx.get('ensemble_score')
            
            # Get verification details
            verification = get_verification_from_api(transaction_id)
            
            alert = {
                "transaction_id": transaction_id,
                "ensemble_score": ensemble_score,
                "purchase_value": tx.get('purchase_value'),
                "hashed_user_id": tx.get('hashed_user_id'),
                "purchase_time": tx.get('purchase_time'),
                "status": verification.get('status') if verification else "deny",
                "explanation": verification.get('explanation') if verification else f"High-risk transaction with score {ensemble_score:.3f}",
                "alert_level": "CRITICAL" if ensemble_score >= 0.8 else "HIGH"
            }
            alerts.append(alert)
            
            # Store alert
            conn = get_db()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        transaction_id INTEGER,
                        ensemble_score REAL,
                        alert_level TEXT,
                        explanation TEXT,
                        created_at TEXT
                    )
                """)
                
                conn.execute("""
                    INSERT INTO alerts 
                    (transaction_id, ensemble_score, alert_level, explanation, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    transaction_id, ensemble_score, alert['alert_level'], alert['explanation'],
                    datetime.now().isoformat()
                ))
                conn.commit()
            finally:
                conn.close()
        
        logging.info(f"Generated {len(alerts)} alerts for high-risk transactions")
        
        return {
            "alert_count": len(alerts),
            "alerts": alerts,
            "message": f"Generated {len(alerts)} alert(s) for high-risk transactions"
        }
    
    except Exception as e:
        logging.error(f"Error generating alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating alerts: {str(e)}")

@app.get("/alerts")
def get_alerts():
    """Get all stored alerts"""
    try:
        conn = get_db()
        try:
            df = pd.read_sql("SELECT * FROM alerts ORDER BY created_at DESC", conn)
            return df.to_dict(orient="records")
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"Error fetching alerts: {e}")
        return []

@app.get("/health")
def health():
    return {"status": "healthy", "service": "alert_api"}

