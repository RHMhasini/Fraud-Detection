from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import logging
from typing import List

app = FastAPI(title="Fraud Detection API")

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

MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "database"

RF_MODEL_PATH = MODELS_DIR / "RF_best_model.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgb_fraud_model.pkl"
DB_PATH = DATABASE_DIR / "fraud.db"

# -------------------
# Load trained models
# -------------------
try:
    rf_model = joblib.load(RF_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load models: {str(e)}")
    raise

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
        sex_M INTEGER
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

# -------------------
# Pydantic model for API input
# -------------------
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
# Validation Function
# -------------------
def validate_transaction(tx: Transaction):
    errors = []
    
    # Check datetime formats
    try:
        pd.to_datetime(tx.signup_time)
    except ValueError:
        errors.append("signup_time must be a valid ISO datetime string.")
    
    try:
        pd.to_datetime(tx.purchase_time)
    except ValueError:
        errors.append("purchase_time must be a valid ISO datetime string.")
    
    # Check ranges
    if tx.purchase_value <= 0:
        errors.append("purchase_value must be positive.")
    
    if not (18 <= tx.age <= 100):
        errors.append("age must be between 18 and 100.")
    
    # Check binary fields (0 or 1)
    binary_fields = [
        tx.source_Ads, tx.source_Direct, tx.source_SEO,
        tx.browser_Chrome, tx.browser_FireFox, tx.browser_IE, tx.browser_Opera, tx.browser_Safari,
        tx.sex_F, tx.sex_M
    ]
    for field in binary_fields:
        if field not in [0, 1]:
            errors.append(f"Binary fields must be 0 or 1 (found invalid value: {field}).")
    
    # Check hashed fields are non-empty
    if not tx.hashed_user_id.strip():
        errors.append("hashed_user_id cannot be empty.")
    if not tx.hashed_device_id.strip():
        errors.append("hashed_device_id cannot be empty.")
    if not tx.hashed_ip_address.strip():
        errors.append("hashed_ip_address cannot be empty.")
    
    if errors:
        raise HTTPException(status_code=400, detail={"validation_errors": errors})

# -------------------
# Endpoint to insert transaction + compute ML score
# -------------------
@app.post("/transaction")
def add_transaction(tx: Transaction):
    try:
        # Input Validation & Preprocessing
        validate_transaction(tx)
        logging.info(f"Transaction validation passed for hashed_user_id: {tx.hashed_user_id}")
        
        df = pd.DataFrame([tx.dict()])
        
        # Compute derived feature
        df['time_to_purchase'] = (pd.to_datetime(df['purchase_time']) - pd.to_datetime(df['signup_time'])).dt.total_seconds()
        
        # Align columns with model
        feature_cols = rf_model.feature_names_in_
        df_model = df[feature_cols]
        
        # Predict scores
        rf_score = float(rf_model.predict_proba(df_model)[:, 1][0])
        xgb_score = float(xgb_model.predict_proba(df_model)[:, 1][0])
        ensemble_score = (rf_score + xgb_score) / 2.0
        
        # Insert transaction and scores
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO transactions (
                    signup_time, purchase_time, purchase_value, age, time_to_purchase,
                    device_id_count, ip_address_count, hashed_user_id, hashed_device_id, hashed_ip_address,
                    source_Ads, source_Direct, source_SEO,
                    browser_Chrome, browser_FireFox, browser_IE, browser_Opera, browser_Safari,
                    sex_F, sex_M
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, tuple(df.iloc[0]))
            transaction_id = cur.lastrowid
            
            cur.execute("""
                INSERT INTO ml_scores (transaction_id, RF_score, XGB_score, ensemble_score, created_at)
                VALUES (?,?,?,?,?)
            """, (transaction_id, rf_score, xgb_score, ensemble_score, datetime.now().isoformat()))
            conn.commit()
        
        logging.info(f"Transaction inserted successfully: ID {transaction_id}, Ensemble Score {ensemble_score}")
        return {
            "transaction_id": transaction_id,
            "RF_score": rf_score,
            "XGB_score": xgb_score,
            "ensemble_score": ensemble_score
        }
    
    except HTTPException:
        raise  # Re-raise validation errors
    except Exception as e:
        error_msg = f"Error processing transaction: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}

# -------------------
# Batch Transaction Endpoint
# -------------------
@app.post("/transactions_batch")
def add_transactions_batch(txs: List[Transaction]):
    results = []
    for i, tx in enumerate(txs):
        try:
            # Reuse single transaction logic
            result = add_transaction(tx)
            results.append({"index": i, "status": "success", **result})
        except Exception as e:
            error_msg = f"Failed for transaction {i}: {str(e)}"
            logging.error(error_msg)
            results.append({"index": i, "status": "error", "error": error_msg})
    
    logging.info(f"Batch processed: {len(results)} transactions")
    return {"batch_results": results}

# -------------------
# Endpoint to fetch all transactions
# -------------------
@app.get("/transactions")
def get_transactions():
    try:
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM transactions").fetchall()
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

# -------------------
# Endpoint to fetch all ML scores
# -------------------
@app.get("/ml_scores")
def get_ml_scores():
    try:
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM ml_scores").fetchall()
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
