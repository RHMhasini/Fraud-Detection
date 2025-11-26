from fastapi import FastAPI, HTTPException
import sqlite3
import pandas as pd
import joblib
import requests
from datetime import datetime
from pathlib import Path
import logging
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import os

app = FastAPI(title="ML Scoring API")

# Logging Setup
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "ml_scoring_api.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "fraud.db"

RF_MODEL_PATH = MODELS_DIR / "RF_best_model.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgb_fraud_model.pkl"

# Load models
try:
    rf_model = joblib.load(RF_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load models: {str(e)}")
    raise

# Initialize scalers (will be fitted from API data)
robust_scaler = RobustScaler()
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

def get_db():
    return sqlite3.connect(DB_PATH)

def get_training_data_from_api() -> pd.DataFrame:
    """Get training data from API to fit scalers"""
    try:
        # Try to get historical transactions from API
        response = requests.get("http://127.0.0.1:8000/transactions", timeout=5)
        if response.status_code == 200:
            transactions = response.json()
            if transactions and len(transactions) > 0:
                df = pd.DataFrame(transactions)
                # Ensure required columns exist
                required_cols = ['purchase_value', 'age', 'time_to_purchase', 'device_id_count', 'ip_address_count']
                if all(col in df.columns for col in required_cols):
                    return df
        logging.warning("Could not get training data from API, will fit scalers on first transaction")
        return pd.DataFrame()
    except Exception as e:
        logging.warning(f"Could not fetch training data from API: {e}")
        return pd.DataFrame()

def initialize_scalers_from_api():
    """Initialize scalers from API data"""
    global robust_scaler, standard_scaler, minmax_scaler
    try:
        df_train = get_training_data_from_api()
        if not df_train.empty and len(df_train) > 1:
            # Fit scalers on API data
            robust_scaler.fit(df_train[['purchase_value']])
            standard_scaler.fit(df_train[['age', 'time_to_purchase']])
            minmax_scaler.fit(df_train[['device_id_count', 'ip_address_count']])
            logging.info(f"Scalers fitted from API data ({len(df_train)} transactions).")
        else:
            # Scalers will be fitted on first transaction if no data available
            logging.info("No historical data available, scalers will be fitted on first transaction")
    except Exception as e:
        logging.error(f"Failed to initialize scalers: {str(e)}")

# Initialize scalers at startup (will be refitted with more data as transactions come in)
initialize_scalers_from_api()

def get_transaction_from_api(transaction_id: int) -> dict:
    """Get transaction data from main API"""
    try:
        response = requests.get(f"http://127.0.0.1:8000/transactions", timeout=2)
        if response.status_code == 200:
            transactions = response.json()
            for tx in transactions:
                if tx.get('transaction_id') == transaction_id:
                    return tx
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")
    except HTTPException:
        raise
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
                raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")
            return df.iloc[0].to_dict()
        finally:
            conn.close()

@app.post("/score/{transaction_id}")
def score_transaction(transaction_id: int):
    """Calculate ML scores for a transaction"""
    try:
        # Get transaction data from API
        tx_data = get_transaction_from_api(transaction_id)
        
        # Prepare DataFrame
        df = pd.DataFrame([tx_data])
        
        # Ensure required columns exist
        required_cols = ['purchase_value', 'age', 'time_to_purchase', 'device_id_count', 'ip_address_count']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # Apply scaling
        try:
            # If scalers haven't been fitted yet, fit them on this transaction
            # (This is a fallback - ideally scalers should be fitted on historical data)
            if not hasattr(robust_scaler, 'center_') or robust_scaler.center_ is None:
                # Fit scalers on current transaction (not ideal but works)
                robust_scaler.fit(df[['purchase_value']])
                standard_scaler.fit(df[['age', 'time_to_purchase']])
                minmax_scaler.fit(df[['device_id_count', 'ip_address_count']])
                logging.info("Scalers fitted on first transaction")
            
            df[['purchase_value']] = robust_scaler.transform(df[['purchase_value']])
            df[['age', 'time_to_purchase']] = standard_scaler.transform(df[['age', 'time_to_purchase']])
            df[['device_id_count', 'ip_address_count']] = minmax_scaler.transform(df[['device_id_count', 'ip_address_count']])
        except Exception as e:
            logging.warning(f"Scaling failed, using unscaled data: {e}")
        
        # Align columns with model
        feature_cols = rf_model.feature_names_in_
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            for col in missing_cols:
                df[col] = 0
        
        df_model = df[feature_cols]
        
        # Predict scores
        rf_score = float(rf_model.predict_proba(df_model)[:, 1][0])
        xgb_score = float(xgb_model.predict_proba(df_model)[:, 1][0])
        ensemble_score = (rf_score + xgb_score) / 2.0
        
        # Store scores
        conn = get_db()
        try:
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
            
            conn.execute("""
                INSERT INTO ml_scores (transaction_id, RF_score, XGB_score, ensemble_score, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (transaction_id, rf_score, xgb_score, ensemble_score, datetime.now().isoformat()))
            conn.commit()
        finally:
            conn.close()
        
        logging.info(f"ML scores calculated for transaction {transaction_id}: RF={rf_score:.4f}, XGB={xgb_score:.4f}, Ensemble={ensemble_score:.4f}")
        
        return {
            "transaction_id": transaction_id,
            "RF_score": rf_score,
            "XGB_score": xgb_score,
            "ensemble_score": ensemble_score
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error calculating ML scores: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating ML scores: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "ml_scoring_api"}

