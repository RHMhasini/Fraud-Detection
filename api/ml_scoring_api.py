from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import sqlite3
from pathlib import Path
import logging

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

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "fraud.db"

# Model paths
MODEL_PATH = MODELS_DIR / "fraud_detection_xgboost_v1_BEST.pkl"
STANDARD_SCALER_PATH = MODELS_DIR / "standard_scaler_v1.pkl"
MINMAX_SCALER_PATH = MODELS_DIR / "minmax_scaler_v1.pkl"
ROBUST_SCALER_PATH = MODELS_DIR / "robust_scaler_v1.pkl"

# Load model and scalers
try:
    xgb_model = joblib.load(MODEL_PATH)
    standard_scaler = joblib.load(STANDARD_SCALER_PATH)
    minmax_scaler = joblib.load(MINMAX_SCALER_PATH)
    robust_scaler = joblib.load(ROBUST_SCALER_PATH)
    logging.info("Model and scalers loaded successfully")
    print(" Model and scalers loaded successfully")
except Exception as e:
    logging.error(f"Error loading model/scalers: {e}")
    print(f"Error loading model/scalers: {e}")
    xgb_model = None

# Expected feature order from training
EXPECTED_FEATURES = [
    'purchase_value', 'age', 'time_to_purchase',
    'device_id_count', 'ip_address_count',
    'source_Ads', 'source_Direct', 'source_SEO',
    'browser_Chrome', 'browser_FireFox', 'browser_IE', 
    'browser_Opera', 'browser_Safari',
    'sex_F', 'sex_M'
]

def get_db():
    return sqlite3.connect(DB_PATH)

def get_transaction_from_db(transaction_id: int):
    """Fetch transaction from database"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return None
    
    columns = [desc[0] for desc in cursor.description]
    conn.close()
    
    return dict(zip(columns, row))

def preprocess_for_model(transaction: dict) -> pd.DataFrame:
    """Preprocess transaction data for model prediction"""
    
    # Extract raw features
    df = pd.DataFrame([{
        'purchase_value': transaction.get('purchase_value', 0),
        'age': transaction.get('age', 0),
        'time_to_purchase': transaction.get('time_to_purchase', 0),
        'device_id_count': transaction.get('device_id_count', 0),
        'ip_address_count': transaction.get('ip_address_count', 0),
        'source': transaction.get('source', 'Direct'),
        'browser': transaction.get('browser', 'Chrome'),
        'sex': transaction.get('sex', 'M')
    }])
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['source', 'browser', 'sex'], drop_first=False)
    
    # Scale numerical features
    df[['age', 'time_to_purchase']] = standard_scaler.transform(
        df[['age', 'time_to_purchase']]
    )
    
    df[['device_id_count', 'ip_address_count']] = minmax_scaler.transform(
        df[['device_id_count', 'ip_address_count']]
    )
    
    df[['purchase_value']] = robust_scaler.transform(
        df[['purchase_value']]
    )
    
    # Align with expected features
    for feature in EXPECTED_FEATURES:
        if feature not in df.columns:
            df[feature] = 0
    
    df = df[EXPECTED_FEATURES]
    
    return df

@app.post("/score/{transaction_id}")
def score_transaction(transaction_id: int):
    """Score a transaction using XGBoost model"""
    
    if xgb_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Get transaction
    transaction = get_transaction_from_db(transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    try:
        # Preprocess
        X = preprocess_for_model(transaction)
        
        # Predict probability
        fraud_probability = float(xgb_model.predict_proba(X)[0, 1])
        
        # Store score in database
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_scores (
                score_id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER UNIQUE,
                xgb_score REAL,
                ensemble_score REAL,
                created_at TEXT
            )
        """)
        
        cursor.execute("""
            INSERT OR REPLACE INTO ml_scores 
            (transaction_id, xgb_score, ensemble_score, created_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (transaction_id, fraud_probability, fraud_probability))
        
        conn.commit()
        conn.close()
        
        logging.info(f"Transaction {transaction_id} scored: {fraud_probability:.4f}")
        
        return {
            "transaction_id": transaction_id,
            "xgb_score": fraud_probability,
            "ensemble_score": fraud_probability,
            "model": "XGBoost V1 (71.41% recall)"
        }
        
    except Exception as e:
        logging.error(f"Scoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": xgb_model is not None,
        "model": "XGBoost V1",
        "recall": "71.41%"
    }