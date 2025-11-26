from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
import logging

app = FastAPI(title="Transaction Ingest & Preprocessing API")

# Logging Setup
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "ingest_api.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "fraud.db"

# Database connection
def get_db():
    return sqlite3.connect(DB_PATH)

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

def calculate_counts_from_db(device_id: str, ip_address: str) -> tuple:
    """Calculate device_id_count and ip_address_count from existing transactions in database"""
    try:
        conn = get_db()
        # Get counts from raw_transactions table if it exists, otherwise from transactions table
        try:
            device_count = pd.read_sql(
                "SELECT COUNT(*) as count FROM raw_transactions WHERE device_id = ?",
                conn, params=[device_id]
            ).iloc[0]['count']
            ip_count = pd.read_sql(
                "SELECT COUNT(*) as count FROM raw_transactions WHERE ip_address = ?",
                conn, params=[ip_address]
            ).iloc[0]['count']
        except:
            # Fallback: try to get from transactions table using hashed values
            hashed_device = hashlib.sha256(device_id.encode()).hexdigest()
            hashed_ip = hashlib.sha256(ip_address.encode()).hexdigest()
            device_count = pd.read_sql(
                "SELECT COUNT(*) as count FROM transactions WHERE hashed_device_id = ?",
                conn, params=[hashed_device]
            ).iloc[0]['count']
            ip_count = pd.read_sql(
                "SELECT COUNT(*) as count FROM transactions WHERE hashed_ip_address = ?",
                conn, params=[hashed_ip]
            ).iloc[0]['count']
        conn.close()
        # Add 1 for current transaction
        return float(device_count + 1), float(ip_count + 1)
    except Exception as e:
        logging.warning(f"Could not calculate counts from DB: {e}. Using default value 1.")
        return 1.0, 1.0

def preprocess(tx: RawTransaction):
    """Preprocess raw transaction"""
    df = pd.DataFrame([tx.dict()])

    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df["time_to_purchase"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()

    # Calculate counts from database via API (or from DB directly)
    device_id_count, ip_address_count = calculate_counts_from_db(tx.device_id, tx.ip_address)
    df["device_id_count"] = device_id_count
    df["ip_address_count"] = ip_address_count

    # Hash IDs
    df["hashed_user_id"] = df["user_id"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
    df["hashed_device_id"] = df["device_id"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
    df["hashed_ip_address"] = df["ip_address"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=["source", "browser", "sex"], drop_first=False)

    return df

@app.post("/ingest")
def ingest(tx: RawTransaction):
    """Ingest raw transaction, preprocess and store"""
    try:
        # Preprocess transaction
        df = preprocess(tx)
        
        # Store raw transaction first
        conn = get_db()
        try:
            # Create raw_transactions table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_transactions (
                    raw_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signup_time TEXT,
                    purchase_time TEXT,
                    purchase_value REAL,
                    age INTEGER,
                    device_id TEXT,
                    ip_address TEXT,
                    user_id TEXT,
                    source TEXT,
                    browser TEXT,
                    sex TEXT,
                    created_at TEXT
                )
            """)
            
            # Insert raw transaction
            conn.execute("""
                INSERT INTO raw_transactions 
                (signup_time, purchase_time, purchase_value, age, device_id, ip_address, 
                 user_id, source, browser, sex, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tx.signup_time, tx.purchase_time, tx.purchase_value, tx.age,
                tx.device_id, tx.ip_address, tx.user_id, tx.source, tx.browser, tx.sex,
                datetime.now().isoformat()
            ))
            raw_id = conn.lastrowid
            
            # Create transactions table if it doesn't exist
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
            
            # Prepare data for transactions table
            row = df.iloc[0]
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
            conn.close()
            
            logging.info(f"Transaction ingested: raw_id={raw_id}, transaction_id={transaction_id}")
            
            return {
                "message": "Transaction processed & stored",
                "raw_id": raw_id,
                "transaction_id": transaction_id,
                "device_id_count": float(row.get('device_id_count', 1)),
                "ip_address_count": float(row.get('ip_address_count', 1))
            }
        except Exception as e:
            conn.rollback()
            conn.close()
            logging.error(f"Error storing transaction: {e}")
            raise HTTPException(status_code=500, detail=f"Error storing transaction: {str(e)}")
    
    except Exception as e:
        logging.error(f"Error ingesting transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingesting transaction: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "ingest_api"}
