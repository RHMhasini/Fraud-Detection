from fastapi import FastAPI, HTTPException
import sqlite3
import pandas as pd
import requests
import json
import os
from datetime import datetime
from pathlib import Path
import logging
from openai import OpenAI

app = FastAPI(title="Transaction Verifier API")

# Logging Setup
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "verifier_api.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "fraud.db"

# Thresholds
THRESHOLD_PASS = 0.3
THRESHOLD_DENY = 0.6

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_db():
    return sqlite3.connect(DB_PATH)

def classify_transaction(ensemble_score: float) -> str:
    """Classify transaction based on ensemble score"""
    if ensemble_score < THRESHOLD_PASS:
        return "pass"
    elif ensemble_score < THRESHOLD_DENY:
        return "flag"
    else:
        return "deny"

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

def get_ml_score_from_api(transaction_id: int) -> dict:
    """Get ML score from main API"""
    try:
        response = requests.get(f"http://127.0.0.1:8000/ml_scores", timeout=2)
        if response.status_code == 200:
            scores = response.json()
            for score in scores:
                if score.get('transaction_id') == transaction_id:
                    return score
        raise HTTPException(status_code=404, detail=f"ML score for transaction {transaction_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logging.warning(f"Could not fetch ML score from API: {e}")
        # Fallback to database
        conn = get_db()
        try:
            df = pd.read_sql(
                "SELECT * FROM ml_scores WHERE transaction_id = ?",
                conn, params=[transaction_id]
            )
            if df.empty:
                raise HTTPException(status_code=404, detail=f"ML score for transaction {transaction_id} not found")
            return df.iloc[0].to_dict()
        finally:
            conn.close()

def generate_explanation(transaction_data: dict, ensemble_score: float, status: str) -> str:
    """Generate LLM explanation for flagged/denied transactions"""
    if not GROQ_API_KEY:
        return f"LLM explanation unavailable: GROQ_API_KEY not set. Transaction {status} based on score {ensemble_score:.3f}."
    
    try:
        client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        prompt = f"""
        Analyze the following transaction data and explain why it was classified as '{status}' (ensemble fraud score: {ensemble_score:.3f}).
        Provide a human-readable explanation based on the details. Focus on suspicious patterns like high device counts, IP counts, purchase value, or other anomalies.
        Transaction Data: {json.dumps(transaction_data, indent=2, default=str)}
        Keep the explanation concise (1-2 sentences) and natural.
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant explaining fraud detection decisions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        explanation = response.choices[0].message.content.strip()
        logging.info(f"LLM Explanation generated for Transaction {transaction_data.get('transaction_id')}: {explanation}")
        return explanation
    except Exception as e:
        logging.error(f"LLM API error: {e}")
        return f"Unable to generate explanation due to API error: {str(e)}. Transaction {status} based on score {ensemble_score:.3f}."

@app.post("/verify/{transaction_id}")
def verify_transaction(transaction_id: int):
    """Verify a transaction and generate explanation if needed"""
    try:
        # Get transaction data from API
        transaction_data = get_transaction_from_api(transaction_id)
        
        # Get ML score from API
        score_data = get_ml_score_from_api(transaction_id)
        ensemble_score = score_data.get('ensemble_score')
        
        if ensemble_score is None:
            raise HTTPException(status_code=404, detail=f"ML score not found for transaction {transaction_id}")
        
        # Classify transaction
        status = classify_transaction(ensemble_score)
        
        # Generate explanation only for flagged/denied transactions
        explanation = None
        if status in ["flag", "deny"]:
            explanation = generate_explanation(transaction_data, ensemble_score, status)
        
        # Store verification result
        conn = get_db()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verifications (
                    verification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id INTEGER,
                    ensemble_score REAL,
                    status TEXT,
                    explanation TEXT,
                    created_at TEXT
                )
            """)
            
            conn.execute("""
                INSERT INTO verifications 
                (transaction_id, ensemble_score, status, explanation, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                transaction_id, ensemble_score, status, explanation, datetime.now().isoformat()
            ))
            conn.commit()
        finally:
            conn.close()
        
        logging.info(f"Transaction {transaction_id} verified: status={status}, score={ensemble_score:.3f}")
        
        return {
            "transaction_id": transaction_id,
            "ensemble_score": ensemble_score,
            "status": status,
            "explanation": explanation
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error verifying transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Error verifying transaction: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "verifier_api"}

