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
from dotenv import load_dotenv

load_dotenv()

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

# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

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

def format_transaction_for_explanation(transaction_data: dict) -> str:
    """Format transaction data into human-readable format for LLM"""
    details = []
    
    # Purchase value
    if 'purchase_value' in transaction_data:
        details.append(f"Purchase amount: ${transaction_data['purchase_value']:,.2f}")
    
    # Age
    if 'age' in transaction_data:
        details.append(f"User age: {int(transaction_data['age'])}")
    
    # Time to purchase (if available)
    if 'time_to_purchase' in transaction_data:
        time_to_purchase = transaction_data['time_to_purchase']
        if time_to_purchase < 60:
            details.append(f"Purchase occurred within {time_to_purchase:.0f} seconds of account signup (very fast)")
        elif time_to_purchase < 3600:
            details.append(f"Purchase occurred within {time_to_purchase/60:.1f} minutes of account signup")
        else:
            details.append(f"Purchase occurred {time_to_purchase/3600:.1f} hours after account signup")
    
    # Device count
    if 'device_id_count' in transaction_data:
        device_count = transaction_data.get('device_id_count', 1)
        if device_count > 10:
            details.append(f"This device has been used for {device_count:.0f} transactions (unusually high)")
        elif device_count > 5:
            details.append(f"This device has been used for {device_count:.0f} transactions")
    
    # IP count
    if 'ip_address_count' in transaction_data:
        ip_count = transaction_data.get('ip_address_count', 1)
        if ip_count > 10:
            details.append(f"This IP address has been used for {ip_count:.0f} transactions (unusually high)")
        elif ip_count > 5:
            details.append(f"This IP address has been used for {ip_count:.0f} transactions")
    
    # Source
    if 'source_Ads' in transaction_data:
        if transaction_data.get('source_Ads'):
            details.append("Traffic source: Advertisement")
        elif transaction_data.get('source_Direct'):
            details.append("Traffic source: Direct")
        elif transaction_data.get('source_SEO'):
            details.append("Traffic source: Search Engine")
    
    # Browser
    browsers = ['Chrome', 'FireFox', 'IE', 'Opera', 'Safari']
    for browser in browsers:
        if transaction_data.get(f'browser_{browser}'):
            details.append(f"Browser: {browser}")
            break
    
    # Purchase time (if available)
    if 'purchase_time' in transaction_data:
        try:
            purchase_time = pd.to_datetime(transaction_data['purchase_time'])
            hour = purchase_time.hour
            if hour < 6 or hour > 23:
                details.append(f"Purchase time: {purchase_time.strftime('%Y-%m-%d %H:%M')} (late night/early morning)")
            else:
                details.append(f"Purchase time: {purchase_time.strftime('%Y-%m-%d %H:%M')}")
        except:
            pass
    
    return "\n".join(details) if details else "Transaction details available for review."

def generate_explanation(transaction_data: dict, ensemble_score: float, status: str) -> str:
    """Generate human-friendly LLM explanation for flagged/denied transactions"""
    if not DEEPSEEK_API_KEY:
        return "Transaction requires manual review due to suspicious activity patterns."
    
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
        
        # Format transaction details in human-readable way
        transaction_summary = format_transaction_for_explanation(transaction_data)
        
        # Determine severity level
        severity = "high-risk" if status == "deny" else "moderate-risk"
        action = "denied" if status == "deny" else "flagged for review"
        
        prompt = f"""You are a fraud detection analyst explaining why a transaction was {action}.

Transaction Details:
{transaction_summary}

This is a {severity} transaction that has been {action}.

Provide a concise, human-friendly explanation (1-2 sentences) explaining why this transaction is suspicious. Focus on:
- Unusual transaction amounts
- Suspicious timing patterns
- Multiple transactions from same device/IP
- Account age and activity patterns
- Any other red flags

DO NOT mention:
- Machine learning scores
- Technical terms like "ensemble score" or "ML model"
- Percentages or numerical scores

Use plain language that a human reviewer would understand. Be specific about what makes this transaction suspicious based on the details provided.

Example good explanations:
- "The transaction amount is unusually high for a new account created just minutes ago."
- "This device has been associated with multiple transactions in a short time period, indicating potential account sharing or fraud."
- "The purchase occurred in the early morning hours from an IP address that has been used for numerous previous transactions."

Provide the explanation now:"""
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a fraud detection analyst who explains transaction risks in clear, actionable language for human reviewers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        explanation = response.choices[0].message.content.strip()
        logging.info(f"DeepSeek Explanation generated for Transaction {transaction_data.get('transaction_id')}: {explanation}")
        return explanation
    except Exception as e:
        logging.error(f"DeepSeek API error: {e}")
        return "Transaction requires manual review due to suspicious activity patterns. Please review the transaction details for unusual amounts, timing, or device usage."

def verify_transaction(transaction_id: int):
    """Verify a transaction and generate explanation if needed (standalone function)"""
    try:
        # Get transaction data from API
        transaction_data = get_transaction_from_api(transaction_id)
        
        # Get ML score from API
        score_data = get_ml_score_from_api(transaction_id)
        ensemble_score = score_data.get('ensemble_score')
        
        if ensemble_score is None:
            raise ValueError(f"ML score not found for transaction {transaction_id}")
        
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
    
    except Exception as e:
        logging.error(f"Error verifying transaction: {e}")
        raise

@app.post("/verify/{transaction_id}")
def verify_transaction_endpoint(transaction_id: int):
    """FastAPI endpoint wrapper for verify_transaction"""
    try:
        result = verify_transaction(transaction_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Error verifying transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Error verifying transaction: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "verifier_api"}

