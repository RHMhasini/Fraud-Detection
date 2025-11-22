from fastapi import FastAPI
import pandas as pd
import sqlite3

app = FastAPI(title="Processed + Score Data API")

def db():
    return sqlite3.connect("database/fraud.db")

def safe_decode(value):
    """Safely decode bytes to string, handling invalid UTF-8."""
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return value.decode('latin-1', errors='ignore')  # Fallback; 'ignore' skips invalid bytes
    return value

@app.get("/processed")
def get_processed(limit: int = 50):
    conn = db()
    df = pd.read_sql("SELECT * FROM processed_transactions LIMIT ?", conn, params=[limit])
    conn.close()
    # Apply safe decoding to all cells in the DataFrame
    df = df.applymap(safe_decode)
    return df.to_dict(orient="records")

@app.get("/scores")
def get_scores(limit: int = 50):
    conn = db()
    df = pd.read_sql("SELECT * FROM model_scores LIMIT ?", conn, params=[limit])
    conn.close()
    # Apply safe decoding to all cells in the DataFrame
    df = df.applymap(safe_decode)
    return df.to_dict(orient="records")
