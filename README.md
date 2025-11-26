# Fraud Detection System

A fully functional fraud detection system with microservices architecture that processes transactions through a complete pipeline: ingestion → anomaly detection → ML scoring → verification → alert generation.

## Architecture

The system follows a microservices architecture with the following services:

1. **Transaction Ingestion API** (`api/ingest_api.py`)
   - Accepts raw transactions
   - Preprocesses data (hashing, one-hot encoding, count calculations)
   - Stores transactions in database

2. **Anomaly Detector API** (`api/anomaly_api.py`)
   - Detects anomalies in transactions
   - Checks for high purchase values, unusual device/IP counts, suspicious timing patterns
   - Returns anomaly scores and detected issues

3. **ML Scoring API** (`api/ml_scoring_api.py`)
   - Calculates fraud scores using Random Forest and XGBoost models
   - Generates ensemble scores
   - Uses scalers fitted from historical API data (not local files)

4. **Transaction Verifier API** (`api/verifier_api.py`)
   - Verifies transactions based on ML scores
   - Classifies as pass/flag/deny
   - Generates LLM explanations for flagged/denied transactions (requires GROQ_API_KEY)

5. **Alert Agent API** (`api/alert_api.py`)
   - Generates alerts for high-risk transactions
   - Retrieves verification details
   - Stores alerts in database

6. **Main Orchestrator API** (`api/app.py`)
   - Coordinates the complete flow
   - Endpoint: `POST /transaction` - processes transaction through entire pipeline
   - Additional endpoints for data retrieval

## Flow

```
Transaction → Ingestion → Anomaly Detection → ML Scoring → Verification → Alert Generation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

2. Set environment variable for LLM (optional):
```bash
export GROQ_API_KEY=your_api_key_here
```

3. Ensure model files exist:
   - `models/RF_best_model.pkl`
   - `models/xgb_fraud_model.pkl`

## Running the System

### Start the API:
```bash
cd api
uvicorn app:app --reload --port 8000
```

### Start the Streamlit UI:
```bash
streamlit run app.py
```

## API Endpoints

### Main Endpoint
- `POST /transaction` - Submit a raw transaction and process through entire pipeline
  - Input: Raw transaction (user_id, device_id, ip_address, etc.)
  - Output: Complete processing result including anomaly detection, ML scores, verification, and alerts

### Data Retrieval Endpoints
- `GET /transactions` - Get all transactions
- `GET /ml_scores` - Get all ML scores
- `POST /verify/{transaction_id}` - Verify a specific transaction
- `POST /verify_last` - Verify the last transaction
- `GET /alerts` - Get all alerts

## Key Features

**Fully API-based**: All data comes from APIs, no local file reads (except model files)

**Microservices Architecture**: Separate services for each stage of processing

**Complete Flow**: Transaction → Anomaly → ML Score → Verification → Alert

**LLM Integration**: Generates explanations for flagged transactions (if API key available)

**Database Storage**: All data stored in SQLite database

**Streamlit UI**: User-friendly interface for submitting transactions and viewing results

## Notes

- The system uses a monolithic mode by default (`USE_MONOLITHIC = True` in `api/app.py`) where all services run together
- To use separate microservices, set `USE_MONOLITHIC = False` and run each service on different ports
- LLM functionality requires `GROQ_API_KEY` environment variable but will work without it (with limited explanations)
- Scalers are initialized from historical API data, not from local CSV files

