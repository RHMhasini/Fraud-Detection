# Fraud Detection System

A machine learning–powered platform that detects fraudulent transactions using an ensemble of Random Forest and XGBoost models, supported by anomaly detection and an interactive Streamlit dashboard.

## Project Overview

The Fraud Detection System processes incoming transactions in real time, evaluates them using ML models, applies anomaly detection rules, and returns a fraud risk classification. The system also provides explainability via LLM-based reasoning and stores all results in a SQLite database.

## Key Features

* Real-time transaction scoring through a FastAPI backend
* Ensemble model (Random Forest + XGBoost) for robust fraud prediction
* Anomaly detection (device/IP frequency, purchase thresholds, timing anomalies, etc.)
* LLM-generated explanations for flagged or denied transactions
* Streamlit dashboard for submitting transactions and viewing analytics
* Alert generation for high-risk transactions
* PDF reporting, interactive charts, transaction filtering
* Privacy-preserving hashing (SHA-256) for sensitive identifiers

## Project Structure

| Directory / File | Type | Description |
|---|---|---|
| `app.py` | File | Main Streamlit dashboard |
| `fraud1.py` | File | Full ML training pipeline |
| `requirements.txt` | File | Backend dependencies |
| `requirements_streamlit.txt` | File | Frontend dependencies |
| `api/` | Directory | All FastAPI service modules |
| `api/app.py` | File | Main orchestrator API |
| `api/ingest_api.py` | File | Ingestion + preprocessing |
| `api/anomaly_api.py` | File | Rule-based anomaly detection |
| `api/ml_scoring_api.py` | File | Model loading + scoring |
| `api/verifier_api.py` | File | Classification + LLM explan. |
| `api/alert_api.py` | File | Alert generation module |
| `models/` | Directory | Trained ML models |
| `models/RF_best_model.pkl` | File | Trained Random Forest |
| `models/xgb_fraud_model.pkl` | File | Trained XGBoost |
| `database/fraud.db` | File | SQLite database |

## Data Processing

* **Hashing:** SHA-256 for user_id, device_id, ip_address
* **Feature engineering:** device/ip counts, time_to_purchase, one-hot encoding
* **Scaling:** RobustScaler, StandardScaler, MinMaxScaler
* **Storage:** processed transactions, ML scores, anomalies, verifications, alerts

## Machine Learning Models

### Random Forest (Primary)

* SMOTE oversampling
* Hyperparameter tuned
* High recall for fraud detection

### XGBoost (Primary)

* Class-weighted training
* RandomizedSearchCV tuning

### Ensemble Output

```
Final_score = (RF_score + XGB_score) / 2
```

## Web Application (Streamlit)

* Submit transactions and view predictions
* Explore transaction history with filtering
* View alerts and ML insights
* Generate PDF reports
* Visualize model behavior and feature importance

## How to Use

**1. Clone the repository**

```
git clone https://github.com/your-repo/fraud-detection-system
cd fraud-detection-system
```

**2. Install dependencies**

```
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

**3. Run backend**

```
uvicorn app:app --reload --port 8000
```

**4. Run frontend**

```
streamlit run app.py
```

## Dependencies

FastAPI - Streamlit - scikit-learn - XGBoost pandas - numpy - plotly - reportlab - openai

## Summary

**Features:** Real-time scoring, anomaly detection, LLM explainability

**Models:** RF + XGBoost (ensemble)

**Dashboard:** Full analytics, history, alerts, PDF export

**Database:** Full pipeline persistence (transactions, scores, alerts)
