# Fraud Detection System

A machine learning system that detects fraudulent transactions in real time using XGBoost, rule-based anomaly detection, and LLM-powered explanations. Built with FastAPI backend and Streamlit dashboard.

## Overview

This system processes incoming transactions, evaluates them using a trained XGBoost model, applies anomaly detection rules, and provides fraud risk classification. It includes explainability through LLM-generated reasoning and stores all results in a SQLite database for analysis.

## Key Features

* Real-time transaction scoring through FastAPI backend
* XGBoost model with 71.41% recall for fraud detection
* Rule-based anomaly detection for suspicious patterns
* LLM-generated explanations for flagged and denied transactions
* Interactive Streamlit dashboard with analytics
* Alert generation for high-risk transactions
* PDF report export and transaction filtering
* Privacy-preserving SHA-256 hashing for sensitive data

## Project Structure
```
fraud_detection/
├── app.py                          # Streamlit dashboard (frontend)
├── fraud1.py                       # ML training pipeline
├── requirements.txt                # Backend dependencies
├── requirements_streamlit.txt      # Frontend dependencies
│
├── api/                            # FastAPI backend services
│   ├── app.py                      # Main orchestrator
│   ├── ingest_api.py               # Data preprocessing
│   ├── anomaly_api.py              # Rule-based detection
│   ├── ml_scoring_api.py           # XGBoost scoring
│   ├── verifier_api.py             # Classification and LLM explanations
│   ├── alert_api.py                # Alert generation
│   └── data_api.py                 # Data access
│
├── models/                         # Trained models and scalers
│   ├── fraud_detection_xgboost_v1_BEST.pkl
│   ├── standard_scaler_v1.pkl
│   ├── minmax_scaler_v1.pkl
│   └── robust_scaler_v1.pkl
│
└── database/
    └── fraud.db                    # SQLite database
```

## How It Works

### Data Processing Pipeline

1. **Input**: Raw transaction data (user ID, purchase value, timestamps, device, IP, etc.)
2. **Preprocessing**: 
   - Hash sensitive identifiers using SHA-256
   - Calculate time_to_purchase from signup to purchase
   - Count device and IP usage frequency
   - Apply one-hot encoding to categorical features
   - Scale numerical features using pre-fitted scalers
3. **Anomaly Detection**: Apply rule-based checks
4. **ML Scoring**: XGBoost model predicts fraud probability
5. **Classification**: Categorize based on thresholds
6. **Explanation**: Generate human-readable explanation for high-risk cases
7. **Storage**: Save all results to database

### Feature Engineering

* **Numerical features**: purchase_value, age, time_to_purchase, device_id_count, ip_address_count
* **Categorical features**: source (Ads/Direct/SEO), browser (Chrome/Firefox/IE/Opera/Safari), sex (F/M)
* **Scaling**:
  - RobustScaler for purchase_value
  - StandardScaler for age and time_to_purchase
  - MinMaxScaler for device_id_count and ip_address_count

### Privacy Protection

All sensitive identifiers are hashed using SHA-256 before storage:
* User IDs
* Device IDs  
* IP addresses

### Classification Thresholds

The system uses two thresholds to classify transactions:

* **Pass** (score < 0.3): Low risk, transaction approved
* **Flag** (0.3 <= score < 0.6): Medium risk, manual review recommended
* **Deny** (score >= 0.6): High risk, transaction rejected

## Machine Learning Model

### XGBoost Classifier
* **Recall**: 71.41% (catches 71% of actual fraud cases)
* **Training approach**: Class-weighted training with hyperparameter tuning
* **Features**: 15 input features after preprocessing
* **Output**: Fraud probability score (0 to 1)

### Model Files
* `fraud_detection_xgboost_v1_BEST.pkl`: Trained XGBoost model
* `standard_scaler_v1.pkl`: Scaler for age and time_to_purchase
* `minmax_scaler_v1.pkl`: Scaler for device and IP counts
* `robust_scaler_v1.pkl`: Scaler for purchase value

## Anomaly Detection Rules

Rule-based checks applied to every transaction:

* High purchase value (> $100,000): +0.3 anomaly score
* High device usage (> 10 transactions): +0.2 anomaly score
* High IP usage (> 10 transactions): +0.2 anomaly score
* Fast purchase (< 60 seconds after signup): +0.3 anomaly score
* Unusual age (< 18 or > 100): +0.1 anomaly score

## Streamlit Dashboard

### Pages

1. **Submit Transaction**: Form to enter transaction details and get instant fraud prediction
2. **Transaction History**: View all transactions with filtering, search, and color-coded status
3. **Alerts**: High-risk transactions requiring immediate attention
4. **ML Insights**: Score distribution charts and feature importance visualization

### Dashboard Features

* Real-time transaction scoring
* Color-coded status indicators (green/yellow/red)
* Search and filter by transaction ID, date range, score range, status
* Summary statistics (total, passed, flagged, denied counts)
* LLM explanations for suspicious transactions
* PDF report export
* Transaction deletion capability

## Database Schema

### Tables

**transactions**: Preprocessed transaction data with hashed identifiers

**ml_scores**: XGBoost fraud probability scores

**anomaly_detections**: Rule-based anomaly check results

**verifications**: Classification status and LLM explanations

**alerts**: High-risk transaction alerts

## Installation and Setup

### 1. Clone Repository
```bash
git clone https://github.com/RHMhasini/Fraud-Detection.git
cd Fraud-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

### 3. Start Backend Server
```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Backend will run at http://127.0.0.1:8000

### 4. Start Frontend Dashboard
```bash
streamlit run app.py
```

Dashboard will open at http://localhost:8501

## API Endpoints

### Main Orchestrator (port 8000)

* `POST /transaction`: Submit new transaction for processing
* `GET /transactions`: Retrieve all transactions
* `GET /ml_scores`: Retrieve all ML scores
* `GET /verifications`: Retrieve all verifications with explanations
* `GET /alerts`: Retrieve high-risk alerts
* `DELETE /transaction/{id}`: Delete transaction and related records
* `POST /verify/{id}`: Verify specific transaction
* `GET /health`: Check API health status

### Individual Services (monolithic mode)

All services run within the main API when `USE_MONOLITHIC = True`

## Technologies Used

* **Backend**: FastAPI, Uvicorn
* **Frontend**: Streamlit
* **ML**: XGBoost, scikit-learn, pandas, numpy
* **Database**: SQLite
* **LLM**: DeepSeek API for explanations
* **Visualization**: Plotly
* **Reports**: ReportLab

## Model Training

The training pipeline (`fraud1.py`) includes:

1. Data loading and exploratory analysis
2. Feature engineering and preprocessing
3. Train-test split (80/20)
4. SMOTE oversampling for class imbalance
5. Hyperparameter tuning with RandomizedSearchCV
6. Model training with class weights
7. Threshold optimization using F2 score
8. Model evaluation and serialization

To retrain the model, run:
```bash
python fraud1.py
```

## Configuration

### Environment Variables

* `DEEPSEEK_API_KEY`: API key for LLM explanations (default provided in code)

### Thresholds

Modify in `api/verifier_api.py`:
```python
THRESHOLD_PASS = 0.3
THRESHOLD_DENY = 0.6
```

## Performance Metrics

* **Recall**: 71.41% - Successfully identifies 71% of actual fraud cases
* **False Positive Rate**: Acceptable for business requirements
* **Real-time Processing**: < 1 second per transaction

## Use Cases

* E-commerce transaction monitoring
* Payment fraud detection
* Account takeover prevention
* Unusual activity flagging
* Risk-based authentication

## Future Enhancements

* Model retraining pipeline with new data
* Additional feature engineering
* Multi-model ensemble approach
* Real-time monitoring dashboard
* Email/SMS alert notifications
* Integration with payment gateways

## License

This project is part of an academic portfolio demonstrating fraud detection capabilities.

## Contact

GitHub: https://github.com/RHMhasini/Fraud-Detection
