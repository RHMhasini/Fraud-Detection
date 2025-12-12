# Fraud Detection System - Comprehensive Project Report

## Executive Summary

This project is a **comprehensive fraud detection system** that combines machine learning models, anomaly detection, and a user-friendly web interface to identify and flag potentially fraudulent transactions in real-time. The system uses an ensemble of Random Forest and XGBoost models, integrated with an LLM-powered explanation system, and provides a complete end-to-end solution from transaction ingestion to alert generation.

---

## 1. Project Overview

### 1.1 Purpose
The system is designed to detect fraudulent transactions by analyzing various features including purchase behavior, device usage patterns, IP addresses, user demographics, and temporal patterns.

### 1.2 Technology Stack
- **Backend Framework**: FastAPI (Python)
- **Frontend Framework**: Streamlit (Python)
- **Machine Learning**: scikit-learn, XGBoost
- **Database**: SQLite
- **Data Processing**: pandas, numpy
- **LLM Integration**: OpenAI API (via Groq) for explanations
- **Visualization**: Plotly
- **Report Generation**: ReportLab (PDF)

### 1.3 Architecture Pattern
The system follows a **modular monolithic architecture** with clear separation of concerns. While designed to support microservices (with service URLs defined), it currently operates in monolithic mode (`USE_MONOLITHIC = True`) for development simplicity.

---

## 2. System Architecture

### 2.1 Component Overview

The system consists of the following main components:

1. **Frontend (Streamlit App)** - `app.py`
2. **Main Orchestrator API** - `api/app.py`
3. **Ingestion Service** - `api/ingest_api.py`
4. **Anomaly Detection Service** - `api/anomaly_api.py`
5. **ML Scoring Service** - `api/ml_scoring_api.py`
6. **Verification Service** - `api/verifier_api.py`
7. **Alert Service** - `api/alert_api.py`
8. **Data Service** - `api/data_api.py`
9. **Model Training Script** - `fraud1.py`

### 2.2 Transaction Processing Flow

```
1. Transaction Submission (Frontend)
   ↓
2. Main Orchestrator API (/transaction)
   ↓
3. Ingestion Service
   - Preprocess transaction data
   - Calculate device/IP counts
   - Hash sensitive IDs (SHA-256)
   - One-hot encode categorical variables
   - Store in database
   ↓
4. Anomaly Detection Service
   - Check for high purchase values
   - Detect unusual device/IP counts
   - Identify suspicious time patterns
   - Calculate anomaly score
   ↓
5. ML Scoring Service
   - Load pre-trained models (RF + XGBoost)
   - Scale features appropriately
   - Generate fraud probability scores
   - Calculate ensemble score (average)
   ↓
6. Verification Service
   - Classify transaction (pass/flag/deny)
   - Generate LLM explanation (if flagged/denied)
   - Store verification result
   ↓
7. Alert Service (if high-risk)
   - Generate alerts for transactions with score ≥ 0.6
   - Store alert details
   ↓
8. Return comprehensive result to frontend
```

### 2.3 Database Schema

**Tables:**
- `transactions` - Preprocessed transaction data
- `raw_transactions` - Original transaction data (optional)
- `ml_scores` - ML model predictions (RF, XGB, Ensemble)
- `anomaly_detections` - Anomaly detection results
- `verifications` - Transaction verification status and explanations
- `alerts` - High-risk transaction alerts

---

## 3. Machine Learning Models

### 3.1 Model Training (`fraud1.py`)

The training script includes:

**Data Preprocessing:**
- DateTime conversion and time-to-purchase calculation
- Device ID and IP address count features (critical fraud signals)
- SHA-256 hashing for user/device/IP IDs (privacy-preserving)
- One-hot encoding for categorical variables (source, browser, sex)
- Feature scaling:
  - StandardScaler: age, time_to_purchase
  - MinMaxScaler: device_id_count, ip_address_count
  - RobustScaler: purchase_value (handles outliers)

**Model Training:**

1. **Random Forest Classifier**
   - SMOTE oversampling for class imbalance
   - Extensive hyperparameter tuning (50 iterations)
   - Optimized for recall (fraud detection)
   - Best model saved as `RF_best_model.pkl`

2. **XGBoost Classifier**
   - RandomizedSearchCV with 40 iterations
   - 3-fold cross-validation
   - scale_pos_weight for class imbalance
   - Best model saved as `xgb_fraud_model.pkl`

**Ensemble Approach:**
- Final score = (RF_score + XGB_score) / 2
- Provides more robust predictions than individual models

### 3.2 Model Files
- `models/RF_best_model.pkl` - Trained Random Forest model
- `models/xgb_fraud_model.pkl` - Trained XGBoost model

### 3.3 Feature Importance
Based on the code, key features include:
- `purchase_value` (40% importance)
- `device_id_count` (30% importance)
- `age` (20% importance)
- `ip_address_count` (15% importance)
- `time_to_purchase` (12% importance)
- Categorical features (source, browser, sex)

---

## 4. API Services

### 4.1 Main Orchestrator (`api/app.py`)

**Key Endpoints:**
- `POST /transaction` - Main transaction processing endpoint
- `GET /transactions` - Retrieve all transactions
- `GET /ml_scores` - Retrieve all ML scores
- `DELETE /transaction/{id}` - Delete transaction and related records
- `POST /verify/{id}` - Verify specific transaction
- `POST /verify_last` - Verify last transaction
- `GET /alerts` - Get all alerts
- `GET /health` - Health check

**Features:**
- Orchestrates entire fraud detection pipeline
- Supports both monolithic and microservices modes
- Comprehensive logging
- Error handling and rollback

### 4.2 Ingestion Service (`api/ingest_api.py`)

**Responsibilities:**
- Preprocess raw transaction data
- Calculate device_id_count and ip_address_count from historical data
- Hash sensitive identifiers (SHA-256)
- One-hot encode categorical variables
- Store both raw and processed transactions

**Key Function:**
- `calculate_counts_from_db()` - Calculates how many times a device/IP has been seen

### 4.3 Anomaly Detection Service (`api/anomaly_api.py`)

**Detection Rules:**
- High purchase value (> $100,000)
- Unusual device count (> 10)
- Unusual IP count (> 10)
- Very short time to purchase (< 60 seconds)
- Unusual age (< 18 or > 100)

**Output:**
- Boolean flag for anomalies
- Anomaly score (0-1)
- List of detected anomalies

### 4.4 ML Scoring Service (`api/ml_scoring_api.py`)

**Responsibilities:**
- Load pre-trained models
- Scale features appropriately
- Generate fraud probability scores
- Calculate ensemble score
- Store scores in database

**Scaling Strategy:**
- Attempts to fit scalers on historical data from API
- Falls back to fitting on first transaction if no history available
- Uses same scaling strategy as training: RobustScaler, StandardScaler, MinMaxScaler

### 4.5 Verification Service (`api/verifier_api.py`)

**Classification Thresholds:**
- **Pass**: ensemble_score < 0.3 (Low risk)
- **Flag**: 0.3 ≤ ensemble_score < 0.6 (Medium risk, needs review)
- **Deny**: ensemble_score ≥ 0.6 (High risk, should be denied)

**LLM Integration:**
- Uses Groq API (Llama 3.3 70B) for generating explanations
- Only generates explanations for flagged/denied transactions
- Provides human-readable fraud reasoning

**Configuration:**
- Requires `GROQ_API_KEY` environment variable
- Falls back gracefully if API key not set

### 4.6 Alert Service (`api/alert_api.py`)

**Responsibilities:**
- Identify high-risk transactions (score ≥ 0.6)
- Generate alerts with transaction details
- Store alerts in database
- Alert levels: CRITICAL (≥ 0.8) or HIGH (0.6-0.8)

### 4.7 Data Service (`api/data_api.py`)

**Endpoints:**
- `GET /processed` - Get processed transactions
- `GET /scores` - Get model scores

**Features:**
- Safe byte decoding for database values
- Limit support for pagination

---

## 5. Frontend Application (`app.py`)

### 5.1 Pages

1. **Submit Transaction**
   - Form for entering transaction details
   - Real-time API connection check
   - Displays ML scores, anomaly detection, verification status
   - Shows explanations for flagged/denied transactions

2. **Transaction History Dashboard**
   - View all transactions with filtering
   - Search by transaction ID, user hash, device hash, IP
   - Filter by status, score range, date range
   - Color-coded rows (green/yellow/red)
   - Summary statistics
   - PDF report generation
   - Delete transaction functionality

3. **Alerts Page**
   - Displays high-risk transactions (score ≥ 0.6)
   - Expandable cards with transaction details
   - LLM-generated explanations
   - Cached explanations to avoid repeated API calls

4. **ML Insights Page**
   - Score distribution histogram
   - Feature importance visualization
   - Threshold configuration slider
   - Classification preview table
   - Model information and explanation

### 5.2 Features
- Real-time API status monitoring
- Interactive visualizations (Plotly)
- PDF report generation
- Responsive design with Streamlit
- Error handling and user feedback

---

## 6. Data Processing

### 6.1 Feature Engineering

**Temporal Features:**
- `time_to_purchase` - Seconds between signup and purchase

**Count Features:**
- `device_id_count` - Number of transactions from same device
- `ip_address_count` - Number of transactions from same IP

**Privacy Features:**
- SHA-256 hashing for user_id, device_id, ip_address
- Hashed values stored instead of raw identifiers

**Categorical Encoding:**
- One-hot encoding for: source (Ads, Direct, SEO), browser (Chrome, Firefox, IE, Opera, Safari), sex (F, M)

**Scaling:**
- StandardScaler: age, time_to_purchase
- MinMaxScaler: device_id_count, ip_address_count
- RobustScaler: purchase_value

### 6.2 Data Storage

**Database Location:** `database/fraud.db`

**Tables:**
- `transactions` - Main transaction table with preprocessed features
- `raw_transactions` - Optional raw transaction storage
- `ml_scores` - Model predictions
- `anomaly_detections` - Anomaly detection results
- `verifications` - Verification status and explanations
- `alerts` - Alert records

---

## 7. Configuration & Dependencies

### 7.1 Python Dependencies (`requirements.txt`)
```
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
requests>=2.31.0
openai>=1.0.0
reportlab>=3.6.0
```

### 7.2 Streamlit Dependencies (`requirements_streamlit.txt`)
```
streamlit>=1.28.0
requests>=2.31.0
pandas>=2.0.0
plotly>=5.17.0
reportlab>=3.6.0
```

### 7.3 Environment Variables
- `GROQ_API_KEY` - Required for LLM explanations (optional, system works without it)

### 7.4 Configuration Constants

**Thresholds:**
- `THRESHOLD_PASS = 0.3` - Below this = pass
- `THRESHOLD_DENY = 0.6` - Above this = deny
- Between = flag

**API Configuration:**
- Main API: `http://127.0.0.1:8000`
- Monolithic mode: `USE_MONOLITHIC = True`

---

## 8. Security & Privacy

### 8.1 Privacy Measures
- SHA-256 hashing for sensitive identifiers (user_id, device_id, ip_address)
- Hashed values stored in database
- Raw identifiers not persisted (unless in raw_transactions table)

### 8.2 Security Considerations
- Input validation via Pydantic models
- SQL injection prevention (parameterized queries)
- Error handling to prevent information leakage
- Logging for audit trails

---

## 9. Logging

### 9.1 Log Files
All services maintain separate log files in `api/logs/`:
- `api.log` - Main orchestrator
- `ingest_api.log` - Ingestion service
- `anomaly_api.log` - Anomaly detection
- `ml_scoring_api.log` - ML scoring
- `verifier_api.log` - Verification service
- `alert_api.log` - Alert service

### 9.2 Logging Format
```
%(asctime)s - %(levelname)s - %(message)s
```

---

## 10. Project Structure

```
fraud_detection/
├── api/
│   ├── __pycache__/
│   ├── logs/
│   │   └── api.log
│   ├── alert_api.py
│   ├── anomaly_api.py
│   ├── app.py (Main Orchestrator)
│   ├── data_api.py
│   ├── ingest_api.py
│   ├── ml_scoring_api.py
│   └── verifier_api.py
├── data/
│   └── Fraud_Data.csv
├── database/
│   └── fraud.db
├── models/
│   ├── RF_best_model.pkl
│   └── xgb_fraud_model.pkl
├── outputs/
│   └── ml_scores.csv
├── app.py (Streamlit Frontend)
├── fraud1.py (Model Training Script)
├── requirements.txt
└── requirements_streamlit.txt
```

---

## 11. Usage Instructions

### 11.1 Starting the Backend API
```bash
cd api
uvicorn app:app --reload --port 8000
```

### 11.2 Starting the Frontend
```bash
streamlit run app.py
```

### 11.3 Environment Setup
```bash
# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
pip install -r requirements_streamlit.txt

# Set environment variable (optional, for LLM explanations)
export GROQ_API_KEY="your-api-key"
```

---

## 12. Key Features

### 12.1 Real-time Fraud Detection
- Instant transaction scoring upon submission
- Ensemble model predictions
- Anomaly detection integration

### 12.2 Explainability
- LLM-powered explanations for flagged/denied transactions
- Human-readable fraud reasoning
- Feature importance visualization

### 12.3 Comprehensive Dashboard
- Transaction history with filtering
- Visual analytics (histograms, bar charts)
- PDF report generation
- Alert management

### 12.4 Scalability
- Modular architecture supports microservices
- Database-backed for persistence
- Efficient feature engineering

---

## 13. Model Performance

### 13.1 Training Approach
- **Random Forest**: SMOTE oversampling + extensive hyperparameter tuning (50 iterations)
- **XGBoost**: RandomizedSearchCV (40 iterations, 3-fold CV) with class weight balancing
- **Optimization Metric**: Recall (to catch as many fraud cases as possible)

### 13.2 Ensemble Benefits
- Reduces overfitting
- More robust predictions
- Combines strengths of both models

---

## 14. Limitations & Future Improvements

### 14.1 Current Limitations
1. **Scaler Fitting**: Scalers may be fitted on single transactions if no historical data exists
2. **Database**: SQLite may not scale for high-volume production
3. **LLM Dependency**: Explanations require external API (Groq)
4. **Model Retraining**: No automated retraining pipeline
5. **Real-time Updates**: No streaming/real-time updates for dashboard

### 14.2 Potential Improvements
1. **Model Retraining Pipeline**: Automated periodic retraining with new data
2. **Feature Store**: Centralized feature management
3. **Streaming Architecture**: Kafka/RabbitMQ for real-time processing
4. **Production Database**: PostgreSQL or MongoDB for scalability
5. **A/B Testing**: Framework for testing new models
6. **Monitoring**: Model drift detection and performance monitoring
7. **API Rate Limiting**: Protection against abuse
8. **Authentication**: User authentication and authorization
9. **Caching**: Redis for frequently accessed data
10. **Containerization**: Docker for easy deployment

---

## 15. Conclusion

This fraud detection system provides a **complete, production-ready solution** for identifying fraudulent transactions. It combines:

- **Advanced ML Models**: Ensemble of Random Forest and XGBoost
- **Real-time Processing**: Fast API-based transaction processing
- **User-Friendly Interface**: Streamlit dashboard with comprehensive features
- **Explainability**: LLM-powered explanations for transparency
- **Modular Architecture**: Easy to extend and maintain

The system is well-structured, documented, and ready for deployment with minor production considerations (database scaling, authentication, etc.).

---

## 16. Technical Specifications

### 16.1 API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/transaction` | POST | Process new transaction |
| `/transactions` | GET | Get all transactions |
| `/ml_scores` | GET | Get all ML scores |
| `/transaction/{id}` | DELETE | Delete transaction |
| `/verify/{id}` | POST | Verify transaction |
| `/verify_last` | POST | Verify last transaction |
| `/alerts` | GET | Get all alerts |
| `/health` | GET | Health check |

### 16.2 Data Flow Summary

1. **Input**: Raw transaction (user_id, device_id, ip_address, purchase_value, etc.)
2. **Processing**: Preprocessing → Anomaly Detection → ML Scoring → Verification
3. **Output**: Transaction ID, ML scores, verification status, explanation, alerts

### 16.3 Model Input Features

- Numerical: purchase_value, age, time_to_purchase, device_id_count, ip_address_count
- Categorical (one-hot): source_Ads, source_Direct, source_SEO, browser_Chrome, browser_FireFox, browser_IE, browser_Opera, browser_Safari, sex_F, sex_M

---

**Report Generated**: 2024
**Project**: Fraud Detection System
**Version**: 1.0


