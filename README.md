# Fraud Detection System

A machine learning-powered platform that detects fraudulent transactions using an ensemble of Random Forest and XGBoost models with optimized decision thresholds, supported by anomaly detection and an interactive Streamlit dashboard.

## Project Overview

The Fraud Detection System processes incoming transactions in real time, evaluates them using ML models with optimized thresholds, applies anomaly detection rules, and returns a fraud risk classification. The system also provides explainability via LLM-based reasoning and stores all results in a SQLite database.

## Key Features

* Real-time transaction scoring through a FastAPI backend
* Ensemble model (Random Forest + XGBoost) with threshold optimization
* Precision-Recall curve optimization using F2 score for fraud detection
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
| `fraud1.py` | File | Full ML training pipeline with threshold optimization |
| `requirements.txt` | File | Backend dependencies |
| `requirements_streamlit.txt` | File | Frontend dependencies |
| `api/` | Directory | All FastAPI service modules |
| `api/app.py` | File | Main orchestrator API |
| `api/ingest_api.py` | File | Ingestion + preprocessing |
| `api/anomaly_api.py` | File | Rule-based anomaly detection |
| `api/ml_scoring_api.py` | File | Model loading + scoring with optimal threshold |
| `api/verifier_api.py` | File | Classification + LLM explanation |
| `api/alert_api.py` | File | Alert generation module |
| `models/` | Directory | Trained ML models |
| `models/fraud_best_model.pkl` | File | Trained Random Forest |
| `models/xgb_fraud_model.pkl` | File | Trained XGBoost with optimal threshold |
| `database/fraud.db` | File | SQLite database |

## Data Processing

* **Hashing:** SHA-256 for user_id, device_id, ip_address
* **Feature engineering:** device/ip counts, time_to_purchase, one-hot encoding
* **Scaling:** RobustScaler, StandardScaler, MinMaxScaler
* **Storage:** processed transactions, ML scores, anomalies, verifications, alerts

## Machine Learning Models

### Random Forest
* SMOTE oversampling for class imbalance
* Hyperparameter tuning via ParameterSampler (50 iterations)
* Optimized for high recall on fraud class
* Features: class_weight='balanced_subsample'

### XGBoost (Optimized)
* Class-weighted training using scale_pos_weight
* RandomizedSearchCV tuning (40 iterations)
* **Threshold Optimization:**
  - Precision-Recall curve analysis
  - F2 score optimization (favors recall over precision)
  - Custom threshold selection for fraud detection
  - Saved as dictionary: `{'model': xgb_model, 'threshold': optimal_threshold}`

### Model Usage
```python
import joblib

# Load XGBoost with optimal threshold
model_data = joblib.load("xgb_fraud_model.pkl")
model = model_data['model']
threshold = model_data['threshold']

# Predict
proba = model.predict_proba(X_new)[:, 1]
predictions = (proba >= threshold).astype(int)
```

### Ensemble Output
```
Final_score = (RF_score + XGB_score_with_threshold) / 2
```

## Threshold Optimization

The system uses F2 score-based threshold optimization to balance precision and recall, with emphasis on catching fraud cases:

* Computes precision-recall curve on test data
* Calculates F2 score: `(5 * precision * recall) / (4 * precision + recall)`
* Selects threshold that maximizes F2 score
* Typically results in threshold < 0.5 for better fraud detection

## Web Application (Streamlit)

* Submit transactions and view predictions
* Explore transaction history with filtering
* View alerts and ML insights
* Generate PDF reports
* Visualize model behavior and feature importance

## How to Use

**1. Clone the repository**
```bash
git clone https://github.com/your-repo/fraud-detection-system
cd fraud-detection-system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

**3. Train models**
```bash
python fraud1.py
```

**4. Run backend**
```bash
uvicorn app:app --reload --port 8000
```

**5. Run frontend**
```bash
streamlit run app.py
```

## Dependencies

FastAPI - Streamlit - scikit-learn - XGBoost - imbalanced-learn - pandas - numpy - plotly - reportlab - openai - joblib

## Model Performance

* **Optimized for Recall:** Catches more fraud cases with acceptable false positive rate
* **Threshold Tuning:** Reduces false negatives significantly
* **Ensemble Approach:** Combines strengths of both RF and XGBoost

## Summary

**Features:** Real-time scoring, optimized thresholds, anomaly detection, LLM explainability

**Models:** RF + XGBoost with F2-optimized thresholds

**Dashboard:** Full analytics, history, alerts, PDF export

**Database:** Full pipeline persistence (transactions, scores, alerts)

**Optimization:** Precision-Recall curve analysis for fraud-focused predictions
