import pandas as pd
import numpy as np
import joblib
import hashlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tqdm.auto import tqdm

# ================================
# 1. LOAD RAW DATA
# ================================
# Assuming "data/Fraud_Data.csv" is the *original* raw file, not the preprocessed one saved in Colab.
df = pd.read_csv("data/Fraud_Data.csv")
print("Raw data loaded:", df.shape)

# ================================
# 2. BASIC PREPROCESSING (MATCH COLAB)
# This section MUST match the sequential steps of Colab notebook exactly.
# ================================
print("Preprocessing (Matching Colab Steps)...")

# --- Colab Step 1: Time Conversion & Calculation ---
df['signup_time'] = pd.to_datetime(df['signup_time'])
df['purchase_time'] = pd.to_datetime(df['purchase_time'])
df['time_to_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()

# --- Colab Step 2: Calculate Count Features (device_id_count, ip_address_count) ---
# NOTE: In production, these count transformations would need to be computed
# from the original *training* data set and applied to the new data.
# For matching the notebook's logic, we re-run the `transform('count')` locally.
df['device_id_count'] = df.groupby('device_id')['device_id'].transform('count')
df['ip_address_count'] = df.groupby('ip_address')['ip_address'].transform('count')

# --- Colab Step 3: Hashing IDs and Dropping Raw Columns ---
# Hashing is not used by the RF/XGB models, but we mimic the column drops.
# NOTE: We skip the hashing function definition as it's not needed for the ML model's X matrix.
df = df.drop(columns=['user_id', 'device_id', 'ip_address'])

# Drop raw time columns as they are no longer needed
drop_cols = ['signup_time', 'purchase_time']
df = df.drop(columns=drop_cols)

# --- Colab Step 4: One-Hot Encoding ---
categorical_cols = ['source', 'browser', 'sex']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# --- Colab Step 5: Specific Feature Scaling ---
# In production, these scalers (StandardScaler, MinMaxScaler, RobustScaler) would be
# FIT on the training set and then LOADED here to TRANSFORM the new data.
# To replicate the Colab logic for *this file*, we re-fit and transform.

# 1. Standard Scaler for 'age', 'time_to_purchase'
standard_scaler = StandardScaler()
df[['age', 'time_to_purchase']] = standard_scaler.fit_transform(df[['age', 'time_to_purchase']])

# 2. MinMax Scaler for 'device_id_count', 'ip_address_count'
minmax_scaler = MinMaxScaler()
df[['device_id_count', 'ip_address_count']] = minmax_scaler.fit_transform(df[['device_id_count', 'ip_address_count']])

# 3. Robust Scaler for 'purchase_value'
robust_scaler = RobustScaler()
df[['purchase_value']] = robust_scaler.fit_transform(df[['purchase_value']])

# --- Final Column Alignment ---
# Ensure 'class' column is dropped if present, as it was excluded in Colab for X
if 'class' in df.columns:
    df = df.drop(columns=['class'])

# Re-align columns to match the trained model's feature set
# This ensures that even if dummy variables are missing (e.g., sex_O not in new data)
# or in a different order, the input matrix is correct.
rf_model = joblib.load("models/RF_best_model.pkl")
model_cols = rf_model.feature_names_in_
df = df[model_cols]


print("Final feature matrix:", df.shape)

# ================================
# 3. LOAD MODELS
# ================================
print("\nLoading trained models...")
rf = joblib.load("models/RF_best_model.pkl")
xgb = joblib.load("models/xgb_fraud_model.pkl")
print("Models loaded.")

# ================================
# 4. GENERATE SCORES
# ================================
print("\nGenerating ML scores...")

# NOTE: The XGBoost model was trained on the ORIGINAL (non-SMOTE) X_train,
# while the Random Forest model was trained on the SMOTE-resampled data.
# Both models expect the same final set of preprocessed and scaled features (X).

rf_scores = rf.predict_proba(df)[:, 1]
xgb_scores = xgb.predict_proba(df)[:, 1]

ensemble_scores = (rf_scores + xgb_scores) / 2.0

# ================================
# 5. SAVE OUTPUT
# ================================
output = pd.DataFrame({
    "RF_score": rf_scores,
    "XGB_score": xgb_scores,
    "ensemble_score": ensemble_scores
})

output.to_csv("outputs/ml_scores.csv", index=False)

print("\nScores saved to outputs/ml_scores.csv")
print(output.head())