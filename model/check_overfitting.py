
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')

# --- 1. Load and Prepare Data ---
if not os.path.exists(PROC_PATH):
    print(f"[ERROR] Processed data not found at {PROC_PATH}")
    exit()

df = pd.read_csv(PROC_PATH)
FEATURES = ['n_mid', 'p_mid', 'k_mid', 'temp_mid', 'humidity_mid', 'ph_mid']
TARGET   = 'CROPS'

X = df[FEATURES]
y = df[TARGET]

le = LabelEncoder()
y_enc = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# --- 2. Train XGBoost (Using same params as train_model.py) ---
print(f"--- Model Balance Check ---")
xgb = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb.fit(X_train, y_train)

# --- 3. Compare Accuracy ---
train_preds = xgb.predict(X_train)
test_preds = xgb.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)
gap = (train_acc - test_acc) * 100

print(f"\n1. ACCURACY METRICS:")
print(f"|-- Training Accuracy: {train_acc*100:.2f}%")
print(f"|-- Testing Accuracy:  {test_acc*100:.2f}%")
print(f"|-- Accuracy Gap:       {gap:.2f}%")

# --- 4. Interpretation ---
print(f"\n2. INTERPRETATION:")
if gap < 2:
    print("STATUS: WELL BALANCED - The model generalizes perfectly.")
elif gap < 5:
    print("STATUS: SLIGHT BIAS - Normal variance. The model is good for deployment.")
elif gap < 10:
    print("STATUS: MODERATE OVERFIT - Consider reducing 'max_depth' or increasing 'subsample'.")
else:
    print("STATUS: HIGH OVERFIT - The model is memorizing the training data. Needs urgent tuning.")

# --- 5. Full Report (Optional but useful for debugging) ---
print(f"\n3. DETAILED REPORT (Test Set):")
print(classification_report(y_test, test_preds, target_names=le.classes_))
