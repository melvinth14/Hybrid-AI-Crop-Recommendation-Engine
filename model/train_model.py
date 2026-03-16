"""
======================================================
  TAMIL NADU CROP RECOMMENDATION SYSTEM
  Step 3: Model Training (XGBoost Upgrade)
======================================================
"""
import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv(PROC_PATH)

# ── 2. Feature Selection (Biological Focus) ─────────────────────────────────────
FEATURES = ['n_mid', 'p_mid', 'k_mid', 'temp_mid', 'humidity_mid', 'ph_mid']
TARGET   = 'CROPS'

X = df[FEATURES]
y = df[TARGET]

# ── 3. Encode & Scale ──────────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# ── 4. Train XGBoost Model ────────────────────────────────────────────────────
print(f"[INFO] Training XGBoost model with features: {FEATURES}")
# We use moderate depth and learning rate to prevent overfitting while boosting accuracy
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

# ── 5. Evaluation ──────────────────────────────────────────────────────────────
y_pred = xgb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n[OK] XGBoost Model Trained!")
print(f"Test Accuracy: {acc*100:.2f}%")

# ── 6. Save Model Artefacts ──────────────────────────────────────────────────
joblib.dump(xgb,      os.path.join(MODEL_DIR, 'crop_model.pkl'))
joblib.dump(le,       os.path.join(MODEL_DIR, 'label_encoder.pkl'))
joblib.dump(scaler,   os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, 'features.pkl'))

print(f"\n[INFO] Model saved as: {os.path.join(MODEL_DIR, 'crop_model.pkl')}")
