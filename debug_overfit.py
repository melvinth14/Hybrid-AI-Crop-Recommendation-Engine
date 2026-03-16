
import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Paths
BASE_DIR = r"c:\Users\ASUS\spem\crop_recommendation_tn"
PROC_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Load Data
df = pd.read_csv(PROC_PATH)
FEATURES = ['n_mid', 'p_mid', 'k_mid', 'temp_mid', 'humidity_mid', 'ph_mid']
TARGET   = 'CROPS'
X = df[FEATURES]
y = df[TARGET]

# Encode & Scale
le = LabelEncoder()
y_enc = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Load existing model
model_path = os.path.join(MODEL_DIR, 'crop_model.pkl')
xgb = joblib.load(model_path)

# Predict
train_preds = xgb.predict(X_train)
test_preds = xgb.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print(f"TRAIN_ACC: {train_acc*100:.2f}%")
print(f"TEST_ACC: {test_acc*100:.2f}%")
