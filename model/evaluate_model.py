"""
======================================================
  TAMIL NADU CROP RECOMMENDATION SYSTEM
  Evaluation: Cross-Validation & Confusion Matrix
======================================================
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Data
if not os.path.exists(PROC_PATH):
    print(f"Error: Processed data not found at {PROC_PATH}")
    exit()

df = pd.read_csv(PROC_PATH)
FEATURES = ['n_mid', 'p_mid', 'k_mid', 'temp_mid', 'humidity_mid', 'ph_mid']
X = df[FEATURES]
y = df['CROPS']

# 2. Preprocess
le = LabelEncoder()
y_enc = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# 3. XGBoost Model (with project-defined parameters)
model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# 4. RANDOM SAMPLING & CROSS-VALIDATION
print("--- Stratified 5-Fold Cross Validation ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y_enc, cv=skf)

print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"Standard Deviation: {cv_scores.std()*100:.2f}%")

# Plot CV Scores
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), cv_scores * 100, color='steelblue', alpha=0.8)
plt.axhline(cv_scores.mean() * 100, color='red', linestyle='--', label=f'Mean = {cv_scores.mean()*100:.2f}%')
plt.title('5-Fold Cross-Validation Scores', fontsize=15, fontweight='bold')
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 105)
for i, v in enumerate(cv_scores):
    plt.text(i + 1, v * 100 + 1, f"{v*100:.2f}%", ha='center')
plt.legend()
plt.tight_layout()
cv_plot_path = os.path.join(OUTPUT_DIR, 'model', 'cv_scores.png')
os.makedirs(os.path.dirname(cv_plot_path), exist_ok=True)
plt.savefig(cv_plot_path)
print(f"[OK] CV scores plot saved to: {cv_plot_path}")

# 5. TEST SET EVALUATION
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# 6. CONFUSION MATRIX
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=False, fmt='d', cmap='Greens', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix: Crop Prediction (Acc: {test_acc*100:.2f}%)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path)
print(f"[OK] Confusion matrix saved to: {cm_path}")

# 7. Classification Report (Summary of Precision/Recall)
report = classification_report(y_test, y_pred, target_names=le.classes_)
report_path = os.path.join(OUTPUT_DIR, 'eval_report.txt')
with open(report_path, 'w') as f:
    f.write(f"5-Fold CV Mean: {cv_scores.mean()*100:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n")
    f.write(report)

print(f"[OK] Full report saved to: {report_path}")
