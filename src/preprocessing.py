"""
======================================================
  TAMIL NADU CROP RECOMMENDATION SYSTEM
  Step 2: Data Cleaning & Preprocessing
======================================================
Run:  python src/preprocessing.py
Output: data/processed_data.csv
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'crop_data.csv')
OUT_PATH  = os.path.join(BASE_DIR, 'data', 'processed_data.csv')

print("=" * 60)
print("  TAMIL NADU CROP RECOMMENDATION — PREPROCESSING")
print("=" * 60)

# ── 1. Load & Shuffle ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\n[OK] Loaded & Shuffled: {df.shape[0]:,} rows x {df.shape[1]} columns")
original_shape = df.shape

# ── 2. Standardise column names ────────────────────────────────────────────────
df.columns = [c.strip().upper() for c in df.columns]

# ── 3. Standardise text values ─────────────────────────────────────────────────
text_cols = ['CROPS', 'TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED', 'WATER_SOURCE']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# ── 4. Pre-Validation Cleaning ────────────────────────────────────────────────
# Drop rows with < 3 non-null values (basically empty)
df.dropna(thresh=3, inplace=True)

# ── 5. Missing Values ──────────────────────────────────────────────────────────
nulls_before = df.isnull().sum().sum()
print(f"\n[STEP] CLEANING missing values (count: {nulls_before})")

# Fill numeric NaN with median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

# Fill categorical NaN with mode
for col in text_cols:
    if col in df.columns and df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

# ── 6. Duplicates ──────────────────────────────────────────────────────────────
dupes_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"   Duplicates removed:    {dupes_before:,} (kept {len(df):,} rows)")

# ── 7. Validate numeric ranges (Hard Clipping) ────────────────────────────────
print("\n[STEP] VALIDATING RANGES")

def clip_column(df, col, min_v, max_v):
    if col in df.columns:
        df[col] = df[col].clip(min_v, max_v)

clip_column(df, 'SOIL_PH', 0, 14)
clip_column(df, 'SOIL_PH_HIGH', 0, 14)
clip_column(df, 'RELATIVE_HUMIDITY', 0, 100)
clip_column(df, 'RELATIVE_HUMIDITY_MAX', 0, 100)
clip_column(df, 'TEMP', -10, 60)
clip_column(df, 'MAX_TEMP', -10, 60)

# ── 8. Feature Engineering: Midpoints ─────────────────────────────────────────
print("\n[STEP] FEATURE ENGINEERING")
pairs = [
    ('SOIL_PH',         'SOIL_PH_HIGH',          'ph_mid'),
    ('CROPDURATION',    'CROPDURATION_MAX',        'duration_mid'),
    ('TEMP',            'MAX_TEMP',               'temp_mid'),
    ('WATERREQUIRED',   'WATERREQUIRED_MAX',       'water_mid'),
    ('RELATIVE_HUMIDITY','RELATIVE_HUMIDITY_MAX',  'humidity_mid'),
    ('N',               'N_MAX',                  'n_mid'),
    ('P',               'P_MAX',                  'p_mid'),
    ('K',               'K_MAX',                  'k_mid'),
]

for lo, hi, mid in pairs:
    if lo in df.columns:
        # If hi column is missing, use lo column
        hi_col = df[hi] if hi in df.columns else df[lo]
        df[mid] = (df[lo] + hi_col) / 2
    
# ── 9. Noise Injection ────────────────────────────────────────────────────────
# Add small Gaussian noise to numeric features to prevent perfect memorization
# and simulate real-world sensor/user variance.
print("\n[STEP] INJECTING NOISE (to reduce overfitting)")
noise_features = ['ph_mid', 'temp_mid', 'humidity_mid', 'water_mid', 'n_mid', 'p_mid', 'k_mid', 'duration_mid']
np.random.seed(42)

for feat in noise_features:
    if feat in df.columns:
        # If std is 0 (constant feature), use a small default to ensure some noise
        sigma = df[feat].std()
        if sigma == 0 or np.isnan(sigma):
            sigma = df[feat].mean() * 0.05
        
        # Inject 8% noise for better generalisation
        noise = np.random.normal(0, sigma * 0.08, df.shape[0])
        df[feat] = df[feat] + noise
        print(f"   Added noise to {feat}")

# ── 10. Categorical Encoding ──────────────────────────────────────────────────
if 'SEASON' in df.columns:
    season_map = {
        'kharif': 0, 'rabi': 1, 'zaid': 2, 'summer': 3,
        'all year': 4, 'whole year': 4, 'perennial': 4
    }
    # Map, fill unknown with -1 (uncommon season), make integer
    df['season_enc'] = df['SEASON'].map(season_map).fillna(-1).astype(int)

if 'WATER_SOURCE' in df.columns:
    # Frequency encoding might be better, but let's stick to label for now
    ws_map = {val: i for i, val in enumerate(df['WATER_SOURCE'].unique())}
    df['water_source_enc'] = df['WATER_SOURCE'].map(ws_map)

# ── 11. Final Save ─────────────────────────────────────────────────────────────
df.to_csv(OUT_PATH, index=False)
print(f"\n[OK] Processed data saved -> {OUT_PATH}")
print(f"   Final Shape: {df.shape}")
print("=" * 60)

