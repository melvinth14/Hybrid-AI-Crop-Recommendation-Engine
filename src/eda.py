"""
======================================================
  TAMIL NADU CROP RECOMMENDATION SYSTEM
  Step 1: Exploratory Data Analysis (EDA)
======================================================
Run:  python src/eda.py
Outputs are saved to:  outputs/eda/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'crop_data.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs', 'eda')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='Set2', font_scale=1.1)
COLORS = sns.color_palette('Set2')

print("=" * 60)
print("  TAMIL NADU CROP RECOMMENDATION — EDA")
print("=" * 60)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. Basic Info ──────────────────────────────────────────────────────────────
print("\n[INFO] COLUMNS & DTYPES")
print("-" * 40)
print(df.dtypes.to_string())

print("\n[INFO] BASIC STATISTICS")
print("-" * 40)
print(df.describe().round(2).to_string())

# ── 3. Missing Values ──────────────────────────────────────────────────────────
print("\n[INFO] MISSING VALUES")
print("-" * 40)
nulls = df.isnull().sum()
nulls_pct = (nulls / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing': nulls, 'Pct (%)': nulls_pct})
missing_df = missing_df[missing_df['Missing'] > 0]
if missing_df.empty:
    print("[OK] No missing values found!")
else:
    print(missing_df)

# ── 4. Duplicates ──────────────────────────────────────────────────────────────
dupes = df.duplicated().sum()
print(f"\n[INFO] DUPLICATES: {dupes:,} rows")

# ── 5. Unique Values in Categorical Columns ───────────────────────────────────
cat_cols = ['CROPS', 'TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED', 'WATER_SOURCE']
print("\n[INFO] UNIQUE VALUES IN CATEGORICAL COLUMNS")
print("-" * 40)
for col in cat_cols:
    if col in df.columns:
        vals = df[col].unique().tolist()
        print(f"  {col} ({df[col].nunique()} unique): {vals[:10]}")

# ── 6. Crop Distribution ───────────────────────────────────────────────────────
print("\n[INFO] CROP DISTRIBUTION")
print("-" * 40)
crop_counts = df['CROPS'].value_counts()
print(crop_counts.to_string())

fig, ax = plt.subplots(figsize=(14, 6))
crop_counts.plot(kind='bar', ax=ax, color=COLORS[0], edgecolor='white')
ax.set_title('Crop Distribution in Dataset', fontsize=16, fontweight='bold')
ax.set_xlabel('Crop', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.tick_params(axis='x', rotation=45)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_crop_distribution.png'), dpi=150)
plt.close()
print("  -> Saved: 01_crop_distribution.png")

# ── 7. Season Distribution ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

season_counts = df['SEASON'].value_counts()
axes[0].pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%',
            colors=COLORS, startangle=90)
axes[0].set_title('Season Distribution', fontsize=14, fontweight='bold')

soil_counts = df['SOIL'].value_counts()
axes[1].pie(soil_counts.values, labels=soil_counts.index, autopct='%1.1f%%',
            colors=COLORS, startangle=90)
axes[1].set_title('Soil Type Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_season_soil_distribution.png'), dpi=150)
plt.close()
print("  -> Saved: 02_season_soil_distribution.png")

# ── 8. Crop Type vs Season ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ct = pd.crosstab(df['TYPE_OF_CROP'], df['SEASON'])
ct.plot(kind='bar', ax=ax, edgecolor='white')
ax.set_title('Crop Type by Season', fontsize=14, fontweight='bold')
ax.set_xlabel('Crop Type', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.legend(title='Season', bbox_to_anchor=(1.05, 1))
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_crop_type_by_season.png'), dpi=150)
plt.close()
print("  -> Saved: 03_crop_type_by_season.png")

# ── 9. NPK Distributions ──────────────────────────────────────────────────────
num_cols = ['N', 'N_MAX', 'P', 'P_MAX', 'K', 'K_MAX']
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    if col in df.columns:
        axes[i].hist(df[col], bins=30, color=COLORS[i % len(COLORS)], edgecolor='white', alpha=0.8)
        axes[i].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={df[col].mean():.1f}')
        axes[i].legend(fontsize=9)
plt.suptitle('NPK Value Distributions', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_npk_distributions.png'), dpi=150)
plt.close()
print("  -> Saved: 04_npk_distributions.png")

# ── 10. Temperature & Humidity Distributions ───────────────────────────────────
clim_cols = ['TEMP', 'MAX_TEMP', 'RELATIVE_HUMIDITY', 'RELATIVE_HUMIDITY_MAX',
             'WATERREQUIRED', 'WATERREQUIRED_MAX']
existing_clim = [c for c in clim_cols if c in df.columns]
n = len(existing_clim)
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(existing_clim):
    axes[i].hist(df[col], bins=30, color=COLORS[(i+2) % len(COLORS)], edgecolor='white', alpha=0.8)
    axes[i].set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Climate & Water Distributions', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05_climate_distributions.png'), dpi=150)
plt.close()
print("  -> Saved: 05_climate_distributions.png")

# ── 11. NPK Box Plots by Crop (Top 10 crops) ──────────────────────────────────
top10 = df['CROPS'].value_counts().head(10).index
df_top = df[df['CROPS'].isin(top10)]

for nutrient in [('N', 'N_MAX'), ('P', 'P_MAX'), ('K', 'K_MAX')]:
    col_min, col_max = nutrient
    df_top = df_top.copy()
    mid_col = f'{col_min}_mid'
    df_top[mid_col] = (df_top[col_min] + df_top[col_max]) / 2
    fig, ax = plt.subplots(figsize=(14, 5))
    df_top.boxplot(column=mid_col, by='CROPS', ax=ax, grid=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=COLORS[0], color='navy'),
                   medianprops=dict(color='red', linewidth=2))
    ax.set_title(f'{col_min} Requirement by Crop (Top 10)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Crop', fontsize=11)
    ax.set_ylabel(f'{col_min} value', fontsize=11)
    plt.suptitle('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'06_{col_min}_by_crop.png'), dpi=150)
    plt.close()
    print(f"  -> Saved: 06_{col_min}_by_crop.png")

# ── 12. Correlation Heatmap ────────────────────────────────────────────────────
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, ax=ax, annot_kws={'size': 8},
            vmin=-1, vmax=1, center=0)
ax.set_title('Feature Correlation Heatmap', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '07_correlation_heatmap.png'), dpi=150)
plt.close()
print("  -> Saved: 07_correlation_heatmap.png")

# ── 13. Soil pH by Crop ────────────────────────────────────────────────────────
if 'SOIL_PH' in df.columns and 'SOIL_PH_HIGH' in df.columns:
    df['pH_mid'] = (df['SOIL_PH'] + df['SOIL_PH_HIGH']) / 2
    top15_crops = df['CROPS'].value_counts().head(15).index
    df_ph = df[df['CROPS'].isin(top15_crops)]
    fig, ax = plt.subplots(figsize=(14, 5))
    crop_ph = df_ph.groupby('CROPS')['pH_mid'].mean().sort_values()
    bars = ax.barh(crop_ph.index, crop_ph.values, color=COLORS[3], edgecolor='white')
    ax.set_title('Average Soil pH Requirement by Crop', fontsize=14, fontweight='bold')
    ax.set_xlabel('Soil pH (midpoint)', fontsize=12)
    ax.axvline(7, color='red', linestyle='--', linewidth=1.5, label='Neutral pH=7')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '08_soil_ph_by_crop.png'), dpi=150)
    plt.close()
    print("  -> Saved: 08_soil_ph_by_crop.png")

# ── 14. Water Source Distribution ─────────────────────────────────────────────
if 'WATER_SOURCE' in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    ws = df['WATER_SOURCE'].value_counts()
    bars = ax.bar(ws.index, ws.values, color=COLORS, edgecolor='white')
    ax.set_title('Water Source Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Water Source', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{int(bar.get_height()):,}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '09_water_source.png'), dpi=150)
    plt.close()
    print("  -> Saved: 09_water_source.png")

# ── 15. Crop Duration Analysis ─────────────────────────────────────────────────
if 'CROPDURATION' in df.columns and 'CROPDURATION_MAX' in df.columns:
    df['duration_mid'] = (df['CROPDURATION'] + df['CROPDURATION_MAX']) / 2
    top15 = df['CROPS'].value_counts().head(15).index
    dur_df = df[df['CROPS'].isin(top15)]
    dur_avg = dur_df.groupby('CROPS')['duration_mid'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(dur_avg.index, dur_avg.values, color=COLORS[1], edgecolor='white')
    ax.set_title('Average Crop Duration (days) by Crop', fontsize=14, fontweight='bold')
    ax.set_xlabel('Duration (days)', fontsize=12)
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.0f}d', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '10_crop_duration.png'), dpi=150)
    plt.close()
    print("  -> Saved: 10_crop_duration.png")

print("\n" + "=" * 60)
print(f"[OK] EDA COMPLETE — {len(os.listdir(OUT_DIR))} plots saved to: outputs/eda/")
print("=" * 60)
