"""
Prediction engine — loads trained model and returns crop recommendation
with hybrid AI logic (XGBoost + Rule-Based reasoning).
"""

import os
import calendar
import joblib
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, 'model')
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'processed_data.csv')

MONTH_ABBR = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
MONTH_NAME = {i: m for i, m in enumerate(calendar.month_abbr) if m}

SEASON_MONTHS = {
    'kharif':    [6, 7, 8, 9, 10],
    'rabi':      [10, 11, 12, 1, 2, 3],
    'zaid':      [3, 4, 5, 6],
    'summer':    [3, 4, 5, 6],
    'all year':  list(range(1, 13)),
    'whole year':list(range(1, 13)),
    'perennial': list(range(1, 13)),
}

def load_model():
    model    = joblib.load(os.path.join(MODEL_DIR, 'crop_model.pkl'))
    le       = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    scaler   = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    features = joblib.load(os.path.join(MODEL_DIR, 'features.pkl'))
    df       = pd.read_csv(DATA_PATH)
    return model, le, scaler, features, df


def predict_crop(inputs: dict, top_n: int = 5):
    model, le, scaler, features, df = load_model()
    filter_type = inputs.get('crop_type', 'Any').lower()

    # Prep features
    n_mid, p_mid, k_mid = inputs.get('n', 50.0), inputs.get('p', 50.0), inputs.get('k', 50.0)
    temp, humidity, ph = inputs.get('temp', 30.0), inputs.get('humidity', 70.0), inputs.get('ph', 6.5)
    
    feat_map = {'n_mid': n_mid, 'p_mid': p_mid, 'k_mid': k_mid, 'temp_mid': temp, 'humidity_mid': humidity, 'ph_mid': ph}
    X = np.array([[feat_map.get(f, 0) for f in features]])
    X_scaled = scaler.transform(X)

    # 1. AI PREDICTION (XGBoost)
    probs = model.predict_proba(X_scaled)[0]
    
    # 2. CROP TYPE FILTERING
    if filter_type != 'any':
        valid_crops = df[df['TYPE_OF_CROP'].str.lower() == filter_type]['CROPS'].unique()
        mask = np.array([c in valid_crops for c in le.classes_])
        probs = probs * mask
        if probs.sum() > 0: probs = probs / probs.sum()
        else: probs = model.predict_proba(X_scaled)[0]

    top_idx  = np.argsort(probs)[::-1][:top_n]
    # Round and cast to float to fix precision display issues
    top_preds = [(le.classes_[i], round(float(probs[i]) * 100, 1)) for i in top_idx]
    best_crop = top_preds[0][0]

    # 3. GET DATASET DETAILS
    crop_rows = df[df['CROPS'] == best_crop]
    row = crop_rows.iloc[0] if not crop_rows.empty else {}
    
    # 4. HYBRID GATEKEEPER & REASONING (Rule-Based)
    reasoning = _generate_reasoning(inputs, row)
    gatekeeper_warnings = _gatekeeper_logic(inputs, row)

    # Dynamic Date Logic
    month = inputs.get('month', 6)
    dur_min = row.get('CROPDURATION', 120)
    harvest_est = _harvest_date(month, dur_min)
    # Extract only the month name for the 'Harvest In' card
    harvest_mon = harvest_est.split(' ')[0] if harvest_est != "N/A" else "N/A"

    details = {
        'best_crop': best_crop,
        'crop_type': str(row.get('TYPE_OF_CROP', 'N/A')).title(),
        'soil': str(row.get('SOIL', 'N/A')).title(),
        'season': str(row.get('SEASON', 'N/A')).title(),
        'sown': MONTH_NAME.get(month, "N/A"),
        'harvested': harvest_mon,
        'water_source': str(row.get('WATER_SOURCE', 'N/A')).title(),
        'duration_min': row.get('CROPDURATION', 0),
        'duration_max': row.get('CROPDURATION_MAX', 0),
        'water_min_mm': row.get('WATERREQUIRED', 0),
        'water_max_mm': row.get('WATERREQUIRED_MAX', 0),
        'n_ideal': (row.get('N', 0), row.get('N_MAX', 0)),
        'p_ideal': (row.get('P', 0), row.get('P_MAX', 0)),
        'k_ideal': (row.get('K', 0), row.get('K_MAX', 0)),
        'season_check': _check_season(row.get('SEASON', 'N/A'), row.get('SOWN', 'N/A'), inputs.get('month', 6)),
        'npk_gap': _fertilizer_gap(n_mid, p_mid, k_mid, (row.get('N', 0), row.get('N_MAX', 0)), (row.get('P', 0), row.get('P_MAX', 0)), (row.get('K', 0), row.get('K_MAX', 0))),
        'water_estimate': _water_estimate(row.get('WATERREQUIRED', 0), row.get('WATERREQUIRED_MAX', 0), inputs.get('farm_acres', 1.0)),
        'harvest_date_est': _harvest_date(inputs.get('month', 6), row.get('CROPDURATION', 120)),
        'alt_crops_for_month': _crops_for_month(df, inputs.get('month', 6), best_crop),
        'reasoning': reasoning,
        'warnings': gatekeeper_warnings
    }

    return top_preds, details

def _gatekeeper_logic(inputs, row):
    """Hybrid AI component: Veto or warn based on environmental mismatch."""
    warnings = []
    rainfall = inputs.get('rainfall', 0)
    w_min = row.get('WATERREQUIRED', 0)
    w_src = inputs.get('water_source', 'rainfed').lower()
    
    # Drought risk
    if rainfall < (w_min * 0.4) and w_src == 'rainfed':
        warnings.append(f"🔴 **Extreme Risk:** This crop needs ~{w_min}mm water, but only {rainfall}mm rainfall is available. Irrigation is MANDATORY.")
    
    # Acidic soil warning
    if inputs.get('ph', 6.5) < 5.5:
        warnings.append("🟡 **Soil Acidic:** Consider adding lime to improve nutrient uptake for this crop.")
        
    return warnings

def _generate_reasoning(inputs, row):
    """Explain why the AI picked this crop."""
    reasons = []
    n_in, n_low, n_hi = inputs.get('n', 50), row.get('N', 0), row.get('N_MAX', 0)
    if n_low <= n_in <= n_hi: reasons.append("✅ Soil Nitrogen matches crop requirements.")
    
    ph_in, ph_low, ph_hi = inputs.get('ph', 6.5), row.get('SOIL_PH', 0), row.get('SOIL_PH_HIGH', 0)
    if ph_low <= ph_in <= ph_hi: reasons.append("✅ Soil pH is in the optimal range.")
    
    temp_in = inputs.get('temp', 30)
    if 20 <= temp_in <= 35: reasons.append(f"🌡️ Current temperature ({temp_in}°C) is biologically supportive.")
    
    if not reasons: reasons.append("💡 This crop is the closest biological match for your current soil and climate conditions.")
    return reasons

def _check_season(crop_season: str, sown: str, month: int) -> dict:
    valid_months = SEASON_MONTHS.get(str(crop_season).lower(), list(range(1, 13)))
    if month in valid_months:
        return {'status': '✅ Great timing!', 'message': f'Month {MONTH_NAME.get(month,"?")} is ideal.'}
    next_good = min(valid_months, key=lambda m: abs(m - month))
    return {'status': '⚠️ Not the best time', 'message': f'Try starting in {MONTH_NAME.get(next_good,"?")}.'}

def _fertilizer_gap(nu, pu, ku, ni, pi, ki):
    def gap(u, low, high):
        m = (low + high)/2 if high else low
        d = m - u
        return f"+{d:.1f} needed" if d > 0 else "✅ Ideal range"
    return {'N': gap(nu, ni[0], ni[1]), 'P': gap(pu, pi[0], pi[1]), 'K': gap(ku, ki[0], ki[1]),
            'N_ideal_range': f"{ni[0]}-{ni[1]}", 'P_ideal_range': f"{pi[0]}-{pi[1]}", 'K_ideal_range': f"{ki[0]}-{ki[1]}"}

def _water_estimate(w_min, w_max, farm_acres):
    try:
        liters = ((float(w_min) + float(w_max))/2) * farm_acres * 4046.86
        return f"{liters/1_000_000:.2f}M liters" if liters > 1e6 else f"{liters/1000:.0f}k liters"
    except: return "N/A"

def _harvest_date(sow_month, duration_days):
    try:
        import datetime
        h = datetime.date(datetime.date.today().year, sow_month, 1) + datetime.timedelta(days=float(duration_days))
        return h.strftime('%B %Y')
    except: return "N/A"

def _crops_for_month(df, month, exclude):
    alts = []
    for _, row in df.drop_duplicates('CROPS').iterrows():
        if month in SEASON_MONTHS.get(str(row.get('SEASON', '')).lower(), []) and row['CROPS'] != exclude:
            alts.append(str(row['CROPS']).title())
    return sorted(set(alts))[:8]
