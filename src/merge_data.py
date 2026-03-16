import pandas as pd
import os

BASE_DIR = r"c:\Users\ASUS\spem\crop_recommendation_tn"
TN_DATA_PATH = os.path.join(BASE_DIR, "data", "crop_data.csv")
NEW_DATA_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")

# 1. SMART MAPPING DICTIONARY (All 22 crops from the 2,200-row dataset)
# Data source: General agricultural standards for these crops
CROP_WISDOM = {
    'rice':        {'type':'cereals','soil': 'Clay soil', 'season': 'Kharif', 'sown': 'Jun-Jul', 'harvest': 'Oct-Nov', 'water': 'High'},
    'maize':       {'type':'cereals','soil': 'Loamy soil', 'season': 'Kharif/Rabi', 'sown': 'Jun-Jul', 'harvest': 'Sep-Oct', 'water': 'Moderate'},
    'chickpea':    {'type':'pulses','soil': 'Black soil', 'season': 'Rabi', 'sown': 'Oct-Nov', 'harvest': 'Feb-Mar', 'water': 'Low'},
    'kidneybeans': {'type':'pulses','soil': 'Loamy soil', 'season': 'Kharif', 'sown': 'Jun-Jul', 'harvest': 'Sep-Oct', 'water': 'Moderate'},
    'pigeonpeas':  {'type':'pulses','soil': 'Red soil', 'season': 'Kharif', 'sown': 'Jun-Jul', 'harvest': 'Dec-Jan', 'water': 'Low'},
    'mothbeans':   {'type':'pulses','soil': 'Sandy soil', 'season': 'Kharif', 'sown': 'Jul-Aug', 'harvest': 'Oct-Nov', 'water': 'Very Low'},
    'mungbean':    {'type':'pulses','soil': 'Loamy soil', 'season': 'Kharif/Rabi', 'sown': 'Jun/Nov', 'harvest': 'Aug/Jan', 'water': 'Moderate'},
    'blackgram':   {'type':'pulses','soil': 'Clay/Loamy soil', 'season': 'Kharif/Rabi', 'sown': 'Jun/Nov', 'harvest': 'Aug/Jan', 'water': 'Moderate'},
    'lentil':      {'type':'pulses','soil': 'Alluvial soil', 'season': 'Rabi', 'sown': 'Oct-Nov', 'harvest': 'Feb-Mar', 'water': 'Low'},
    'pomegranate': {'type':'fruits','soil': 'Red soil', 'season': 'Whole Year', 'sown': 'Jun-Jul', 'harvest': 'Dec-Jan', 'water': 'Low'},
    'banana':      {'type':'fruits','soil': 'Alluvial soil', 'season': 'Whole Year', 'sown': 'Any Month', 'harvest': '10-12 months', 'water': 'High'},
    'mango':       {'type':'fruits','soil': 'Deep loamy', 'season': 'Perennial', 'sown': 'Jul-Sep', 'harvest': 'Mar-May', 'water': 'Rainfed'},
    'grapes':      {'type':'fruits','soil': 'Sandy loamy', 'season': 'Rabi', 'sown': 'Oct-Nov', 'harvest': 'Feb-Apr', 'water': 'Drip irrigated'},
    'watermelon':  {'type':'fruits','soil': 'Sandy soil', 'season': 'Summer/Zaid', 'sown': 'Jan-Feb', 'harvest': 'Apr-May', 'water': 'Moderate'},
    'muskmelon':   {'type':'fruits','soil': 'Sandy soil', 'season': 'Summer/Zaid', 'sown': 'Jan-Feb', 'harvest': 'Apr-May', 'water': 'Moderate'},
    'apple':       {'type':'fruits','soil': 'Loamy soil', 'season': 'Perennial', 'sown': 'Jan-Feb', 'harvest': 'Aug-Oct', 'water': 'Rainfed'},
    'orange':      {'type':'fruits','soil': 'Black soil', 'season': 'Perennial', 'sown': 'Jun-Aug', 'harvest': 'Dec-Feb', 'water': 'Irrigated'},
    'papaya':      {'type':'fruits','soil': 'Sandy soil', 'season': 'Whole Year', 'sown': 'Any Month', 'harvest': '9-10 months', 'water': 'Well-drained'},
    'coconut':     {'type':'commercial','soil': 'Sandy coastal', 'season': 'Perennial', 'sown': 'Jun-Sep', 'harvest': 'Whole Year', 'water': 'High'},
    'cotton':      {'type':'fibre crop','soil': 'Black soil', 'season': 'Kharif', 'sown': 'Jun-Jul', 'harvest': 'Oct-Dec', 'water': 'Moderate'},
    'jute':        {'type':'fibre crop','soil': 'Alluvial soil', 'season': 'Kharif', 'sown': 'Mar-Apr', 'harvest': 'Jul-Aug', 'water': 'High'},
    'coffee':      {'type':'commercial','soil': 'Laterite soil', 'season': 'Perennial', 'sown': 'Jun-Aug', 'harvest': 'Nov-Jan', 'water': 'Rainfed'}
}

# 2. Load and Clean
if not os.path.exists(TN_DATA_PATH):
    print("Error: TN crop_data.csv not found.")
    exit()

df_tn = pd.read_csv(TN_DATA_PATH)
df_tn.columns = [c.strip().upper() for c in df_tn.columns]

# Get list of existing crops to avoid duplicates
existing_crops = set(df_tn['CROPS'].astype(str).str.lower().unique())
print(f"Crops already in 57k dataset: {len(existing_crops)}")

df_new = pd.read_csv(NEW_DATA_PATH)
df_new['label'] = df_new['label'].str.lower()

# 3. Process Only "New" Crops (Not in 57k)
mapped_rows = []
skipped_count = 0
added_crops = set()

for i, row in df_new.iterrows():
    c_label = row['label']
    
    # Check if this crop is already in the 57k dataset
    if c_label in existing_crops:
        skipped_count += 1
        continue
        
    wisdom = CROP_WISDOM.get(c_label, {'type':'other', 'soil': 'Loamy soil', 'season': 'Whole Year', 'sown': 'See Nursery', 'harvest': 'Varies', 'water': 'Irrigated'})
    
    added_crops.add(c_label)
    mapped_rows.append({
        'CROPS': c_label.title(),
        'TYPE_OF_CROP': wisdom['type'],
        'SOIL': wisdom['soil'],
        'SEASON': wisdom['season'],
        'SOWN': wisdom['sown'],
        'HARVESTED': wisdom['harvest'],
        'WATER_SOURCE': wisdom['water'],
        'SOIL_PH': row['ph'],
        'SOIL_PH_HIGH': row['ph'],
        'CROPDURATION': 120,
        'CROPDURATION_MAX': 150,
        'TEMP': row['temperature'],
        'MAX_TEMP': row['temperature'],
        'WATERREQUIRED': row['rainfall'],
        'WATERREQUIRED_MAX': row['rainfall'],
        'RELATIVE_HUMIDITY': row['humidity'],
        'RELATIVE_HUMIDITY_MAX': row['humidity'],
        'N': row['N'],
        'N_MAX': row['N'],
        'P': row['P'],
        'P_MAX': row['P'],
        'K': row['K'],
        'K_MAX': row['K']
    })

print(f"Skipped {skipped_count} rows (Crops: {existing_crops.intersection(set(df_new['label'].unique()))})")
print(f"Adding {len(mapped_rows)} rows for new crops: {added_crops}")

if mapped_rows:
    df_final = pd.concat([df_tn, pd.DataFrame(mapped_rows)], ignore_index=True)
    df_final.to_csv(TN_DATA_PATH, index=False)
    print(f"Successfully merged new crops. Total rows: {len(df_final)}")
else:
    print("No new crops to add.")
