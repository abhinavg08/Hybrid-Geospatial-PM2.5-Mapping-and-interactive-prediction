#!/usr/bin/env python3
import os
import glob
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# 1. Setup paths
BASE_DIR = "."
CSV_FILES = glob.glob(os.path.join(BASE_DIR, "Raw_data_*_2025_*.csv"))

if not CSV_FILES:
    print(f"❌ No 2025 Raw Data CSV files found in {BASE_DIR}. Check your folder path!")
    exit()

print(f"🔄 Found {len(CSV_FILES)} CPCB 2025 Raw Data CSV files. Initiating Smart Parser...")

# 2. Hardcoded coordinates for the 14 Hyderabad stations
STATION_COORDS = {
    "Zoo_Park": (17.3497, 78.4513),
    "Central_University": (17.4585, 78.3288),
    "IITH_Kandi": (17.5280, 78.1235),
    "Nacharam": (17.4410, 78.5815),
    "New_Malakpet": (17.3739, 78.4984),
    "ECIL_Kapra": (17.4725, 78.5620),
    "Kokapet": (17.3948, 78.3242),
    "Bollaram": (17.5451, 78.3512),
    "Ramachandrapuram": (17.5028, 78.3039),
    "Kompally": (17.5342, 78.4842),
    "Somajiguda": (17.4246, 78.4552),
    "IDA_Pashamylaram": (17.5262, 78.1887),
    "Sanathnagar": (17.4566, 78.4436),
    "ICRISAT": (17.5113, 78.2745)
}

master_list = []

# 3. Process each file
for file in CSV_FILES:
    filename = os.path.basename(file)
    
    # Identify the station
    station_key = None
    for key in STATION_COORDS.keys():
        if key in filename:
            station_key = key
            break
            
    if not station_key:
        print(f"⚠️ Could not match station name for {filename}. Skipping.")
        continue

    lat, lon = STATION_COORDS[station_key]

    try:
        # Check first 30 rows for headers
        df = None
        for skip in range(30):
            try:
                temp_df = pd.read_csv(file, skiprows=skip)
                cols = [str(c).lower() for c in temp_df.columns]
                
                # Look for date/timestamp AND PM2.5
                has_date = any('date' in c or 'timestamp' in c for c in cols)
                has_pm25 = any('pm2.5' in c or 'pm 2.5' in c for c in cols)
                
                if has_date and has_pm25:
                    df = temp_df
                    break
            except Exception:
                continue
                
        if df is None:
            print(f"⚠️ Could not find a valid table in {filename}.")
            continue
            
        # Locate the exact Date/Timestamp column
        date_col = next((c for c in df.columns if "timestamp" in str(c).lower() or "date" in str(c).lower()), None)
        pm25_col = next((c for c in df.columns if "pm2.5" in str(c).lower() or "pm 2.5" in str(c).lower()), None)
        
        # Extract the data
        df = df[[date_col, pm25_col]].copy()
        df.columns = ["date", "actual_pm25"]
        
        # Clean the data (Handle "None" text, NaNs, etc.)
        df['actual_pm25'] = pd.to_numeric(df['actual_pm25'], errors='coerce')
        # Format the datetime object 
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'actual_pm25'])
        
        # Add metadata
        df['station_name'] = station_key.replace("_", " ")
        df['lat'] = lat
        df['lon'] = lon
        
        # Reorder and sort
        df_clean = df[['station_name', 'lat', 'lon', 'date', 'actual_pm25']].sort_values('date')
        
        master_list.append(df_clean)
        print(f"  ✅ Processed {station_key}: extracted {len(df_clean)} valid daily readings.")
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

# 4. Combine and Save
if master_list:
    final_df = pd.concat(master_list, ignore_index=True)
    
    out_path = os.path.join(BASE_DIR, "CPCB_Actual_PM25.csv")
    final_df.to_csv(out_path, index=False)
    
    print("\n" + "="*50)
    print("🎉 2025 DATA STITCHING COMPLETE! 🎉")
    print("="*50)
    print(f"Total combined 2025 ground-truth data points: {len(final_df)}")
    print(f"Saved cleanly to: {out_path}")
else:
    print("No data was successfully processed.")