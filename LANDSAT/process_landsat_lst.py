import os
import glob
import numpy as np
import rasterio

# ==========================================
# 1. SETUP AND FILE DISCOVERY
# ==========================================
print("Looking for Landsat 8/9 files...")

# Automatically find the ST_B10 and QA_PIXEL files in the current folder
st_files = glob.glob("*_ST_B10.TIF")
qa_files = glob.glob("*_QA_PIXEL.TIF")

if not st_files or not qa_files:
    raise FileNotFoundError("Could not find the ST_B10 or QA_PIXEL files. Are they in this folder?")

st_filepath = st_files[0]
qa_filepath = qa_files[0]

print(f"Found Temperature File: {os.path.basename(st_filepath)}")
print(f"Found Cloud Mask File: {os.path.basename(qa_filepath)}")

# ==========================================
# 2. READ THE DATA
# ==========================================
print("\nReading rasters...")
with rasterio.open(st_filepath) as st_src, rasterio.open(qa_filepath) as qa_src:
    # Read the raw temperature data as floats
    st_data = st_src.read(1).astype('float32')
    
    # Read the Quality Assessment (Cloud Mask) data
    qa_data = qa_src.read(1)
    
    # Save the geospatial metadata (already in UTM 44N!)
    profile = st_src.profile

# ==========================================
# 3. BUILD THE CLOUD MASK (Bitwise Operations)
# ==========================================
print("Masking out clouds and shadows...")
# In Landsat Collection 2 QA_PIXEL, specific "bits" indicate clouds
# Bit 0: Fill (No Data)
# Bit 1: Dilated Cloud
# Bit 2: Cirrus Cloud
# Bit 3: Cloud
# Bit 4: Cloud Shadow

# Create boolean masks for anything we don't want
is_fill = (qa_data & (1 << 0)) != 0
is_dilated_cloud = (qa_data & (1 << 1)) != 0
is_cirrus = (qa_data & (1 << 2)) != 0
is_cloud = (qa_data & (1 << 3)) != 0
is_cloud_shadow = (qa_data & (1 << 4)) != 0

# Combine them all into one "Bad Pixel" mask
bad_pixels = is_fill | is_dilated_cloud | is_cirrus | is_cloud | is_cloud_shadow

# ==========================================
# 4. APPLY NASA MATH (RAW -> KELVIN -> CELSIUS)
# ==========================================
print("Applying NASA scaling factors to convert to Celsius...")

# Formula from the MTL.txt file: (Pixel * 0.00341802 + 149.0) = Kelvin
# Then subtract 273.15 to get Celsius
lst_celsius = (st_data * 0.00341802 + 149.0) - 273.15

# Erase the temperature values where there were clouds or empty edges
lst_celsius[bad_pixels] = np.nan
lst_celsius[st_data == 0] = np.nan

# ==========================================
# 5. EXPORT THE FINAL GEOTIFF
# ==========================================
print("\nExporting final Urban Heat Island map...")
final_filename = "Landsat_LST_Celsius_UTM44N.tif"

# Update profile to handle decimals and NoData properly
profile.update(
    dtype=rasterio.float32,
    nodata=np.nan,
    compress='lzw' # Shrinks the file size significantly
)

with rasterio.open(final_filename, 'w', **profile) as dst:
    dst.write(lst_celsius, 1)

print(f"\nSUCCESS! File saved as: {final_filename}")
print("Drag this into QGIS, set symbology to 'Pseudocolor' (Magma or Spectral), and you have your Heat Map!")