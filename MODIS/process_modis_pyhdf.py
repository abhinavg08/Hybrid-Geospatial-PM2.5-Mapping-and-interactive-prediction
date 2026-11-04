import os
import glob
import numpy as np
import warnings
from pyhdf.SD import SD, SDC
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Ignore the "Mean of empty slice" warning for cloudy pixels
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
input_folder = "."  
search_pattern = os.path.join(input_folder, "MODIS_AOD_Feb*.hdf")

TARGET_SUBDATASET = 'Optical_Depth_047'
NODATA_VALUE = -28672
SCALE_FACTOR = 0.001

# Coordinate Systems
MODIS_CRS = CRS.from_proj4("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs")
DST_CRS = CRS.from_epsg(32644) # UTM 44N for Hyderabad

hdf_files = glob.glob(search_pattern)
if not hdf_files:
    raise FileNotFoundError(f"No files found matching {search_pattern}")

print(f"Found {len(hdf_files)} MODIS files. Starting extraction...")

datasets = []

# ==========================================
# 2. READ ALL DATA AND EXTRACT BOUNDS
# ==========================================
for file in hdf_files:
    print(f"Reading: {os.path.basename(file)}")
    hdf = SD(file, SDC.READ)
    struct_meta = hdf.attributes()['StructMetadata.0']
    ul_x, ul_y, lr_x, lr_y = None, None, None, None

    for line in struct_meta.split('\n'):
        if 'UpperLeftPointMtrs' in line:
            ul_coords = line.split('=')[1].strip('()').split(',')
            ul_x, ul_y = float(ul_coords[0]), float(ul_coords[1])
        elif 'LowerRightMtrs' in line:
            lr_coords = line.split('=')[1].strip('()').split(',')
            lr_x, lr_y = float(lr_coords[0]), float(lr_coords[1])

    sds_obj = hdf.select(TARGET_SUBDATASET)
    data = sds_obj.get().astype('float32')

    # Mask clouds and scale
    data[data == NODATA_VALUE] = np.nan
    data = data * SCALE_FACTOR

    # ==========================================
    # THE FIX: FLATTEN MULTIPLE ORBITS
    # ==========================================
    if data.ndim == 3:
        # Average across the orbit dimension (axis=0) to flatten it to 2D
        data = np.nanmean(data, axis=0)
    elif data.ndim > 3:
        # Just in case NASA throws a 4D array at us
        data = np.squeeze(data)
        if data.ndim == 3:
            data = np.nanmean(data, axis=0)

    h, w = data.shape
    # Create the mathematical transform for this specific file
    src_transform = from_bounds(ul_x, lr_y, lr_x, ul_y, w, h)

    datasets.append({
        'data': data,
        'transform': src_transform,
        'width': w,
        'height': h,
        'bounds': (ul_x, lr_y, lr_x, ul_y) # Left, Bottom, Right, Top
    })
    hdf.end()

# ==========================================
# 3. CREATE THE MASTER GRID (UTM 44N)
# ==========================================
# We use the bounds of the first file to define our "Master Canvas"
master = datasets[0]
dst_transform, dst_width, dst_height = calculate_default_transform(
    MODIS_CRS, DST_CRS, master['width'], master['height'], *master['bounds']
)

# Prepare a blank 3D array perfectly sized to the Master Canvas
aligned_arrays = np.empty((len(datasets), dst_height, dst_width), dtype=np.float32)
aligned_arrays[:] = np.nan

# ==========================================
# 4. REPROJECT ALL FILES TO THE MASTER GRID
# ==========================================
print("\nAligning and reprojecting all rasters to a unified Master Grid...")
for i, ds in enumerate(datasets):
    reproject(
        source=ds['data'],
        destination=aligned_arrays[i],
        src_transform=ds['transform'],
        src_crs=MODIS_CRS,
        dst_transform=dst_transform,
        dst_crs=DST_CRS,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear
    )

# ==========================================
# 5. CALCULATE MEAN & EXPORT
# ==========================================
print("Calculating the 10-day Mean composite...")
mean_aod = np.nanmean(aligned_arrays, axis=0)

final_filename = "MODIS_AOD_Feb1_10_Mean_UTM44N.tif"

with rasterio.open(
    final_filename, 'w',
    driver='GTiff',
    height=dst_height, width=dst_width,
    count=1, dtype=np.float32,
    crs=DST_CRS, transform=dst_transform,
    nodata=np.nan
) as dst:
    dst.write(mean_aod, 1)

print(f"\nSUCCESS! All overlapping arrays aligned and averaged.")
print(f"Final file saved as: {final_filename}")