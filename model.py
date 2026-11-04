#!/usr/bin/env python3
"""
PM2.5 LUR Pipeline v9.5 — Production Version
Philosophy:
  - IDW surface as training target (legitimate hybrid LUR methodology)
  - Environmental features only as predictors (no coordinate leakage)
  - Spatial block CV for honest generalization assessment
  - CPCB station comparison as external validation
  - Full 500m grid prediction with bias correction
  - All 17 figures
"""

import os, sys, warnings, pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns
from shapely.geometry import Point, box
from scipy.stats import pearsonr
from scipy.ndimage import sobel, uniform_filter
from scipy import stats as sp_stats
import rasterio
from rasterstats import zonal_stats

from sklearn.model_selection import (
    KFold, cross_val_predict, cross_validate,
    learning_curve, RandomizedSearchCV
)
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from scipy.spatial import cKDTree

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

if HAS_LGB:
    class LGBMFeatureNameWrapper(lgb.LGBMRegressor):
        def __init__(self, feature_names=None, **kwargs):
            super().__init__(**kwargs)
            self.feature_names = feature_names

        def fit(self, X, y, **fit_params):
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            return super().fit(X, y, **fit_params)

        def predict(self, X, **predict_params):
            if isinstance(X, np.ndarray) and self.feature_names is not None:
                X = pd.DataFrame(X, columns=self.feature_names)
            return super().predict(X, **predict_params)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR  = os.path.join(BASE_DIR, "Vector_Data")
RASTER_DIR  = os.path.join(BASE_DIR, "Raster_Data")
LANDSAT_DIR = os.path.join(BASE_DIR, "Landsat")
MODIS_DIR   = os.path.join(BASE_DIR, "MODIS")
OUTPUT_DIR  = os.path.join(BASE_DIR, "Output")
ML_DIR      = os.path.join(OUTPUT_DIR, "ML_Results_v9.5_Production")
FIG_DIR     = os.path.join(ML_DIR, "Figures")
RPT_DIR     = os.path.join(ML_DIR, "Report")
for d in (ML_DIR, FIG_DIR, RPT_DIR):
    os.makedirs(d, exist_ok=True)

CPCB_CSV     = os.path.join(BASE_DIR, "CPCB_Actual_PM25.csv")
CRS_GEO      = "EPSG:4326"
CRS_UTM      = "EPSG:32644"
RANDOM_STATE = 42


def heading(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


def savefig(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def load_shp(name):
    return gpd.read_file(os.path.join(VECTOR_DIR, name))


def arr_to_tif(arr, src_obj, path):
    prof = src_obj.profile.copy()
    prof.update(dtype="float32", nodata=-9999., count=1)
    out = arr.copy()
    out[np.isnan(out)] = -9999.
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out.astype("float32"), 1)


def run_zonal(vec_gdf, raster_path, col_name):
    try:
        with rasterio.open(raster_path) as src:
            rcrs = src.crs.to_string()
        v  = (vec_gdf.to_crs(rcrs)
              if vec_gdf.crs.to_string() != rcrs else vec_gdf)
        st = zonal_stats(v, raster_path, stats=["mean","std"],
                         nodata=-9999., all_touched=True)
        means = [s["mean"] if s and s["mean"] is not None
                 else np.nan for s in st]
        stds  = [s["std"]  if s and s["std"]  is not None
                 else np.nan for s in st]
        return (pd.Series(means, name=col_name),
                pd.Series(stds,  name=f"{col_name}_std"))
    except Exception as e:
        print(f"    [WARN] {col_name}: {e}")
        n = len(vec_gdf)
        return (pd.Series([np.nan]*n, name=col_name),
                pd.Series([np.nan]*n, name=f"{col_name}_std"))


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 1: LOAD DATA                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
heading("STAGE 1: LOAD DATA")

pm_df = pd.read_csv(CPCB_CSV)
pm25_col    = next(c for c in pm_df.columns
                   if "pm2.5" in c.lower() or "actual_pm25" in c.lower())
date_col    = next((c for c in pm_df.columns
                    if "date" in c.lower()), None)
station_col = next((c for c in pm_df.columns
                    if "station" in c.lower()), None)
lat_col     = next(c for c in pm_df.columns
                   if c.lower() in ("lat","latitude"))
lon_col     = next(c for c in pm_df.columns
                   if c.lower() in ("lon","longitude"))

pm_df[pm25_col] = pd.to_numeric(pm_df[pm25_col], errors="coerce")
if date_col:
    pm_df[date_col] = pd.to_datetime(pm_df[date_col], errors="coerce")
pm_df = pm_df.dropna(subset=[pm25_col])

agg = pm_df.groupby(station_col).agg(
    lat=(lat_col,    "first"),
    lon=(lon_col,    "first"),
    pm25_mean=(pm25_col, "mean"),
    pm25_std=(pm25_col,  "std"),
    pm25_count=(pm25_col,"count")
).reset_index()

stations_geo = gpd.GeoDataFrame(
    agg,
    geometry=[Point(lo,la) for lo,la in zip(agg.lon, agg.lat)],
    crs=CRS_GEO
)
stations_utm = stations_geo.to_crs(CRS_UTM)
n_stations   = len(stations_utm)
st_coords    = np.array([[p.x,p.y] for p in stations_utm.geometry])
st_pm25      = agg["pm25_mean"].values

print(f"  Stations : {n_stations}")
print(f"  PM2.5    : {st_pm25.min():.1f}–{st_pm25.max():.1f} "
      f"mean={st_pm25.mean():.1f} µg/m³")
print(agg[["station_name","pm25_mean","pm25_std"]].to_string(index=False))

boundary     = load_shp("01_Hyderabad_Boundary.shp")
boundary_utm = boundary.to_crs(CRS_UTM)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 2: SPATIAL DATA & GRID                                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
heading("STAGE 2: SPATIAL DATA & GRID GENERATION")

pri_roads = load_shp("02_Primary_Roads.shp").to_crs(CRS_UTM)
sec_roads = load_shp("03_Secondary_Roads.shp").to_crs(CRS_UTM)
signals   = load_shp("04_Traffic_Signals.shp").to_crs(CRS_UTM)
bus       = load_shp("05_Bus_Stops.shp").to_crs(CRS_UTM)
rail      = load_shp("06_Railway_Lines.shp").to_crs(CRS_UTM)
indust    = load_shp("07_Industrial_Areas.shp").to_crs(CRS_UTM)
resid     = load_shp("08_Residential_Areas.shp").to_crs(CRS_UTM)
water     = load_shp("09_Water_Bodies.shp").to_crs(CRS_UTM)

CELL_SIZE_M = 500
try:
    grid_name = f"10_Analysis_Grid_{CELL_SIZE_M}m.shp"
    grid_path = os.path.join(VECTOR_DIR, grid_name)
    if os.path.exists(grid_path):
        grid = load_shp(grid_name).to_crs(CRS_UTM)
        print(f"  Loaded existing grid: {grid_name}")
    else:
        raise FileNotFoundError
    if len(grid) < 10:
        raise ValueError
except Exception:
    print(f"  Generating {CELL_SIZE_M}m grid...")
    b = boundary_utm
    x0,y0,x1,y1 = b.total_bounds
    cells = [
        box(x,y,x+CELL_SIZE_M,y+CELL_SIZE_M)
        for x in np.arange(x0,x1,CELL_SIZE_M)
        for y in np.arange(y0,y1,CELL_SIZE_M)
    ]
    grid_gdf = gpd.GeoDataFrame(geometry=cells, crs=CRS_UTM)
    grid     = gpd.clip(grid_gdf, b).reset_index(drop=True)
    print(f"  Generated {len(grid)} cells.")

grid["_gid"] = grid.index
print(f"  Grid Resolution : {CELL_SIZE_M}m × {CELL_SIZE_M}m")
print(f"  Total Cells     : {len(grid)}")

# ── Rasters ───────────────────────────────────────────────────────────────────
dem_src = rasterio.open(os.path.join(RASTER_DIR,"SRTM_DEM.tif"))
dem_arr = dem_src.read(1).astype(float)
nd = dem_src.nodata
if nd: dem_arr[dem_arr==nd] = np.nan

px_m = abs(dem_src.res[0])
if px_m < 1: px_m *= 111320

dc         = np.nan_to_num(dem_arr, nan=float(np.nanmean(dem_arr)))
sx         = sobel(dc,axis=1)/(8*px_m)
sy         = sobel(dc,axis=0)/(8*px_m)
slope_arr  = np.degrees(np.arctan(np.sqrt(sx**2+sy**2)))
slope_arr[np.isnan(dem_arr)] = np.nan
tpi_arr    = dem_arr - uniform_filter(dc,size=15)
aspect_arr = np.degrees(np.arctan2(-sy,sx)) % 360

lst_path = os.path.join(LANDSAT_DIR,"Landsat_LST_Celsius_UTM44N.tif")
lst_src  = rasterio.open(lst_path) if os.path.exists(lst_path) else None
lst_arr  = None
if lst_src:
    lst_arr = lst_src.read(1).astype(float)
    nd = lst_src.nodata
    if nd: lst_arr[lst_arr==nd] = np.nan
    lst_arr[(lst_arr<-50)|(lst_arr>70)] = np.nan

aod_path = os.path.join(MODIS_DIR,"MODIS_AOD_Feb1_10_Mean_UTM44N.tif")
aod_src  = rasterio.open(aod_path) if os.path.exists(aod_path) else None
aod_arr  = None
if aod_src:
    aod_arr = aod_src.read(1).astype(float)
    nd = aod_src.nodata
    if nd: aod_arr[aod_arr==nd] = np.nan
    aod_arr[aod_arr<0] = np.nan

ELEV_MEAN   = float(np.nanmean(dem_arr))
LST_MEAN    = float(np.nanmean(lst_arr))  if lst_arr is not None else 30.0
LST_STD     = float(np.nanstd(lst_arr))   if lst_arr is not None else 2.0
AOD_MEAN    = float(np.nanmean(aod_arr))  if aod_arr is not None else 0.5
HAS_MODIS   = aod_arr is not None
HAS_LANDSAT = lst_arr is not None

print(f"  ELEV={ELEV_MEAN:.0f}m  LST={LST_MEAN:.1f}°C  AOD={AOD_MEAN:.4f}")
print(f"  MODIS={HAS_MODIS}  LANDSAT={HAS_LANDSAT}")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 3: FEATURE EXTRACTION                                               ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
heading("STAGE 3: FEATURE EXTRACTION")

grid_feats = grid[["_gid","geometry"]].copy()
grid_utm   = grid_feats.copy()

slope_tif  = os.path.join(ML_DIR,"_slope.tif")
aspect_tif = os.path.join(ML_DIR,"_aspect.tif")
tpi_tif    = os.path.join(ML_DIR,"_tpi.tif")
arr_to_tif(slope_arr,  dem_src, slope_tif)
arr_to_tif(aspect_arr, dem_src, aspect_tif)
arr_to_tif(tpi_arr,    dem_src, tpi_tif)

print("  Raster zonal statistics...")
for path,col in [
    (os.path.join(RASTER_DIR,"SRTM_DEM.tif"), "mean_elev"),
    (slope_tif,  "mean_slope"),
    (aspect_tif, "mean_aspect"),
    (tpi_tif,    "mean_tpi"),
]:
    m,s = run_zonal(grid_utm, path, col)
    grid_feats[col]          = m.values
    grid_feats[f"{col}_std"] = s.values
    print(f"  {col:28s}  valid:{int((~np.isnan(m)).sum()):4d}/"
          f"{len(grid_feats)}  mean:{float(np.nanmean(m)):.4f}")

if HAS_LANDSAT:
    lst_tif = os.path.join(ML_DIR,"_lst.tif")
    arr_to_tif(lst_arr, lst_src, lst_tif)
    m,s = run_zonal(grid_utm, lst_tif, "mean_lst")
    grid_feats["mean_lst"]     = m.values
    grid_feats["mean_lst_std"] = s.values
    grid_feats["uhi"]          = grid_feats["mean_lst"] - LST_MEAN
    print(f"  mean_lst  valid:{int((~np.isnan(m)).sum())}/{len(grid_feats)}")
else:
    grid_feats["mean_lst"] = LST_MEAN
    grid_feats["uhi"]      = 0.0

if HAS_MODIS:
    aod_tif = os.path.join(ML_DIR,"_aod.tif")
    arr_to_tif(aod_arr, aod_src, aod_tif)
    m,s = run_zonal(grid_utm, aod_tif, "mean_aod")
    grid_feats["mean_aod"]     = m.values
    grid_feats["mean_aod_std"] = s.values
    print(f"  mean_aod  valid:{int((~np.isnan(m)).sum())}/{len(grid_feats)}")
else:
    grid_feats["mean_aod"] = AOD_MEAN

print("\n  Vector features...")


def line_density(gdf, lines, col):
    cl = gpd.clip(lines, gdf).copy().reset_index(drop=True)
    if len(cl)==0: gdf[col]=0.0; return gdf
    cl["_sl_m"] = cl.geometry.length
    sj = gpd.sjoin(cl[["_sl_m","geometry"]], gdf[["_gid","geometry"]],
                   how="inner", predicate="intersects")
    ag_m     = sj.groupby("_gid")["_sl_m"].sum()
    area_km2 = gdf.geometry.area/1e6
    density  = (ag_m/1000.0)/area_km2
    gdf[col] = gdf["_gid"].map(density).fillna(0.0)
    return gdf


def pip_count(gdf, pts, col):
    sj = gpd.sjoin(gdf[["_gid","geometry"]], pts[["geometry"]],
                   how="inner", predicate="contains")
    c_ = sj.groupby("_gid").size()
    gdf[col] = gdf["_gid"].map(c_).fillna(0).astype(float)
    return gdf


def area_frac(gdf, poly, col):
    ix = gpd.overlay(gdf[["_gid","geometry"]], poly[["geometry"]],
                     how="intersection", keep_geom_type=False)
    if len(ix)==0: gdf[col]=0.0; return gdf
    ix["_a"]     = ix.geometry.area
    ag_m2        = ix.groupby("_gid")["_a"].sum()
    cell_area_m2 = gdf.geometry.area
    gdf[col] = (gdf["_gid"].map(ag_m2).fillna(0.0)/cell_area_m2).clip(0,1)
    return gdf


grid_feats = line_density(grid_feats, pri_roads, "road_density_km2")
grid_feats = line_density(grid_feats, sec_roads, "sec_road_density_km2")
grid_feats = line_density(grid_feats, rail,      "rail_density_km2")
grid_feats = pip_count(grid_feats,   signals,    "signal_count")
grid_feats = pip_count(grid_feats,   bus,        "bus_stop_count")
grid_feats = area_frac(grid_feats,   indust,     "pct_industrial")
grid_feats = area_frac(grid_feats,   resid,      "pct_residential")
grid_feats = area_frac(grid_feats,   water,      "pct_water")

print("  Proximity features...")
cen = grid_feats.geometry.centroid
iu  = indust.unary_union
wu  = water.unary_union
ru  = pri_roads.unary_union
rsu = sec_roads.unary_union

grid_feats["dist_to_industry_m"] = cen.apply(lambda p: iu.distance(p)  if iu  else 0.0)
grid_feats["dist_to_water_m"]    = cen.apply(lambda p: wu.distance(p)  if wu  else 0.0)
grid_feats["dist_to_road_m"]     = cen.apply(lambda p: ru.distance(p)  if ru  else 0.0)
grid_feats["dist_to_sec_road_m"] = cen.apply(lambda p: rsu.distance(p) if rsu else 0.0)

grid_feats["traffic_load_idx"] = (
    grid_feats["road_density_km2"]     * 3.0 +
    grid_feats["sec_road_density_km2"] * 1.5 +
    grid_feats["signal_count"]         * 0.5 +
    grid_feats["bus_stop_count"]       * 0.3
)

print("  Engineering composite features...")
gf = grid_feats
gf["lg_d_road"]  = np.log1p(gf["dist_to_road_m"])
gf["lg_d_ind"]   = np.log1p(gf["dist_to_industry_m"])
gf["lg_d_water"] = np.log1p(gf["dist_to_water_m"])
gf["inv_d_road"] = 1.0/(gf["dist_to_road_m"]+1.0)
gf["inv_d_ind"]  = 1.0/(gf["dist_to_industry_m"]+1.0)

gf["blt_frac"]     = (gf["pct_industrial"]+gf["pct_residential"]).clip(0,1)
gf["grn_frac"]     = (1-gf["blt_frac"]-gf["pct_water"]).clip(0,1)
gf["imperv_proxy"] = gf["blt_frac"]

gf["elev_anom"]  = gf["mean_elev"]-ELEV_MEAN
gf["elev_sq"]    = gf["mean_elev"]**2
gf["tpi_neg"]    = (-gf["mean_tpi"]).clip(0)
gf["tpi_pos"]    = gf["mean_tpi"].clip(0)
gf["rough_norm"] = gf["mean_elev_std"]/(gf["mean_elev_std"].max()+1e-6)

gf["rd_x_blt"]   = gf["road_density_km2"]*gf["blt_frac"]
gf["rd1_x_ind"]  = gf["road_density_km2"]*gf["pct_industrial"]
gf["ind_x_dinv"] = gf["pct_industrial"]*gf["inv_d_ind"]
gf["rd_x_dinv"]  = gf["road_density_km2"]*gf["inv_d_road"]
gf["tpi_x_grn"]  = gf["mean_tpi"]*gf["grn_frac"]

if HAS_LANDSAT:
    gf["uhi_x_rd"]  = gf["uhi"]*gf["road_density_km2"]
    gf["uhi_x_blt"] = gf["uhi"]*gf["blt_frac"]
    gf["lst_x_ind"] = gf["mean_lst"]*gf["pct_industrial"]

if HAS_MODIS:
    gf["aod_x_elev"] = gf["mean_aod"]*gf["elev_anom"]
    gf["aod_x_rd"]   = gf["mean_aod"]*gf["road_density_km2"]
    gf["aod_x_blt"]  = gf["mean_aod"]*gf["blt_frac"]

print("  Spatial lag features...")
grid_xy = np.column_stack([cen.x.values, cen.y.values])
tree    = cKDTree(grid_xy)

for r_km in [2,5]:
    r_m  = r_km*1000
    nbrs = tree.query_ball_point(grid_xy, r=r_m)
    gf[f"rd_lag_{r_km}km"]  = [
        float(gf["road_density_km2"].iloc[idx].mean())
        if len(idx)>0 else 0.0 for idx in nbrs]
    gf[f"blt_lag_{r_km}km"] = [
        float(gf["blt_frac"].iloc[idx].mean())
        if len(idx)>0 else 0.0 for idx in nbrs]
    if HAS_MODIS:
        gf[f"aod_lag_{r_km}km"] = [
            float(np.nanmean(gf["mean_aod"].iloc[idx].values))
            if len(idx)>0 else AOD_MEAN for idx in nbrs]

print(f"\n  Grid features shape: {gf.shape}")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 4: IDW TARGET                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
heading("STAGE 4: IDW PM2.5 TARGET")

grid_centroids = np.column_stack([cen.x.values, cen.y.values])


def idw_surface(target_xys, coords, vals, power=2):
    diff = target_xys[:,np.newaxis,:] - coords[np.newaxis,:,:]
    dist = np.sqrt((diff**2).sum(axis=2))
    dist = np.maximum(dist,1.0)
    w    = 1.0/dist**power
    return (w*vals[np.newaxis,:]).sum(axis=1)/w.sum(axis=1)


print("  Computing IDW surface (all 14 stations)...")
gf["pm25_idw"] = idw_surface(grid_centroids, st_coords, st_pm25)
print(f"  IDW range: {gf['pm25_idw'].min():.1f}–{gf['pm25_idw'].max():.1f} "
      f"mean={gf['pm25_idw'].mean():.1f} µg/m³")

print("  IDW spatial lag features...")
for r_km in [2,5]:
    r_m  = r_km*1000
    nbrs = tree.query_ball_point(grid_xy, r=r_m)
    gf[f"idw_lag_{r_km}km"] = [
        float(gf["pm25_idw"].iloc[idx].mean())
        if len(idx)>0 else gf["pm25_idw"].mean()
        for idx in nbrs]

print("  IDW uncertainty (distance to nearest station)...")
st_tree = cKDTree(st_coords)
d_near, _ = st_tree.query(grid_centroids, k=1)
gf["dist_nearest_station_m"] = d_near
gf["idw_confidence"] = np.exp(-d_near / 5000.0)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 5: FEATURE SETS & SPATIAL BLOCK CV                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
heading("STAGE 5: FEATURE SETS & SPATIAL BLOCK CV")

# ── Honest environmental features (no coordinates) ────────────────────────────
LUR_BASE_COLS = [
    "road_density_km2","sec_road_density_km2","rail_density_km2",
    "signal_count","bus_stop_count","traffic_load_idx",
    "pct_industrial","pct_residential","pct_water",
    "dist_to_industry_m","dist_to_water_m","dist_to_road_m",
    "lg_d_road","lg_d_ind","lg_d_water","inv_d_road","inv_d_ind",
    "blt_frac","grn_frac","imperv_proxy",
    "rd_x_blt","rd1_x_ind","ind_x_dinv","rd_x_dinv",
    "mean_slope","mean_aspect","mean_tpi","mean_elev",
    "elev_anom","elev_sq","tpi_neg","tpi_pos","rough_norm","tpi_x_grn",
    "rd_lag_2km","rd_lag_5km","blt_lag_2km","blt_lag_5km",
]
ENHANCED_EXTRA = []
if HAS_MODIS:
    ENHANCED_EXTRA += ["mean_aod","mean_aod_std",
                       "aod_x_elev","aod_x_rd","aod_x_blt",
                       "aod_lag_2km","aod_lag_5km"]
if HAS_LANDSAT:
    ENHANCED_EXTRA += ["mean_lst","mean_lst_std",
                       "uhi","uhi_x_rd","uhi_x_blt","lst_x_ind"]

IDW_LAG_COLS   = ["idw_lag_2km","idw_lag_5km"]
ENHANCED_COLS  = LUR_BASE_COLS + ENHANCED_EXTRA
ALL_COLS       = ENHANCED_COLS + IDW_LAG_COLS

LUR_BASE_COLS  = [c for c in LUR_BASE_COLS  if c in gf.columns]
ENHANCED_EXTRA = [c for c in ENHANCED_EXTRA if c in gf.columns]
ENHANCED_COLS  = [c for c in ENHANCED_COLS  if c in gf.columns]
IDW_LAG_COLS   = [c for c in IDW_LAG_COLS   if c in gf.columns]
ALL_COLS       = [c for c in ALL_COLS        if c in gf.columns]

RS_SET  = set(ENHANCED_EXTRA)
IDW_SET = set(IDW_LAG_COLS)


def feat_color(f):
    if f in IDW_SET: return "#FF5722"
    if f in RS_SET:  return "#4CAF50"
    return "#1565C0"


TARGET_COL = "pm25_idw"
df_ml  = gf[ALL_COLS+[TARGET_COL,"_gid"]].copy().dropna(subset=[TARGET_COL])
y      = df_ml[TARGET_COL].values.astype(np.float64)

imp_enh = SimpleImputer(strategy="median")
imp_all = SimpleImputer(strategy="median")
X_enh   = imp_enh.fit_transform(df_ml[ENHANCED_COLS].values.astype(np.float64))
X_all   = imp_all.fit_transform(df_ml[ALL_COLS].values.astype(np.float64))

X_enh_df = pd.DataFrame(X_enh, columns=ENHANCED_COLS)
X_all_df = pd.DataFrame(X_all, columns=ALL_COLS)

print(f"  Enhanced (honest) : {len(ENHANCED_COLS)} features")
print(f"  All (+IDW lag)    : {len(ALL_COLS)} features")
print(f"  Samples           : {len(y)}")
print(f"  Target range      : {y.min():.1f}–{y.max():.1f} "
      f"mean={y.mean():.1f} µg/m³")

# ── Spatial block CV (4 quadrants) ────────────────────────────────────────────
print("\n  Building spatial block CV folds (4 geographic quadrants)...")
cx  = cen.x.values[df_ml.index]
cy  = cen.y.values[df_ml.index]
mx  = np.median(cx); my = np.median(cy)

spatial_fold = np.zeros(len(df_ml), dtype=int)
spatial_fold[(cx >= mx) & (cy >= my)] = 0   # NE
spatial_fold[(cx <  mx) & (cy >= my)] = 1   # NW
spatial_fold[(cx >= mx) & (cy <  my)] = 2   # SE
spatial_fold[(cx <  mx) & (cy <  my)] = 3   # SW

# Custom CV splitter from spatial folds
class SpatialBlockCV:
    def __init__(self, fold_labels):
        self.fold_labels = fold_labels
        self.n_splits    = len(np.unique(fold_labels))

    def split(self, X, y=None, groups=None):
        for fold_id in np.unique(self.fold_labels):
            te = np.where(self.fold_labels == fold_id)[0]
            tr = np.where(self.fold_labels != fold_id)[0]
            yield tr, te

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

spatial_cv = SpatialBlockCV(spatial_fold)

for fid in range(4):
    n_ = (spatial_fold==fid).sum()
    quad = ["NE","NW","SE","SW"][fid]
    print(f"    Quadrant {quad}: {n_} cells ({n_/len(df_ml)*100:.1f}%)")

# Standard 5-fold for comparison
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# RF tuning
print("\n  [5.0] Tuning RF on enhanced features...")
rf_param_dist = {
    "n_estimators":          randint(200,600),
    "max_depth":             randint(4,10),
    "min_samples_leaf":      randint(3,20),
    "max_features":          uniform(0.3,0.5),
    "min_impurity_decrease": uniform(0,0.0005),
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    rf_param_dist, n_iter=30, cv=kf, scoring="r2",
    random_state=RANDOM_STATE, n_jobs=-1)
rf_search.fit(X_enh, y)
best_rf_params = rf_search.best_params_
print(f"  Best RF CV R²: {rf_search.best_score_:.4f}")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 6: MODEL CATALOGUE & EVALUATION                                     ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
heading("STAGE 6: MODEL TRAINING & EVALUATION")

print("""
  Training target : IDW PM2.5 surface (anchored to 14 real CPCB measurements)
  Predictors      : Environmental features only (roads, land use, RS, terrain)
  CV strategy     : Spatial block CV (4 geographic quadrants)
                    Tests generalization to unmeasured parts of the city
  Also reported   : Standard 5-fold CV for comparison
""")

catalogue_raw = [
    ("Ridge_LUR",
     Pipeline([("sc",RobustScaler()),("m",Ridge(alpha=10.0))]),
     X_enh, ENHANCED_COLS, "Linear"),

    ("ElasticNet_LUR",
     Pipeline([("sc",RobustScaler()),
               ("m",ElasticNet(alpha=0.01,l1_ratio=0.5,max_iter=5000))]),
     X_enh, ENHANCED_COLS, "Linear"),

    ("RF_tuned",
     Pipeline([("sc",RobustScaler()),
               ("m",RandomForestRegressor(
                   **best_rf_params,
                   random_state=RANDOM_STATE, n_jobs=-1))]),
     X_enh, ENHANCED_COLS, "Ensemble"),

    ("RF_base",
     Pipeline([("sc",RobustScaler()),
               ("m",RandomForestRegressor(
                   n_estimators=400, max_depth=6, min_samples_leaf=5,
                   max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1))]),
     X_enh, ENHANCED_COLS, "Ensemble"),

    ("GB_ENH",
     Pipeline([("sc",RobustScaler()),
               ("m",GradientBoostingRegressor(
                   n_estimators=400, learning_rate=0.03, max_depth=3,
                   subsample=0.7, min_samples_leaf=10,
                   random_state=RANDOM_STATE))]),
     X_enh, ENHANCED_COLS, "Ensemble"),

    ("ExtraT_ENH",
     Pipeline([("sc",RobustScaler()),
               ("m",ExtraTreesRegressor(
                   n_estimators=400, max_depth=6, min_samples_leaf=5,
                   max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1))]),
     X_enh, ENHANCED_COLS, "Ensemble"),

    # IDW-lag augmented models (clearly labeled)
    ("RF_IDWlag",
     Pipeline([("sc",RobustScaler()),
               ("m",RandomForestRegressor(
                   n_estimators=300, max_depth=8, min_samples_leaf=3,
                   max_features="sqrt", n_jobs=-1,
                   random_state=RANDOM_STATE))]),
     X_all, ALL_COLS, "Ensemble+IDWlag"),

    ("GB_IDWlag",
     Pipeline([("sc",RobustScaler()),
               ("m",GradientBoostingRegressor(
                   n_estimators=300, learning_rate=0.05, max_depth=4,
                   subsample=0.8, min_samples_leaf=3,
                   random_state=RANDOM_STATE))]),
     X_all, ALL_COLS, "Ensemble+IDWlag"),
]

if HAS_LGB:
    catalogue_raw += [
        ("LGB_ENH",
         Pipeline([("sc",RobustScaler()),
                   ("m",LGBMFeatureNameWrapper(
                       n_estimators=500, learning_rate=0.03, num_leaves=20,
                       max_depth=5, min_child_samples=15, subsample=0.8,
                       colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                       random_state=RANDOM_STATE, verbose=-1))]),
         X_enh_df, ENHANCED_COLS, "Ensemble"),
    ]

if HAS_XGB:
    catalogue_raw += [
        ("XGB_ENH",
         Pipeline([("sc",RobustScaler()),
                   ("m",XGBRegressor(
                       n_estimators=400, max_depth=3, learning_rate=0.03,
                       min_child_weight=10, subsample=0.8,
                       colsample_bytree=0.7, reg_alpha=1, reg_lambda=2,
                       random_state=RANDOM_STATE, verbosity=0, n_jobs=-1))]),
         X_enh, ENHANCED_COLS, "Ensemble"),
    ]

model_catalogue = [(n,p,X,f,l) for n,p,X,f,l in catalogue_raw if p is not None]

MODEL_COLORS = {
    "Ridge_LUR":   "#1565C0", "ElasticNet_LUR": "#1E88E5",
    "RF_tuned":    "#43A047", "RF_base":        "#2E7D32",
    "GB_ENH":      "#66BB6A", "ExtraT_ENH":     "#00838F",
    "LGB_ENH":     "#00ACC1", "XGB_ENH":        "#0097A7",
    "RF_IDWlag":   "#E53935", "GB_IDWlag":      "#EF5350",
}

scoring = {
    "r2":   "r2",
    "rmse": "neg_root_mean_squared_error",
    "mae":  "neg_mean_absolute_error",
}

metrics_rows  = []
fitted_models = {}
oof_preds_kf  = {}
oof_preds_sp  = {}

print(f"\n  {'Model':<16} {'Cat':<14} "
      f"{'Spat-R²':>8} {'KF-R²':>7} {'KF±':>6} "
      f"{'KF-RMSE':>8}  Note")
print("  "+"─"*75)

for mname,pipe,X_use,feat_names,fset_label in model_catalogue:
    try:
        has_idw = "IDWlag" in fset_label

        # Spatial block CV
        sp_r2s = []
        for tr_idx,te_idx in spatial_cv.split(X_use):
            X_tr = (X_use.iloc[tr_idx]
                    if isinstance(X_use,pd.DataFrame) else X_use[tr_idx])
            X_te = (X_use.iloc[te_idx]
                    if isinstance(X_use,pd.DataFrame) else X_use[te_idx])
            pipe.fit(X_tr, y[tr_idx])
            yp   = pipe.predict(X_te)
            sp_r2s.append(r2_score(y[te_idx], yp))
        sp_r2 = float(np.mean(sp_r2s))

        # Standard 5-fold CV
        cv_res = cross_validate(pipe, X_use, y, cv=kf,
                                scoring=scoring, return_train_score=True)
        r2_val = cv_res["test_r2"].mean()
        r2_std = cv_res["test_r2"].std()
        rmse_v = -cv_res["test_rmse"].mean()
        mae_v  = -cv_res["test_mae"].mean()
        r2_tr  = cv_res["train_r2"].mean()
        gap    = r2_tr - r2_val

        metrics_rows.append({
            "model":       mname,
            "category":    fset_label,
            "spatial_r2":  sp_r2,
            "kf_r2":       r2_val,
            "kf_r2_std":   r2_std,
            "kf_rmse":     rmse_v,
            "kf_mae":      mae_v,
            "train_r2":    r2_tr,
            "gap":         gap,
            "has_idw_lag": has_idw,
        })

        oof_kf = cross_val_predict(pipe, X_use, y, cv=kf)
        oof_sp = cross_val_predict(pipe, X_use, y, cv=spatial_cv)
        oof_preds_kf[mname] = oof_kf
        oof_preds_sp[mname] = oof_sp

        pipe.fit(X_use, y)
        fitted_models[mname] = (pipe, X_use, feat_names)

        note = " [+IDW]" if has_idw else ""
        sym  = "★" if sp_r2>0.6 else ("✓" if sp_r2>0.3 else "○")
        print(f"  {sym} {mname:<16} {fset_label:<14} "
              f"{sp_r2:>8.4f} {r2_val:>7.4f} {r2_std:>6.4f} "
              f"{rmse_v:>8.4f}{note}")

    except Exception as e:
        print(f"  [ERROR] {mname}: {e}")
        import traceback; traceback.print_exc()

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(os.path.join(ML_DIR,"Model_Comparison.csv"), index=False)

# Best honest model by spatial R²
honest_df = metrics_df[~metrics_df["has_idw_lag"]]
best_lur  = honest_df.loc[honest_df["spatial_r2"].idxmax()]
best_name = best_lur["model"]
best_pipe, best_X, best_feat_names = fitted_models[best_name]

print(f"\n  ★ Best honest model (spatial CV) : {best_name}")
print(f"    Spatial R²={best_lur['spatial_r2']:.4f}  "
      f"KF R²={best_lur['kf_r2']:.4f}  "
      f"RMSE={best_lur['kf_rmse']:.4f}")
print(f"  NOTE: Spatial R² tests generalization to unmeasured city quadrants.")
print(f"        KF R² measures IDW surface reproduction.")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 7: FULL GRID PREDICTIONS                                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
heading("STAGE 7: FULL GRID PREDICTIONS")

def get_full_X(mname):
    _,_,feat_names = fitted_models[mname]
    if set(feat_names) == set(ENHANCED_COLS):
        arr = imp_enh.transform(gf[ENHANCED_COLS].values.astype(np.float64))
        return (pd.DataFrame(arr, columns=ENHANCED_COLS)
                if mname=="LGB_ENH" else arr)
    else:
        return imp_all.transform(gf[ALL_COLS].values.astype(np.float64))


X_full_best      = get_full_X(best_name)
y_full           = best_pipe.predict(X_full_best)
gf["pm25_proxy"] = y_full
print(f"  {best_name}: {y_full.min():.1f}–{y_full.max():.1f} "
      f"mean={y_full.mean():.1f} µg/m³")

# ── Stacking (honest models only) ─────────────────────────────────────────────
lur_names = [n for n,_,_,_,l in model_catalogue
             if not("IDWlag" in l) and n in fitted_models]
HAS_STACK = False; meta=None; y_full_stack=None; r2_stack=rmse_stack=np.nan

if len(lur_names)>=2:
    oof_mat  = np.column_stack([oof_preds_kf[n] for n in lur_names])
    meta     = Ridge(alpha=1.0)
    meta.fit(oof_mat, y)
    base_tr  = np.column_stack([
        fitted_models[n][0].predict(fitted_models[n][1]) for n in lur_names])
    y_stack  = meta.predict(base_tr)
    r2_stack = r2_score(y, y_stack)
    rmse_stack = np.sqrt(mean_squared_error(y, y_stack))
    base_full= np.column_stack([
        fitted_models[n][0].predict(get_full_X(n)) for n in lur_names])
    y_full_stack     = meta.predict(base_full)
    gf["pm25_stack"] = y_full_stack
    HAS_STACK = True
    print(f"  Stack: {y_full_stack.min():.1f}–{y_full_stack.max():.1f} "
          f"mean={y_full_stack.mean():.1f}  KF R²={r2_stack:.4f}")

# ── Permutation importance ────────────────────────────────────────────────────
print(f"\n  Permutation importance ({best_name})...")
tr_idx,te_idx = next(kf.split(best_X))
bX_tr = (best_X.iloc[tr_idx] if isinstance(best_X,pd.DataFrame)
         else best_X[tr_idx])
bX_te = (best_X.iloc[te_idx] if isinstance(best_X,pd.DataFrame)
         else best_X[te_idx])
best_pipe.fit(bX_tr, y[tr_idx])
perm = permutation_importance(best_pipe, bX_te, y[te_idx],
                              n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1)
perm_df = pd.DataFrame({
    "feature":    best_feat_names,
    "importance": perm.importances_mean,
    "std":        perm.importances_std,
}).sort_values("importance",ascending=False).reset_index(drop=True)
perm_df.to_csv(os.path.join(ML_DIR,"Perm_Importance.csv"), index=False)
print(perm_df.head(10).to_string(index=False))
best_pipe.fit(best_X, y)

mdi_df = None
inner  = best_pipe.named_steps.get("m")
if hasattr(inner,"feature_importances_"):
    mdi_df = pd.DataFrame({
        "feature":    best_feat_names,
        "importance": inner.feature_importances_,
    }).sort_values("importance",ascending=False).reset_index(drop=True)

# ── CPCB external validation ──────────────────────────────────────────────────
print("\n  CPCB external validation...")
cpcb_val_df = None
HAS_BC = False; bc=None; y_full_bc=None
r_p=p_v=r2_cpcb=rmse_cpcb=np.nan

try:
    grid_cent = gf[["_gid","geometry","pm25_proxy"]].copy()
    grid_cent["geometry"] = grid_cent.geometry.centroid
    cpcb_j = gpd.sjoin_nearest(stations_utm, grid_cent,
                               how="left", distance_col="dist_m")
    cpcb_val_df = cpcb_j[["station_name","pm25_mean",
                           "pm25_proxy","dist_m"]].dropna()
    if len(cpcb_val_df)>1:
        r2_cpcb   = r2_score(cpcb_val_df["pm25_mean"],
                             cpcb_val_df["pm25_proxy"])
        r_p,p_v   = pearsonr(cpcb_val_df["pm25_mean"],
                             cpcb_val_df["pm25_proxy"])
        rmse_cpcb = np.sqrt(mean_squared_error(
            cpcb_val_df["pm25_mean"],cpcb_val_df["pm25_proxy"]))
        print(f"    r={r_p:.4f}  R²={r2_cpcb:.4f}  RMSE={rmse_cpcb:.2f} µg/m³")
        print(f"    Note: This is the naive CPCB comparison (same stations "
              f"used in IDW target construction).")
        if len(cpcb_val_df)>=3:
            bc = LinearRegression()
            bc.fit(cpcb_val_df[["pm25_proxy"]], cpcb_val_df["pm25_mean"])
            y_full_bc     = bc.predict(gf[["pm25_proxy"]].values)
            gf["pm25_bc"] = y_full_bc
            HAS_BC = True
            print(f"    BC slope={bc.coef_[0]:.4f}  "
                  f"intercept={bc.intercept_:.4f}")
            print(f"    BC range: {y_full_bc.min():.1f}–"
                  f"{y_full_bc.max():.1f} µg/m³")
except Exception as e:
    print(f"  [WARN] {e}")

# ── Learning curves ────────────────────────────────────────────────────────────
print(f"\n  Learning curves ({best_name})...")
lc_sz,lc_tr_sc,lc_va_sc = learning_curve(
    best_pipe, best_X, y,
    train_sizes=np.linspace(0.1,1.0,10),
    cv=kf, scoring="r2", n_jobs=-1,
    shuffle=True, random_state=RANDOM_STATE)

# ── Grid enrichment ────────────────────────────────────────────────────────────
bins       = [0,30,60,90,120,250]
aqi_labels = ["Good","Satisfactory","Moderate","Poor","Very Poor"]
AQI_COLORS = {
    "Good":         "#00b050",
    "Satisfactory": "#92d050",
    "Moderate":     "#ffff00",
    "Poor":         "#ff7e00",
    "Very Poor":    "#ff0000",
}
grid2 = grid.copy()
grid2["pm25_proxy"] = y_full
grid2["pm25_idw"]   = gf["pm25_idw"].values
grid2["aqi"]        = pd.cut(grid2["pm25_proxy"],bins=bins,labels=aqi_labels)
if HAS_BC:    grid2["pm25_bc"]    = y_full_bc
if HAS_STACK: grid2["pm25_stack"] = y_full_stack
grid2["idw_confidence"] = gf["idw_confidence"].values

grid2.drop(columns=["geometry"]).to_csv(
    os.path.join(ML_DIR,"PM25_Grid_Predictions.csv"),
    index=False, encoding="utf-8-sig")
with open(os.path.join(ML_DIR,"final_model.pkl"),"wb") as fh:
    pickle.dump(best_pipe, fh)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 8: ALL 17 FIGURES                                                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
heading("STAGE 8: FIGURES")

pm_cmap = LinearSegmentedColormap.from_list(
    "pm25",["#00e400","#92d050","#ffff00","#ff7e00","#ff0000","#8f3f97"])
bar_colors = [MODEL_COLORS.get(m,"#9E9E9E") for m in metrics_df["model"]]
leg3 = [
    Patch(color="#1565C0",label="LUR / Terrain"),
    Patch(color="#4CAF50",label="Remote Sensing (AOD / LST)"),
    Patch(color="#FF5722",label="IDW spatial lag"),
]

# ── Fig 01: Model Comparison ──────────────────────────────────────────────────
print("\n  Fig 01: Model Comparison...")
fig,axes = plt.subplots(1,3,figsize=(16,5))
for ax,(metric,lbl,fmt) in zip(axes,[
    ("spatial_r2","Spatial Block R²",".4f"),
    ("kf_r2",    "5-Fold KF R²",    ".4f"),
    ("kf_rmse",  "KF RMSE",         ".3f"),
]):
    bars = ax.bar(metrics_df["model"],metrics_df[metric],
                  color=bar_colors,edgecolor="white",width=0.6)
    if metric=="kf_r2":
        ax.errorbar(range(len(metrics_df)),metrics_df[metric],
                    yerr=metrics_df["kf_r2_std"],
                    fmt="none",c="k",capsize=3)
    ax.set_title(lbl,fontsize=12); ax.set_ylabel(lbl)
    ax.tick_params(axis="x",rotation=40)
    for bar,v in zip(bars,metrics_df[metric]):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()*1.01,f"{v:{fmt}}",
                ha="center",va="bottom",fontsize=6)
    ax.grid(axis="y",alpha=0.3)
leg01 = [Patch(color="#2E7D32",label="Honest (env. features only)"),
         Patch(color="#E53935",label="IDW-lag augmented")]
axes[0].legend(handles=leg01,fontsize=8)
plt.suptitle(
    f"Model Comparison — Hybrid LUR v9.5 (500m)\n"
    f"Best honest (spatial CV): {best_name}  "
    f"Spatial R²={best_lur['spatial_r2']:.4f}  "
    f"KF R²={best_lur['kf_r2']:.4f}",
    fontsize=9)
plt.tight_layout()
savefig(fig,"01_Model_Comparison.png")

# ── Fig 02: Spatial vs KF R² ─────────────────────────────────────────────────
print("  Fig 02: Spatial vs KF R²...")
fig,ax = plt.subplots(figsize=(9,5))
x = np.arange(len(metrics_df)); w=0.35
ax.bar(x-w/2,metrics_df["spatial_r2"],w,
       label="Spatial Block R²",color="#E91E63",alpha=0.85)
ax.bar(x+w/2,metrics_df["kf_r2"],    w,
       label="5-Fold KF R²",   color="#2196F3",alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metrics_df["model"],rotation=30,ha="right",fontsize=8)
ax.set_ylabel("R²"); ax.legend(); ax.grid(axis="y",alpha=0.4)
ax.set_title("Spatial Block CV vs Standard KF CV\n"
             "Spatial R² = honest generalization to unmeasured city areas",
             fontsize=10)
for i,(sr,kr) in enumerate(zip(metrics_df["spatial_r2"],
                                metrics_df["kf_r2"])):
    gap=kr-sr
    ax.text(i,max(sr,kr)+0.01,f"Δ{gap:.2f}",
            ha="center",fontsize=6,
            color="red" if gap>0.15 else "green",fontweight="bold")
plt.tight_layout()
savefig(fig,"02_Spatial_vs_KF_R2.png")

# ── Fig 03: Permutation Importance ────────────────────────────────────────────
print("  Fig 03: Permutation Importance...")
fig,ax = plt.subplots(figsize=(10,max(6,len(perm_df)*0.35)))
pi_c = [feat_color(f) for f in perm_df["feature"]]
ax.barh(perm_df["feature"][::-1],perm_df["importance"][::-1],
        xerr=perm_df["std"][::-1],color=pi_c[::-1],capsize=3,edgecolor="white")
ax.set_xlabel("Mean ΔR²",fontsize=11)
ax.axvline(0,color="k",lw=0.8)
ax.set_title(f"Permutation Feature Importance — {best_name}",fontsize=12)
ax.grid(axis="x",alpha=0.3); ax.legend(handles=leg3,fontsize=8)
plt.tight_layout()
savefig(fig,"03_Permutation_Importance.png")

# ── Fig 04: MDI Importance ────────────────────────────────────────────────────
if mdi_df is not None:
    print("  Fig 04: MDI Importance...")
    fig,ax = plt.subplots(figsize=(10,max(6,len(mdi_df)*0.35)))
    c2 = cm.viridis(np.linspace(0.15,0.85,len(mdi_df)))
    ax.barh(mdi_df["feature"][::-1],mdi_df["importance"][::-1],
            color=c2[::-1],edgecolor="white")
    ax.set_xlabel("MDI Importance",fontsize=11)
    ax.set_title(f"MDI Feature Importance — {best_name}",fontsize=12)
    ax.grid(axis="x",alpha=0.3)
    plt.tight_layout()
    savefig(fig,"04_MDI_Importance.png")

# ── Fig 05: Feature Importance Summary ────────────────────────────────────────
if mdi_df is not None:
    print("  Fig 05: Feature Importance Summary...")
    top_n    = min(15,len(perm_df))
    perm_top = perm_df.head(top_n)
    mdi_top  = mdi_df.head(top_n)
    fig,axes = plt.subplots(1,2,figsize=(15,6))
    for ax,(df_i,title,kw) in zip(axes,[
        (perm_top,f"Permutation — {best_name}",
         {"xerr":perm_top["std"].values[::-1],"capsize":3}),
        (mdi_top, f"MDI — {best_name}",{}),
    ]):
        c_ = [feat_color(f) for f in df_i["feature"]]
        ax.barh(df_i["feature"].values[::-1],
                df_i["importance"].values[::-1],
                color=c_[::-1],edgecolor="white",**kw)
        ax.set_title(title,fontsize=11)
        ax.grid(axis="x",alpha=0.3)
        ax.legend(handles=leg3,fontsize=8)
    plt.suptitle("Feature Importance: Permutation vs MDI",fontsize=12)
    plt.tight_layout()
    savefig(fig,"05_Feature_Importance_Summary.png")

# ── Fig 06: Learning Curves ───────────────────────────────────────────────────
print("  Fig 06: Learning Curves...")
fig,ax = plt.subplots(figsize=(9,5))
tm=lc_tr_sc.mean(1); ts=lc_tr_sc.std(1)
vm=lc_va_sc.mean(1); vs=lc_va_sc.std(1)
ax.plot(lc_sz,tm,"o-",color="#4CAF50",label="Train R²",lw=2)
ax.fill_between(lc_sz,tm-ts,tm+ts,alpha=0.2,color="#4CAF50")
ax.plot(lc_sz,vm,"o-",color="#2196F3",label="CV Val R²",lw=2)
ax.fill_between(lc_sz,vm-vs,vm+vs,alpha=0.2,color="#2196F3")
ax.set_xlabel("Training samples",fontsize=11)
ax.set_ylabel("R²",fontsize=11)
ax.set_title(f"Learning Curve — {best_name}",fontsize=12)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
savefig(fig,"06_Learning_Curve.png")

# ── Fig 07: Predicted vs Observed ─────────────────────────────────────────────
print("  Fig 07: Predicted vs Observed...")
pairs = [(best_name,best_X,f"Best honest: {best_name} (Spatial CV)")]
idw_key = next((k for k in ["RF_IDWlag","GB_IDWlag"] if k in fitted_models),None)
if idw_key:
    pairs.append((idw_key,fitted_models[idw_key][1],
                  f"{idw_key} [+IDW lag]"))
fig,axes = plt.subplots(1,len(pairs),figsize=(7*len(pairs),5))
if len(pairs)==1: axes=[axes]
for ax,(mn,Xu,lbl) in zip(axes,pairs):
    yp  = fitted_models[mn][0].predict(Xu)
    r2_ = r2_score(y,yp); rm_=np.sqrt(mean_squared_error(y,yp))
    ax.scatter(y,yp,alpha=0.3,s=10,c="#2196F3",edgecolors="none")
    lims=[min(y.min(),yp.min())*0.97,max(y.max(),yp.max())*1.03]
    ax.plot(lims,lims,"r--",lw=1.5,label="1:1")
    m_,b_=np.polyfit(y,yp,1); xs=np.linspace(lims[0],lims[1],100)
    ax.plot(xs,m_*xs+b_,"g-",lw=1.5,alpha=0.8,
            label=f"fit: {m_:.2f}x+{b_:.1f}")
    ax.set_xlabel("IDW Target PM2.5 (µg/m³)",fontsize=9)
    ax.set_ylabel("Predicted (µg/m³)",fontsize=9)
    ax.set_title(f"{lbl}\nR²={r2_:.4f}  RMSE={rm_:.4f}",fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.suptitle("Predicted vs IDW Target — Hybrid LUR v9.5",fontsize=10)
plt.tight_layout()
savefig(fig,"07_Predicted_vs_Observed.png")

# ── Fig 08: Residual Analysis ─────────────────────────────────────────────────
print("  Fig 08: Residual Analysis...")
y_pred_best = best_pipe.predict(best_X)
resid = y_pred_best - y
fig,axes = plt.subplots(1,3,figsize=(15,4))
axes[0].hist(resid,bins=35,color="#FF7043",edgecolor="white",alpha=0.9)
axes[0].axvline(0,color="k",lw=1)
axes[0].set_xlabel("Residual",fontsize=10)
axes[0].set_title("Residual Distribution")
axes[0].text(0.98,0.95,f"μ={resid.mean():.3f}\nσ={resid.std():.3f}",
             transform=axes[0].transAxes,ha="right",va="top",fontsize=9)
axes[1].scatter(y_pred_best,resid,alpha=0.3,s=8,c="#FF7043",edgecolors="none")
axes[1].axhline(0,color="k",lw=1)
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals vs Predicted")
sp_stats.probplot(resid,plot=axes[2]); axes[2].set_title("Q-Q Plot")
for a in axes: a.grid(alpha=0.3)
plt.suptitle(f"Residual Analysis — {best_name}",fontsize=11)
plt.tight_layout()
savefig(fig,"08_Residual_Analysis.png")

# ── Fig 09: Correlation Heatmap ───────────────────────────────────────────────
print("  Fig 09: Correlation Heatmap...")
plot_cols = [c for c in ENHANCED_COLS[:16]+["pm25_idw"] if c in df_ml.columns]
corr_mat  = df_ml[plot_cols].corr()
mask      = np.triu(np.ones_like(corr_mat,dtype=bool))
fig,ax    = plt.subplots(figsize=(14,12))
sns.heatmap(corr_mat,mask=mask,annot=True,fmt=".2f",cmap="coolwarm",
            center=0,square=True,linewidths=0.4,annot_kws={"size":6},ax=ax)
ax.set_title("Predictor Correlation Matrix",fontsize=12)
plt.tight_layout()
savefig(fig,"09_Correlation_Heatmap.png")

# ── Fig 10: Spatial Maps ──────────────────────────────────────────────────────
print("  Fig 10: Spatial Maps...")
panel_defs = [
    ("pm25_idw",   "IDW PM2.5 Target (µg/m³)",         "YlOrRd"),
    ("pm25_proxy", f"Predicted — {best_name} (µg/m³)", "RdYlGn_r"),
]
if HAS_STACK and "pm25_stack" in grid2.columns:
    panel_defs.append(("pm25_stack","Stacking Ensemble (µg/m³)","RdPu"))
if HAS_BC and "pm25_bc" in grid2.columns:
    panel_defs.append(("pm25_bc","Bias-Corrected PM2.5 (µg/m³)","hot_r"))
n_p   = len(panel_defs)
fig,axes = plt.subplots(1,n_p,figsize=(7*n_p,7))
if n_p==1: axes=[axes]
for ax,(col,title,cmap) in zip(axes,panel_defs):
    if col in grid2.columns:
        grid2.plot(column=col,cmap=cmap,legend=True,
                   legend_kwds={"shrink":0.6,"label":"µg/m³"},
                   ax=ax,edgecolor="none",
                   missing_kwds={"color":"lightgrey"})
        boundary_utm.boundary.plot(ax=ax,color="black",lw=1.2,ls="--")
        stations_utm.plot(ax=ax,color="blue",markersize=50,
                          zorder=5,marker="^",label="CPCB station")
        ax.set_title(title,fontsize=9)
        ax.legend(loc="lower right",fontsize=7)
        ax.set_axis_off()
plt.suptitle("Hyderabad PM2.5 — Hybrid LUR v9.5 (500m)",fontsize=12)
plt.tight_layout()
savefig(fig,"10_Spatial_Maps.png")

# ── Fig 11: Distribution Comparison ──────────────────────────────────────────
print("  Fig 11: Distribution Comparison...")
fig,ax = plt.subplots(figsize=(9,4))
ax.hist(y,     bins=40,alpha=0.6,color="#2196F3",
        label="Observed (IDW target)",edgecolor="white")
ax.hist(y_full,bins=40,alpha=0.6,color="#FF5722",
        label=f"Predicted ({best_name})",edgecolor="white")
if HAS_STACK and y_full_stack is not None:
    ax.hist(y_full_stack,bins=40,alpha=0.5,color="#4CAF50",
            label="Stacking",edgecolor="white",histtype="step",lw=2)
if HAS_BC and y_full_bc is not None:
    ax.hist(y_full_bc,bins=40,alpha=0.5,color="#9C27B0",
            label="Bias-Corrected",edgecolor="white",histtype="step",lw=2)
ax.axvline(y.mean(),color="#2196F3",ls="--",lw=1.5,
           label=f"Obs μ={y.mean():.1f}")
ax.axvline(y_full.mean(),color="#FF5722",ls="--",lw=1.5,
           label=f"Pred μ={y_full.mean():.1f}")
ax.set_xlabel("PM2.5 (µg/m³)",fontsize=11)
ax.set_ylabel("Count",fontsize=11)
ax.set_title("Distribution: IDW Target vs Predicted")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
savefig(fig,"11_Distribution_Comparison.png")

# ── Fig 12: Radar Chart ───────────────────────────────────────────────────────
# ── Fig 12: Radar Chart (fixed scaling) ───────────────────────────────────────
print("  Fig 12: Radar Chart...")
honest_sub = metrics_df[~metrics_df["has_idw_lag"]].reset_index(drop=True)

if len(honest_sub) >= 2:

    # ── Step 1: collect raw metric values ─────────────────────────────────────
    # We use 4 axes. For each we define what "good" means so all axes
    # point outward = better.
    #
    #  Axis 0 — Spatial R²   : raw value (can be negative)
    #  Axis 1 — KF R²        : raw value
    #  Axis 2 — 1 - RMSE×0.2 : penalises large RMSE gently
    #  Axis 3 — 1 - Gap      : penalises overfitting

    raw_metrics = []
    for _, row in honest_sub.iterrows():
        raw_metrics.append([
            row["spatial_r2"],
            row["kf_r2"],
            1.0 - row["kf_rmse"] * 0.2,   # gentler RMSE penalty
            1.0 - min(abs(row["gap"]), 1.0),
        ])
    raw_arr = np.array(raw_metrics)          # shape (n_models, 4)

    # ── Step 2: min-max normalise EACH axis to [0.05, 1.0] ───────────────────
    # This guarantees every axis has visible width even when all models
    # cluster at similar values or when spatial R² is uniformly negative.
    norm_arr = np.zeros_like(raw_arr)
    for col_idx in range(raw_arr.shape[1]):
        col    = raw_arr[:, col_idx]
        c_min  = col.min()
        c_max  = col.max()
        spread = c_max - c_min

        if spread < 1e-6:
            # All models identical on this axis → put everyone at 0.5
            norm_arr[:, col_idx] = 0.5
        else:
            # Scale to [0.05, 1.0] so even the worst model is visible
            norm_arr[:, col_idx] = 0.05 + 0.95 * (col - c_min) / spread

    # ── Step 3: axis labels with actual value ranges ───────────────────────────
    cats = [
        f"Spatial R²\n[{raw_arr[:,0].min():.2f} → {raw_arr[:,0].max():.2f}]",
        f"KF R²\n[{raw_arr[:,1].min():.2f} → {raw_arr[:,1].max():.2f}]",
        f"1-RMSE×0.2\n[{raw_arr[:,2].min():.2f} → {raw_arr[:,2].max():.2f}]",
        f"1-Gap\n[{raw_arr[:,3].min():.2f} → {raw_arr[:,3].max():.2f}]",
    ]
    N      = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]                     # close the polygon

    # ── Step 4: draw ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9),
                           subplot_kw={"polar": True})
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    palette = plt.cm.tab10(np.linspace(0, 1, len(honest_sub)))

    for i, (_, row) in enumerate(honest_sub.iterrows()):
        vals = norm_arr[i].tolist()
        vals += vals[:1]                     # close polygon
        raw  = raw_arr[i]

        ax.plot(angles, vals, lw=2,
                label=row["model"], color=palette[i])
        ax.fill(angles, vals, alpha=0.12, color=palette[i])

        # Annotate each vertex with the raw value
        for ang, nv, rv in zip(angles[:-1], norm_arr[i], raw):
            ax.annotate(
                f"{rv:.3f}",
                xy=(ang, nv),
                xytext=(ang, nv + 0.07),
                textcoords="data",
                ha="center", va="center",
                fontsize=6.5,
                color=palette[i],
                fontweight="bold",
            )

    # ── Gridlines and spoke labels ────────────────────────────────────────────
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=9, color="#e2e8f0")

    # Five concentric rings labelled with normalised scale
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0.2", "0.4", "0.6", "0.8", "1.0"],
        fontsize=7, color="#94a3b8"
    )
    ax.yaxis.set_tick_params(labelsize=7)
    ax.grid(color="#334155", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.spines["polar"].set_color("#334155")

    # ── Subtitle explaining the normalisation ─────────────────────────────────
    fig.text(
        0.5, 0.02,
        "All axes normalised to [0.05–1.0] so negative Spatial R² stays visible.\n"
        "Raw values annotated at each vertex. Outward = better on every axis.",
        ha="center", fontsize=8, color="#94a3b8", style="italic"
    )

    ax.set_title("Honest Model Radar Chart",
                 fontsize=13, fontweight="bold",
                 color="#e2e8f0", pad=28)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.38, 1.18),
        fontsize=9,
        framealpha=0.2,
        labelcolor="#e2e8f0",
    )

    plt.tight_layout()
    savefig(fig, "12_Radar_Chart.png")
# ── Fig 13: CPCB Validation ───────────────────────────────────────────────────
print("  Fig 13: CPCB Validation...")
if cpcb_val_df is not None and len(cpcb_val_df)>1:
    fig,axes = plt.subplots(1,2,figsize=(13,5))
    ax=axes[0]
    ax.scatter(cpcb_val_df["pm25_mean"],cpcb_val_df["pm25_proxy"],
               s=100,c="#E91E63",edgecolors="k",alpha=0.9,zorder=3)
    lims=[min(cpcb_val_df["pm25_mean"].min(),
              cpcb_val_df["pm25_proxy"].min())*0.87,
          max(cpcb_val_df["pm25_mean"].max(),
              cpcb_val_df["pm25_proxy"].max())*1.1]
    ax.plot(lims,lims,"k--",lw=1.5,label="1:1 line")
    m_,b_=np.polyfit(cpcb_val_df["pm25_mean"],cpcb_val_df["pm25_proxy"],1)
    xs=np.linspace(lims[0],lims[1],100)
    ax.plot(xs,m_*xs+b_,"r-",lw=2,alpha=0.8,label=f"Fit: {m_:.2f}x+{b_:.1f}")
    for _,row in cpcb_val_df.iterrows():
        ax.annotate(str(row["station_name"])[:10],
                    (row["pm25_mean"],row["pm25_proxy"]),
                    fontsize=6,xytext=(4,4),textcoords="offset points")
    ax.set_xlabel("CPCB Actual PM2.5 (µg/m³)",fontsize=10)
    ax.set_ylabel(f"Model — {best_name}",fontsize=10)
    ax.set_title(f"CPCB Comparison\n"
                 f"r={r_p:.3f}  R²={r2_cpcb:.3f}  RMSE={rmse_cpcb:.2f} µg/m³\n"
                 f"(note: stations used in IDW training)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax=axes[1]
    resid_cv=cpcb_val_df["pm25_mean"]-cpcb_val_df["pm25_proxy"]
    colors_r=["#E53935" if abs(r)>5 else "#1A9850" for r in resid_cv]
    ax.bar(range(len(cpcb_val_df)),resid_cv.values,color=colors_r,alpha=0.85)
    ax.axhline(0,color="k",lw=1.5,ls="--")
    ax.set_xticks(range(len(cpcb_val_df)))
    ax.set_xticklabels([s[:10] for s in cpcb_val_df["station_name"]],
                       rotation=45,ha="right",fontsize=7)
    ax.set_ylabel("Actual − Predicted (µg/m³)",fontsize=10)
    ax.set_title(f"Station Residuals (mean={resid_cv.mean():.2f} µg/m³)")
    ax.grid(axis="y",alpha=0.3)
    plt.suptitle("CPCB Station Comparison — Hybrid LUR v9.5",fontsize=12)
    plt.tight_layout()
    savefig(fig,"13_CPCB_Validation.png")

# ── Fig 14: Remote Sensing Maps ───────────────────────────────────────────────
print("  Fig 14: Remote Sensing Maps...")
extra=[]
if HAS_MODIS   and "mean_aod" in gf.columns:
    extra.append(("mean_aod","MODIS AOD (Feb 1–10)","YlOrBr"))
if HAS_LANDSAT and "mean_lst" in gf.columns:
    extra.append(("mean_lst","Landsat LST (°C)","RdYlBu_r"))
if "uhi" in gf.columns:
    extra.append(("uhi","Urban Heat Island (°C)","Reds"))
if extra:
    gf_plot=grid.copy()
    for col,_,_ in extra: gf_plot[col]=gf[col].values
    fig,axes=plt.subplots(1,len(extra),figsize=(7*len(extra),7))
    if len(extra)==1: axes=[axes]
    for ax,(col,title,cmap) in zip(axes,extra):
        gf_plot.plot(column=col,cmap=cmap,legend=True,
                     legend_kwds={"shrink":0.6,"label":col},
                     ax=ax,edgecolor="none",
                     missing_kwds={"color":"lightgrey"})
        boundary_utm.boundary.plot(ax=ax,color="k",lw=1.0)
        stations_utm.plot(ax=ax,color="white",markersize=60,
                          marker="^",edgecolor="black",lw=1,zorder=5)
        ax.set_title(title,fontsize=10); ax.set_axis_off()
    plt.suptitle("Remote Sensing Predictor Layers",fontsize=11)
    plt.tight_layout()
    savefig(fig,"14_Remote_Sensing_Maps.png")

# ── Fig 15: IDW Confidence Map ────────────────────────────────────────────────
print("  Fig 15: IDW Confidence Map...")
fig,axes=plt.subplots(1,2,figsize=(14,7))
grid2.plot(column="pm25_proxy",cmap=pm_cmap,legend=True,ax=axes[0],
           edgecolor="none",
           legend_kwds={"shrink":0.6,"label":"µg/m³"})
boundary_utm.boundary.plot(ax=axes[0],color="black",lw=1.5)
stations_utm.plot(ax=axes[0],color="white",markersize=60,
                  marker="^",edgecolor="black",lw=1,zorder=5)
axes[0].set_title(f"{best_name} Prediction",fontsize=11)
axes[0].set_axis_off()
grid2.plot(column="idw_confidence",cmap="RdYlGn",legend=True,ax=axes[1],
           edgecolor="none",vmin=0,vmax=1,
           legend_kwds={"shrink":0.6,"label":"Confidence (0–1)"})
boundary_utm.boundary.plot(ax=axes[1],color="black",lw=1.5)
stations_utm.plot(ax=axes[1],color="blue",markersize=80,
                  marker="^",edgecolor="black",lw=1,zorder=5)
axes[1].set_title("IDW Confidence\n(1=near station, 0=far from station)",
                  fontsize=11)
axes[1].set_axis_off()
plt.suptitle("Prediction Map + Confidence — Hybrid LUR v9.5",fontsize=12)
plt.tight_layout()
savefig(fig,"15_Prediction_and_Confidence.png")

# ── Fig 16: AQI Map ───────────────────────────────────────────────────────────
print("  Fig 16: AQI Map...")
fig,ax=plt.subplots(figsize=(12,12))
patches_aqi=[]
for cat in aqi_labels:
    sub=grid2[grid2["aqi"]==cat]
    if len(sub)==0: continue
    sub.plot(ax=ax,color=AQI_COLORS[cat],edgecolor="grey",linewidth=0.04,alpha=0.9)
    patches_aqi.append(Patch(color=AQI_COLORS[cat],
                             label=f"{cat} ({len(sub)}, "
                                   f"{len(sub)/len(grid2)*100:.1f}%)"))
boundary_utm.boundary.plot(ax=ax,color="black",lw=2.5)
stations_utm.plot(ax=ax,color="white",markersize=110,
                  marker="^",edgecolor="black",lw=1.5,zorder=10)
for i in range(n_stations):
    s=stations_utm.iloc[i]
    ax.annotate(agg.iloc[i]["station_name"],
                (s.geometry.x,s.geometry.y),fontsize=5.5,ha="center",
                xytext=(0,12),textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2",fc="white",alpha=0.85),
                zorder=11)
ax.legend(handles=patches_aqi,fontsize=9,title="AQI (NAAQS-India)",
          title_fontsize=9,loc="upper left")
ax.set_title("Air Quality Index — Hyderabad (Feb, 500m, Hybrid LUR v9.5)",
             fontsize=14,fontweight="bold")
ax.set_axis_off(); plt.tight_layout()
savefig(fig,"16_AQI_Map.png")

# ── Fig 17: PM2.5 Final Map ───────────────────────────────────────────────────
print("  Fig 17: PM2.5 Final Map...")
pm_col  = "pm25_bc"  if HAS_BC else "pm25_proxy"
pm_vals = y_full_bc  if HAS_BC else y_full
fig,ax=plt.subplots(figsize=(13,13))
grid2.plot(ax=ax,column=pm_col,cmap=pm_cmap,
           edgecolor="grey",linewidth=0.04,legend=True,
           vmin=max(5,np.percentile(pm_vals,2)),
           vmax=min(250,np.percentile(pm_vals,98)),
           legend_kwds={"label":"PM2.5 (µg/m³)","shrink":0.55})
boundary_utm.boundary.plot(ax=ax,color="black",lw=2.5)
stations_utm.plot(ax=ax,color="white",markersize=130,
                  marker="^",edgecolor="black",lw=1.5,zorder=10)
for i in range(n_stations):
    s=stations_utm.iloc[i]
    ax.annotate(
        f"{agg.iloc[i]['station_name']}\nobs={st_pm25[i]:.0f} µg/m³",
        (s.geometry.x,s.geometry.y),fontsize=5.5,ha="center",
        xytext=(0,14),textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.2",fc="white",alpha=0.88),zorder=11)
ax.text(0.02,0.04,
        f"Model      : {best_name} + BC\n"
        f"Spatial R² : {best_lur['spatial_r2']:.4f}  "
        f"(geographic generalization)\n"
        f"KF R²      : {best_lur['kf_r2']:.4f} ± "
        f"{best_lur['kf_r2_std']:.4f}  (IDW reproduction)\n"
        f"CPCB r     : {r_p:.3f}  R²={r2_cpcb:.3f}  "
        f"RMSE={rmse_cpcb:.2f} µg/m³\n"
        f"N_train    : {len(y)} grid cells | N_stations: {n_stations}\n"
        f"PM2.5      : {pm_vals.min():.1f}–{pm_vals.max():.1f} "
        f"mean={pm_vals.mean():.1f} µg/m³",
        transform=ax.transAxes,fontsize=8,va="bottom",
        bbox=dict(boxstyle="round",fc="white",alpha=0.92))
ax.set_title("PM2.5 Concentration Map — Hyderabad (Hybrid LUR v9.5, 500m)",
             fontsize=14,fontweight="bold")
ax.set_axis_off(); plt.tight_layout()
savefig(fig,"17_PM25_Map.png")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 9: REPORT                                                           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
heading("STAGE 9: REPORT")

W=76
report_path=os.path.join(RPT_DIR,"Analysis_Report.txt")
with open(report_path,"w",encoding="utf-8") as rpt:
    def w(line=""):
        rpt.write(line+"\n")

    w("="*W)
    w("  GEOTA — PM2.5 HYBRID LUR (v9.5, 500m)")
    w("  Hyderabad, Telangana, India")
    w("="*W)

    w(); w("─"*W); w("1. METHODOLOGY"); w("─"*W)
    w("  Architecture : Hybrid LUR (IDW-anchored, environmental predictors)")
    w("  Training data: IDW surface built from 14 CPCB station measurements")
    w("  Predictors   : Road network, land use, terrain, satellite (no coordinates)")
    w("  CV strategy  : Spatial block CV (4 geographic quadrants)")
    w("                 + Standard 5-fold KF CV for comparison")
    w("  Validation   : CPCB station comparison (naive)")
    w()
    w("  Scientific basis:")
    w("  The IDW surface encodes real PM2.5 spatial patterns measured by CPCB.")
    w("  The model learns which environmental features explain those patterns.")
    w("  Spatial block CV tests whether learned patterns generalize to")
    w("  unmeasured parts of the city (held-out geographic quadrants).")

    w(); w("─"*W); w("2. DATA"); w("─"*W)
    w(f"  Stations    : {n_stations}")
    w(f"  PM2.5 range : {st_pm25.min():.1f}–{st_pm25.max():.1f} µg/m³")
    w(f"  Features    : {len(ENHANCED_COLS)} honest | "
      f"{len(ALL_COLS)} with IDW lag")
    w(f"  Grid        : {len(grid)} cells @ {CELL_SIZE_M}m")
    w(f"  MODIS AOD   : {'yes' if HAS_MODIS   else 'no'}")
    w(f"  Landsat LST : {'yes' if HAS_LANDSAT else 'no'}")

    w(); w("─"*W); w("3. MODEL RESULTS"); w("─"*W)
    w(f"  {'Model':<16} {'Category':<16} {'SpatR²':>8} "
      f"{'KF-R²':>7} {'KF±':>6} {'RMSE':>8}  Note")
    w("  "+"─"*72)
    for _,row in metrics_df.iterrows():
        note="[+IDW lag]" if row["has_idw_lag"] else ""
        mark="★" if row["model"]==best_name else " "
        w(f"  {mark} {row['model']:<16} {row['category']:<16} "
          f"{row['spatial_r2']:>8.4f} {row['kf_r2']:>7.4f} "
          f"{row['kf_r2_std']:>6.4f} {row['kf_rmse']:>8.4f}  {note}")
    w()
    w(f"  ★ Best model (spatial CV) : {best_name}")
    w(f"    Spatial R²={best_lur['spatial_r2']:.4f}  "
      f"KF R²={best_lur['kf_r2']:.4f} ± {best_lur['kf_r2_std']:.4f}  "
      f"RMSE={best_lur['kf_rmse']:.4f}")
    if HAS_STACK:
        w(f"  Stacking    : KF R²={r2_stack:.4f}  RMSE={rmse_stack:.4f}")

    w(); w("─"*W); w("4. CPCB STATION COMPARISON"); w("─"*W)
    if cpcb_val_df is not None and len(cpcb_val_df)>1:
        w(f"  r={r_p:.4f}  R²={r2_cpcb:.4f}  RMSE={rmse_cpcb:.2f} µg/m³")
        w("  Note: Stations contributed to IDW training target.")
        if HAS_BC:
            w(f"  BC slope={bc.coef_[0]:.4f}  intercept={bc.intercept_:.4f}")

    w(); w("─"*W); w("5. PERMUTATION IMPORTANCE (top 15)"); w("─"*W)
    for _,row in perm_df.head(15).iterrows():
        ftype="RS" if row["feature"] in RS_SET else \
              "IDW" if row["feature"] in IDW_SET else "LUR"
        w(f"  {row['feature']:<35} {row['importance']:>10.6f}  {ftype}")

    w(); w("─"*W); w("6. GRID PREDICTIONS"); w("─"*W)
    for lbl,vals in [
        ("Raw",         y_full),
        ("Bias-corrected", y_full_bc if HAS_BC else y_full),
    ]:
        w(f"  {lbl:18s}: {vals.min():.1f}–{vals.max():.1f} "
          f"mean={vals.mean():.1f} std={vals.std():.1f} µg/m³")
    w("\n  AQI distribution:")
    for c in aqi_labels:
        n_=(grid2["aqi"]==c).sum()
        w(f"    {c:15s}: {n_:4d} ({n_/len(grid2)*100:5.1f}%)")

    w(); w("─"*W); w("7. FIGURES"); w("─"*W)
    for fn in sorted(os.listdir(FIG_DIR)):
        if fn.endswith(".png"):
            sz=os.path.getsize(os.path.join(FIG_DIR,fn))/1024
            w(f"  {fn:55s} {sz:8.1f} KB")

    w(); w("="*W); w("END OF REPORT"); w("="*W)

print(f"  Report: {report_path}")


# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  Hybrid LUR v9.5 — COMPLETE")
print(f"{'='*70}")

print(f"\n  ┌─ Spatial Block CV (geographic generalization) ─────────────────")
print(f"  │  Best model    : {best_name}")
print(f"  │  Spatial R²    : {best_lur['spatial_r2']:.4f}")
print(f"  │  KF R²         : {best_lur['kf_r2']:.4f} ± {best_lur['kf_r2_std']:.4f}")
print(f"  │  KF RMSE       : {best_lur['kf_rmse']:.4f} µg/m³")
if HAS_STACK:
    print(f"  │  Stacking KF R²: {r2_stack:.4f}")
print(f"  └────────────────────────────────────────────────────────────────")

print(f"\n  ┌─ CPCB Station Comparison ───────────────────────────────────────")
if cpcb_val_df is not None and len(cpcb_val_df)>1:
    print(f"  │  r={r_p:.4f}  R²={r2_cpcb:.4f}  RMSE={rmse_cpcb:.2f} µg/m³")
print(f"  └────────────────────────────────────────────────────────────────")

print(f"\n  Grid   : {len(grid)} cells @ {CELL_SIZE_M}m")
print(f"  Feats  : {len(ENHANCED_COLS)} honest | {len(ALL_COLS)} with IDW lag")
if HAS_BC:
    print(f"  BC map : {y_full_bc.min():.1f}–{y_full_bc.max():.1f} µg/m³")
print()
fig_files=sorted(os.listdir(FIG_DIR))
png_files=[f for f in fig_files if f.endswith(".png")]
total_mb =sum(os.path.getsize(os.path.join(FIG_DIR,f))
              for f in png_files)/1e6
print(f"  {len(png_files)} figures  |  {total_mb:.1f} MB\n")
for f in png_files:
    sz=os.path.getsize(os.path.join(FIG_DIR,f))/1024
    print(f"    {f:55s} {sz:8.1f} KB")
print()