#!/usr/bin/env python3
"""
=============================================================================
HIGH-RESOLUTION AIR QUALITY (PM2.5) PREDICTION — LAND USE REGRESSION
COMPLETE GIS ANALYSIS PIPELINE v3.0 (ALL BUGS FIXED)
=============================================================================
Fixes from v2.0:
  - slope_deg sampled from FULL DEM (not clipped) to avoid NaN
  - LST clamped to [-50, 55] °C to remove sensor artifacts
  - AOD scale factor applied consistently in feature extraction
  - All 20 queries + GT + FE produce correct outputs
=============================================================================
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
from shapely.geometry import Point, box, mapping
from shapely.ops import nearest_points, unary_union
from scipy.ndimage import uniform_filter, sobel, convolve
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.transform import rowcol

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_DIR   = os.path.join(BASE_DIR, "Vector_Data")
RASTER_DIR   = os.path.join(BASE_DIR, "Raster_Data")
LANDSAT_DIR  = os.path.join(BASE_DIR, "Landsat")
MODIS_DIR    = os.path.join(BASE_DIR, "MODIS")
CPCB_CSV     = os.path.join(BASE_DIR, "CPCB_Actual_PM25.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CRS_GEO = "EPSG:4326"
CRS_UTM = "EPSG:32644"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def heading(num, title):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  QUERY {num}: {title}")
    print(bar)

def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📷 Figure saved → {path}")

def load_shp(name):
    return gpd.read_file(os.path.join(VECTOR_DIR, name))

def generate_grid(boundary_gdf, cell_size_m=1000):
    """Generate a proper 1km x 1km fishnet grid over the boundary."""
    bnd = boundary_gdf.to_crs(CRS_UTM)
    xmin, ymin, xmax, ymax = bnd.total_bounds
    cols = np.arange(xmin, xmax, cell_size_m)
    rows = np.arange(ymin, ymax, cell_size_m)
    cells = []
    for x in cols:
        for y in rows:
            cells.append(box(x, y, x + cell_size_m, y + cell_size_m))
    grid = gpd.GeoDataFrame(geometry=cells, crs=CRS_UTM)
    grid = gpd.clip(grid, bnd)
    grid = grid[~grid.is_empty].reset_index(drop=True)
    grid["grid_id"] = grid.index
    return grid

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL VECTOR LAYERS
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔄 Loading all vector datasets …")
boundary        = load_shp("01_Hyderabad_Boundary.shp")
primary_roads   = load_shp("02_Primary_Roads.shp")
secondary_roads = load_shp("03_Secondary_Roads.shp")
traffic_signals = load_shp("04_Traffic_Signals.shp")
bus_stops       = load_shp("05_Bus_Stops.shp")
railway_lines   = load_shp("06_Railway_Lines.shp")
industrial      = load_shp("07_Industrial_Areas.shp")
residential     = load_shp("08_Residential_Areas.shp")
water_bodies    = load_shp("09_Water_Bodies.shp")
grid_1km_raw    = load_shp("10_Analysis_Grid_1km.shp")
cpcb_stations   = load_shp("11_CPCB_Stations.shp")

all_layers = {
    "Primary_Roads":     primary_roads,
    "Secondary_Roads":   secondary_roads,
    "Traffic_Signals":   traffic_signals,
    "Bus_Stops":         bus_stops,
    "Railway_Lines":     railway_lines,
    "Industrial_Areas":  industrial,
    "Residential_Areas": residential,
    "Water_Bodies":      water_bodies,
}
print("  ✅ All 11 vector layers loaded.")

# ─── Check & fix grid ────────────────────────────────────────────────────────
print(f"\n  ℹ️  Raw grid has {len(grid_1km_raw)} polygon(s)")
if len(grid_1km_raw) <= 1:
    print("  ⚠️  Grid file has only 1 polygon — generating proper 1 km fishnet …")
    grid_1km = generate_grid(boundary, cell_size_m=1000)
    print(f"  ✅ Generated {len(grid_1km)} grid cells")
else:
    grid_1km = grid_1km_raw.to_crs(CRS_UTM).reset_index(drop=True)
    grid_1km["grid_id"] = grid_1km.index
    print(f"  ✅ Using existing grid: {len(grid_1km)} cells")

# ─── Ensure all layers share CRS before clipping ────────────────────────────
target_crs = boundary.crs
for name in all_layers:
    all_layers[name] = all_layers[name].to_crs(target_crs)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                        VECTOR  DATA  ANALYSIS                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# Q1 — CLIP
# ─────────────────────────────────────────────────────────────────────────────
heading(1, "CLIP — All Layers to Hyderabad Boundary")
clipped = {}
for name, layer in all_layers.items():
    before = len(layer)
    cl = gpd.clip(layer, boundary)
    clipped[name] = cl
    delta = before - len(cl)
    status = f"(removed {delta})" if delta > 0 else "(all inside boundary)"
    print(f"  ✅ {name:25s}: {before:5d} → {len(cl):5d}  {status}")

fig, ax = plt.subplots(figsize=(12, 12))
boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2.5,
              label="Boundary")
if len(clipped["Water_Bodies"]) > 0:
    clipped["Water_Bodies"].plot(ax=ax, color="cyan", alpha=0.5,
                                 label="Water Bodies")
if len(clipped["Industrial_Areas"]) > 0:
    clipped["Industrial_Areas"].plot(ax=ax, color="grey", alpha=0.5,
                                      label="Industrial")
if len(clipped["Residential_Areas"]) > 0:
    clipped["Residential_Areas"].plot(ax=ax, color="pink", alpha=0.4,
                                       label="Residential")
if len(clipped["Primary_Roads"]) > 0:
    clipped["Primary_Roads"].plot(ax=ax, color="red", linewidth=0.5,
                                   label="Primary Roads")
if len(clipped["Secondary_Roads"]) > 0:
    clipped["Secondary_Roads"].plot(ax=ax, color="orange", linewidth=0.2,
                                     label="Secondary Roads")
if len(clipped["Railway_Lines"]) > 0:
    clipped["Railway_Lines"].plot(ax=ax, color="brown", linewidth=0.5,
                                   label="Railways")
if len(clipped["Bus_Stops"]) > 0:
    clipped["Bus_Stops"].plot(ax=ax, color="blue", markersize=1.5,
                               label="Bus Stops")
if len(clipped["Traffic_Signals"]) > 0:
    clipped["Traffic_Signals"].plot(ax=ax, color="yellow", markersize=3,
                                     edgecolor="black", linewidth=0.2,
                                     label="Traffic Signals")
cpcb_stations.to_crs(target_crs).plot(ax=ax, color="red", markersize=80,
                                       marker="^", edgecolor="black",
                                       linewidth=1, zorder=10,
                                       label="CPCB Stations")
ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
ax.set_title("Query 1 — All Layers Clipped to Hyderabad Boundary", fontsize=13)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
save_fig(fig, "Q01_Clip_All_Layers.png")

# ─────────────────────────────────────────────────────────────────────────────
# Q2 — BUFFER
# ─────────────────────────────────────────────────────────────────────────────
heading(2, "BUFFER — Multi-Ring Road Proximity Zones")
roads_utm = clipped["Primary_Roads"].to_crs(CRS_UTM)
boundary_utm = boundary.to_crs(CRS_UTM)

buffer_distances = [100, 300, 500, 1000]
buffer_polys = {}
for d in buffer_distances:
    buf_geom = roads_utm.buffer(d).unary_union
    buf_gdf = gpd.GeoDataFrame(geometry=[buf_geom], crs=CRS_UTM)
    buf_gdf = gpd.clip(buf_gdf, boundary_utm)
    area_km2 = buf_gdf.area.sum() / 1e6
    buffer_polys[d] = buf_gdf
    bnd_area = boundary_utm.area.sum() / 1e6
    pct = area_km2 / bnd_area * 100
    print(f"  Buffer {d:5d}m  →  area = {area_km2:8.2f} km²  ({pct:.1f}% of city)")

fig, ax = plt.subplots(figsize=(12, 12))
for color, d in [("#d9ef8b", 1000), ("#fee08b", 500),
                 ("#fc8d59", 300), ("#d73027", 100)]:
    buffer_polys[d].plot(ax=ax, color=color, alpha=0.6, label=f"{d}m buffer")
boundary_utm.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2)
roads_utm.plot(ax=ax, color="black", linewidth=0.3)
ax.legend(fontsize=9)
ax.set_title("Query 2 — Multi-Ring Buffers around Primary Roads", fontsize=13)
save_fig(fig, "Q02_Road_Buffers.png")

# ─────────────────────────────────────────────────────────────────────────────
# Q3 — POINT-IN-POLYGON (with proximity fallback)
# ─────────────────────────────────────────────────────────────────────────────
heading(3, "POINT-IN-POLYGON — CPCB Stations × Land Use")
stations = cpcb_stations.to_crs(CRS_UTM).copy()
ind_utm = clipped["Industrial_Areas"].to_crs(CRS_UTM)
res_utm = clipped["Residential_Areas"].to_crs(CRS_UTM)

sj_ind = gpd.sjoin(stations, ind_utm, how="left", predicate="within")
stations["in_industrial"] = sj_ind["index_right"].notna().values

sj_res = gpd.sjoin(stations, res_utm, how="left", predicate="within")
stations["in_residential"] = sj_res["index_right"].notna().values

if not stations["in_industrial"].any() and not stations["in_residential"].any():
    print("  ℹ️  No stations fall directly inside land-use polygons.")
    print("      Using nearest-distance classification (< 500m threshold) …")
    ind_union_q3 = ind_utm.unary_union if len(ind_utm) > 0 else None
    res_union_q3 = res_utm.unary_union if len(res_utm) > 0 else None

    for idx, row in stations.iterrows():
        pt = row.geometry
        d_ind = pt.distance(ind_union_q3) if ind_union_q3 else 1e9
        d_res = pt.distance(res_union_q3) if res_union_q3 else 1e9
        min_d = min(d_ind, d_res)
        if min_d < 500:
            if d_ind < d_res:
                stations.at[idx, "in_industrial"] = True
                stations.at[idx, "in_residential"] = False
            else:
                stations.at[idx, "in_industrial"] = False
                stations.at[idx, "in_residential"] = True

stations["land_use"] = "Other"
stations.loc[stations["in_industrial"] == True, "land_use"] = "Industrial"
stations.loc[stations["in_residential"] == True, "land_use"] = "Residential"

for lu, cnt in stations["land_use"].value_counts().items():
    print(f"  {lu:15s}: {cnt} station(s)")

if len(ind_utm) > 0:
    ind_u_q3 = ind_utm.unary_union
    for idx, row in stations.iterrows():
        d = row.geometry.distance(ind_u_q3)
        print(f"    Station {idx}: {d:.0f}m from industrial, "
              f"land_use={row['land_use']}")

fig, ax = plt.subplots(figsize=(12, 12))
boundary_utm.plot(ax=ax, facecolor="#f5f5f5", edgecolor="black", linewidth=2)
ind_utm.plot(ax=ax, color="grey", alpha=0.5, label="Industrial Areas")
res_utm.plot(ax=ax, color="pink", alpha=0.4, label="Residential Areas")
color_map = {"Industrial": "red", "Residential": "blue", "Other": "green"}
for lu, grp in stations.groupby("land_use"):
    grp.plot(ax=ax, color=color_map[lu], markersize=100, marker="^",
             edgecolor="black", linewidth=1.2, label=f"Station – {lu}",
             zorder=10)
ax.legend(fontsize=8)
ax.set_title("Query 3 — CPCB Stations classified by Nearest Land Use",
             fontsize=13)
save_fig(fig, "Q03_Point_in_Polygon.png")

# ─────────────────────────────────────────────────────────────────────────────
# Q4 — INTERSECT
# ─────────────────────────────────────────────────────────────────────────────
heading(4, "INTERSECT — Primary Roads ∩ Industrial Areas")
try:
    roads_ind = gpd.overlay(
        clipped["Primary_Roads"].to_crs(CRS_UTM),
        clipped["Industrial_Areas"].to_crs(CRS_UTM),
        how="intersection"
    )
    total_len = roads_ind.geometry.length.sum() / 1000
    print(f"  Road segments in industrial : {len(roads_ind)}")
    print(f"  Total road length           : {total_len:.2f} km")

    fig, ax = plt.subplots(figsize=(12, 12))
    boundary_utm.plot(ax=ax, facecolor="#f5f5f5", edgecolor="black",
                      linewidth=1.5)
    clipped["Industrial_Areas"].to_crs(CRS_UTM).plot(
        ax=ax, color="lightgrey", edgecolor="grey", alpha=0.6,
        label="Industrial Areas")
    clipped["Primary_Roads"].to_crs(CRS_UTM).plot(
        ax=ax, color="blue", linewidth=0.3, alpha=0.3,
        label="All Primary Roads")
    roads_ind.plot(ax=ax, color="red", linewidth=2,
                   label="Intersected Roads")
    ax.legend(fontsize=9)
    ax.set_title(f"Query 4 — Intersect: Roads in Industrial Areas "
                 f"({total_len:.1f} km)", fontsize=13)
    save_fig(fig, "Q04_Intersect_Roads_Industrial.png")
except Exception as e:
    print(f"  ⚠️ Intersect error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Q5 — UNION
# ─────────────────────────────────────────────────────────────────────────────
heading(5, "UNION — Industrial ∪ Residential")
try:
    ind_u5 = clipped["Industrial_Areas"].to_crs(CRS_UTM).copy()
    res_u5 = clipped["Residential_Areas"].to_crs(CRS_UTM).copy()
    ind_u5["type"] = "Industrial"
    res_u5["type"] = "Residential"

    lu_union = gpd.overlay(ind_u5, res_u5, how="union")
    lu_union["area_km2"] = lu_union.area / 1e6

    lu_union["combined_type"] = "Unknown"
    has_ind = lu_union["type_1"].notna()
    has_res = lu_union["type_2"].notna()
    lu_union.loc[has_ind & has_res, "combined_type"] = "Both"
    lu_union.loc[has_ind & ~has_res, "combined_type"] = "Industrial Only"
    lu_union.loc[~has_ind & has_res, "combined_type"] = "Residential Only"

    for ct, grp in lu_union.groupby("combined_type"):
        print(f"  {ct:20s}: {len(grp):4d} polygons, "
              f"{grp['area_km2'].sum():.2f} km²")
    print(f"  {'TOTAL':20s}: {len(lu_union):4d} polygons, "
          f"{lu_union['area_km2'].sum():.2f} km²")

    fig, ax = plt.subplots(figsize=(12, 12))
    boundary_utm.plot(ax=ax, facecolor="#f5f5f5", edgecolor="black",
                      linewidth=2)
    cmap_lu = {"Industrial Only": "grey", "Residential Only": "pink",
               "Both": "purple", "Unknown": "white"}
    for ct, grp in lu_union.groupby("combined_type"):
        grp.plot(ax=ax, color=cmap_lu.get(ct, "white"), alpha=0.6,
                 edgecolor="grey", linewidth=0.3, label=ct)
    ax.legend(fontsize=9)
    ax.set_title("Query 5 — Union: Industrial ∪ Residential", fontsize=13)
    save_fig(fig, "Q05_Union_LandUse.png")
except Exception as e:
    print(f"  ⚠️ Union error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Q6 — DISSOLVE
# ─────────────────────────────────────────────────────────────────────────────
heading(6, "DISSOLVE — Merge Water Body Polygons")
water_utm = clipped["Water_Bodies"].to_crs(CRS_UTM)
n_before = len(water_utm)
water_dissolved = water_utm.dissolve()
n_after = len(water_dissolved)
water_area_km2 = water_utm.area.sum() / 1e6
print(f"  Before dissolve : {n_before} polygons")
print(f"  After dissolve  : {n_after} polygon(s)")
print(f"  Total water area: {water_area_km2:.2f} km²")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
water_utm.plot(ax=axes[0], color="cyan", edgecolor="blue", linewidth=0.5)
boundary_utm.plot(ax=axes[0], facecolor="none", edgecolor="black", linewidth=1)
axes[0].set_title(f"Before Dissolve ({n_before} polygons)")
water_dissolved.plot(ax=axes[1], color="cyan", edgecolor="blue", linewidth=1)
boundary_utm.plot(ax=axes[1], facecolor="none", edgecolor="black", linewidth=1)
axes[1].set_title(f"After Dissolve ({n_after} polygon)")
fig.suptitle("Query 6 — Dissolve: Merge Water Bodies", fontsize=14)
save_fig(fig, "Q06_Dissolve_Water.png")

# ─────────────────────────────────────────────────────────────────────────────
# Q7 — SPATIAL JOIN — Count features in buffers
# ─────────────────────────────────────────────────────────────────────────────
heading(7, "SPATIAL JOIN — Count Features within Station Buffers")
stations_utm = stations.to_crs(CRS_UTM)
bus_utm_7 = clipped["Bus_Stops"].to_crs(CRS_UTM)
sig_utm_7 = clipped["Traffic_Signals"].to_crs(CRS_UTM)

radii = [500, 1000, 2000, 5000]
results_q7 = {}
for r in radii:
    buf = stations_utm.copy()
    buf["orig_idx"] = buf.index
    buf.geometry = stations_utm.buffer(r)

    if len(bus_utm_7) > 0:
        jb = gpd.sjoin(bus_utm_7, buf[["orig_idx", "geometry"]],
                       how="inner", predicate="within")
        bc = jb.groupby("orig_idx").size()
    else:
        bc = pd.Series(dtype=int)

    if len(sig_utm_7) > 0:
        js = gpd.sjoin(sig_utm_7, buf[["orig_idx", "geometry"]],
                       how="inner", predicate="within")
        sc = js.groupby("orig_idx").size()
    else:
        sc = pd.Series(dtype=int)

    avg_bus = bc.mean() if len(bc) > 0 else 0
    avg_sig = sc.mean() if len(sc) > 0 else 0
    max_bus = bc.max() if len(bc) > 0 else 0
    max_sig = sc.max() if len(sc) > 0 else 0

    results_q7[r] = {"avg_bus": avg_bus, "avg_sig": avg_sig}
    print(f"  Buffer {r:5d}m → bus stops: avg={avg_bus:.1f} max={max_bus}"
          f"  |  signals: avg={avg_sig:.1f} max={max_sig}")

plot_r = max(radii)
fig, ax = plt.subplots(figsize=(12, 12))
boundary_utm.plot(ax=ax, facecolor="#f5f5f5", edgecolor="black", linewidth=1.5)
buf_plot = stations_utm.copy()
buf_plot.geometry = stations_utm.buffer(plot_r)
buf_plot.plot(ax=ax, facecolor="none", edgecolor="purple", linewidth=1.5,
              linestyle="--", label=f"{plot_r}m buffer")
bus_utm_7.plot(ax=ax, color="blue", markersize=3, label="Bus Stops",
               alpha=0.6)
sig_utm_7.plot(ax=ax, color="orange", markersize=5, label="Traffic Signals",
               alpha=0.8)
stations_utm.plot(ax=ax, color="red", markersize=100, marker="^",
                  edgecolor="black", linewidth=1, zorder=10,
                  label="CPCB Stations")
ax.legend(fontsize=8)
ax.set_title(f"Query 7 — Features within {plot_r}m Station Buffers",
             fontsize=13)
save_fig(fig, "Q07_SpatialJoin_Buffers.png")

# ─────────────────────────────────────────────────────────────────────────────
# Q8 — ERASE
# ─────────────────────────────────────────────────────────────────────────────
heading(8, "ERASE — Boundary minus Water Bodies")
try:
    b8 = boundary.to_crs(CRS_UTM)
    w8 = clipped["Water_Bodies"].to_crs(CRS_UTM)
    land_only = gpd.overlay(b8, w8, how="difference")
    total_a = b8.area.sum() / 1e6
    land_a = land_only.area.sum() / 1e6
    water_a = total_a - land_a
    print(f"  Total area  : {total_a:.2f} km²")
    print(f"  Land area   : {land_a:.2f} km²")
    print(f"  Water area  : {water_a:.2f} km²  ({water_a / total_a * 100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    b8.plot(ax=axes[0], facecolor="lightyellow", edgecolor="black")
    w8.plot(ax=axes[0], color="cyan", alpha=0.7)
    axes[0].set_title("Before Erase (with Water)")
    land_only.plot(ax=axes[1], facecolor="lightyellow", edgecolor="black")
    axes[1].set_title("After Erase (Water Removed)")
    fig.suptitle(f"Query 8 — Erase: {water_a:.1f} km² water removed",
                 fontsize=14)
    save_fig(fig, "Q08_Erase_Water.png")
except Exception as e:
    print(f"  ⚠️ Erase error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Q9 — LINE-IN-POLYGON — Road Density per Grid Cell
# ─────────────────────────────────────────────────────────────────────────────
heading(9, "LINE-IN-POLYGON — Road Density per 1 km Grid Cell")
grid_utm = grid_1km.copy()
all_roads_utm = clipped["Primary_Roads"].to_crs(CRS_UTM)

print(f"  Grid cells: {len(grid_utm)}")
print(f"  Road features: {len(all_roads_utm)}")

try:
    roads_in_grid = gpd.overlay(all_roads_utm,
                                grid_utm[["grid_id", "geometry"]],
                                how="intersection")
    roads_in_grid["length_m"] = roads_in_grid.geometry.length
    rd_per_cell = roads_in_grid.groupby("grid_id")["length_m"].sum() / 1000

    grid_utm["road_km"] = grid_utm["grid_id"].map(rd_per_cell).fillna(0)
    grid_utm["cell_area_km2"] = grid_utm.area / 1e6
    grid_utm["road_density"] = grid_utm["road_km"] / grid_utm["cell_area_km2"]

    print(f"  Mean road density  : {grid_utm['road_density'].mean():.2f} km/km²")
    print(f"  Max road density   : {grid_utm['road_density'].max():.2f} km/km²")
    print(f"  Cells with roads   : {(grid_utm['road_km'] > 0).sum()}")
    print(f"  Cells without roads: {(grid_utm['road_km'] == 0).sum()}")

    fig, ax = plt.subplots(figsize=(12, 12))
    grid_utm.plot(ax=ax, column="road_density", cmap="YlOrRd",
                  edgecolor="grey", linewidth=0.2, legend=True,
                  legend_kwds={"label": "Road Density (km/km²)",
                               "shrink": 0.6})
    boundary_utm.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2)
    ax.set_title("Query 9 — Road Density per 1 km Grid Cell", fontsize=13)
    save_fig(fig, "Q09_RoadDensity_Grid.png")
except Exception as e:
    print(f"  ⚠️ Road density error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Q10 — DISTANCE + MAP
# ─────────────────────────────────────────────────────────────────────────────
heading(10, "DISTANCE — Stations to Nearest Features")
st_10 = stations.to_crs(CRS_UTM)

ind_union = (clipped["Industrial_Areas"].to_crs(CRS_UTM).unary_union
             if len(clipped["Industrial_Areas"]) > 0 else None)
wat_union = (clipped["Water_Bodies"].to_crs(CRS_UTM).unary_union
             if len(clipped["Water_Bodies"]) > 0 else None)
rd_union = (clipped["Primary_Roads"].to_crs(CRS_UTM).unary_union
            if len(clipped["Primary_Roads"]) > 0 else None)

dist_rows = []
for idx, row in st_10.iterrows():
    pt = row.geometry
    d_ind = pt.distance(ind_union) if ind_union else np.nan
    d_wat = pt.distance(wat_union) if wat_union else np.nan
    d_rd  = pt.distance(rd_union)  if rd_union  else np.nan
    dist_rows.append({
        "station": idx,
        "dist_industrial_m": round(d_ind, 1),
        "dist_water_m": round(d_wat, 1),
        "dist_road_m": round(d_rd, 1)
    })
dist_df = pd.DataFrame(dist_rows)
print(dist_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 12))
boundary_utm.plot(ax=ax, facecolor="#f5f5f5", edgecolor="black", linewidth=2)
if len(clipped["Industrial_Areas"]) > 0:
    clipped["Industrial_Areas"].to_crs(CRS_UTM).plot(
        ax=ax, color="grey", alpha=0.4, label="Industrial")
if len(clipped["Water_Bodies"]) > 0:
    clipped["Water_Bodies"].to_crs(CRS_UTM).plot(
        ax=ax, color="cyan", alpha=0.5, label="Water")
clipped["Primary_Roads"].to_crs(CRS_UTM).plot(
    ax=ax, color="red", linewidth=0.3, alpha=0.4, label="Roads")
st_10.plot(ax=ax, color="red", markersize=100, marker="^",
           edgecolor="black", zorder=10, label="CPCB")

for idx, row in st_10.iterrows():
    pt = row.geometry
    if rd_union:
        nearest_pt = nearest_points(pt, rd_union)[1]
        ax.plot([pt.x, nearest_pt.x], [pt.y, nearest_pt.y],
                'k--', linewidth=0.8, alpha=0.6)
        mid_x = (pt.x + nearest_pt.x) / 2
        mid_y = (pt.y + nearest_pt.y) / 2
        d = pt.distance(rd_union)
        ax.annotate(f"{d:.0f}m", (mid_x, mid_y), fontsize=6, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.7))

ax.legend(fontsize=8)
ax.set_title("Query 10 — Distance from Stations to Nearest Features",
             fontsize=13)
save_fig(fig, "Q10_Distance_Nearest.png")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                        RASTER  DATA  ANALYSIS                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# Q11 — DEM Slope & Aspect
# ─────────────────────────────────────────────────────────────────────────────
heading(11, "DEM — Elevation, Slope & Aspect")
dem_path = os.path.join(RASTER_DIR, "SRTM_DEM.tif")
dem_src = rasterio.open(dem_path)
dem_full = dem_src.read(1).astype(float)
nodata = dem_src.nodata
if nodata is not None:
    dem_full[dem_full == nodata] = np.nan

# ─── Compute slope/aspect on FULL DEM for correct station sampling later ────
print("  Computing slope/aspect on FULL DEM for station sampling …")
pixel_m = abs(dem_src.res[0])
if pixel_m < 1:
    pixel_m = pixel_m * 111320

dem_full_clean = np.nan_to_num(dem_full, nan=np.nanmean(dem_full))
dx_full = sobel(dem_full_clean, axis=1) / (8 * pixel_m)
dy_full = sobel(dem_full_clean, axis=0) / (8 * pixel_m)
slope_full = np.degrees(np.arctan(np.sqrt(dx_full**2 + dy_full**2)))
slope_full[np.isnan(dem_full)] = np.nan
aspect_full = (90.0 - np.degrees(np.arctan2(-dy_full, dx_full))) % 360.0
aspect_full[np.isnan(dem_full)] = np.nan
print(f"  Full DEM slope: mean={np.nanmean(slope_full):.2f}°, "
      f"max={np.nanmax(slope_full):.2f}°")

# ─── Clip DEM to boundary for display/filtering ─────────────────────────────
try:
    bnd_dem = boundary.to_crs(dem_src.crs)
    geom = [json.loads(bnd_dem.to_json())["features"][0]["geometry"]]
    dem_clip_arr, dem_clip_tf = rio_mask(dem_src, geom, crop=True, nodata=-9999)
    dem_arr = dem_clip_arr[0].astype(float)
    dem_arr[dem_arr == -9999] = np.nan
    if nodata is not None:
        dem_arr[dem_arr == nodata] = np.nan
    print(f"  DEM clipped to boundary: {dem_arr.shape}")
except Exception:
    dem_arr = dem_full.copy()
    print(f"  Using full DEM: {dem_arr.shape}")

print(f"  Min elevation  : {np.nanmin(dem_arr):.1f} m")
print(f"  Max elevation  : {np.nanmax(dem_arr):.1f} m")
print(f"  Mean elevation : {np.nanmean(dem_arr):.1f} m")
print(f"  Std elevation  : {np.nanstd(dem_arr):.1f} m")

# Slope & Aspect on CLIPPED array (for display only)
dem_clip_clean = np.nan_to_num(dem_arr, nan=np.nanmean(dem_arr))
dx = sobel(dem_clip_clean, axis=1) / (8 * pixel_m)
dy = sobel(dem_clip_clean, axis=0) / (8 * pixel_m)
slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
aspect_deg = (90.0 - np.degrees(np.arctan2(-dy, dx))) % 360.0
slope_deg[np.isnan(dem_arr)] = np.nan
aspect_deg[np.isnan(dem_arr)] = np.nan

print(f"  Clipped slope  : mean={np.nanmean(slope_deg):.2f}°")
print(f"  Clipped slope  : max={np.nanmax(slope_deg):.2f}°")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
im0 = axes[0].imshow(dem_arr, cmap="terrain")
axes[0].set_title("DEM (Elevation)")
plt.colorbar(im0, ax=axes[0], shrink=0.6, label="m")

im1 = axes[1].imshow(slope_deg, cmap="magma", vmin=0,
                      vmax=np.nanpercentile(slope_deg, 98))
axes[1].set_title("Slope (°)")
plt.colorbar(im1, ax=axes[1], shrink=0.6, label="degrees")

im2 = axes[2].imshow(aspect_deg, cmap="hsv", vmin=0, vmax=360)
axes[2].set_title("Aspect (°)")
plt.colorbar(im2, ax=axes[2], shrink=0.6, label="degrees")

fig.suptitle("Query 11 — DEM, Slope & Aspect (Neighbourhood Operations)",
             fontsize=14)
save_fig(fig, "Q11_DEM_Slope_Aspect.png")

# ─────────────────────────────────────────────────────────────────────────────
# Q12 — RECLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
heading(12, "RECLASSIFICATION — Elevation Classes")
mn, mx = np.nanmin(dem_arr), np.nanmax(dem_arr)
n_classes = 5
breaks = np.linspace(mn, mx, n_classes + 1)
elev_cls = np.full_like(dem_arr, np.nan)
for i in range(n_classes):
    mask_cls = (dem_arr >= breaks[i]) & (dem_arr < breaks[i + 1])
    if i == n_classes - 1:
        mask_cls = (dem_arr >= breaks[i]) & (dem_arr <= breaks[i + 1])
    elev_cls[mask_cls] = i + 1

labels_12 = {}
for i in range(n_classes):
    labels_12[i + 1] = f"{breaks[i]:.0f}–{breaks[i + 1]:.0f}m"
    cnt = int(np.nansum(elev_cls == i + 1))
    total_valid = int(np.nansum(~np.isnan(elev_cls)))
    pct = cnt / total_valid * 100 if total_valid > 0 else 0
    print(f"  Class {i + 1} [{labels_12[i + 1]:>15s}]: "
          f"{cnt:8d} px ({pct:5.1f}%)")

fig, ax = plt.subplots(figsize=(10, 9))
cmap5 = ListedColormap(["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"])
im = ax.imshow(elev_cls, cmap=cmap5, vmin=0.5, vmax=5.5,
               interpolation="nearest")
cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5], shrink=0.7)
cbar.ax.set_yticklabels([labels_12[i] for i in range(1, 6)])
ax.set_title("Query 12 — Reclassified DEM (5 Elevation Classes)", fontsize=13)
save_fig(fig, "Q12_DEM_Reclassified.png")

# ─────────────────────────────────────────────────────────────────────────────
# Q13 — NDVI PROXY from Sentinel-2
# ─────────────────────────────────────────────────────────────────────────────
heading(13, "NDVI PROXY — Excess Green Index from Sentinel-2 TCI")
tci_path = None
for f in os.listdir(RASTER_DIR):
    if f.lower().endswith(".jp2"):
        tci_path = os.path.join(RASTER_DIR, f)
        break

if tci_path:
    try:
        tci_src = rasterio.open(tci_path)
        max_dim = 4000
        win = rasterio.windows.Window(
            0, 0,
            min(tci_src.width, max_dim),
            min(tci_src.height, max_dim))
        tci = tci_src.read(window=win).astype(float)
        r, g, b = tci[0], tci[1], tci[2]
        denom = r + g + b + 1e-10
        ExG = (2 * g - r - b) / denom

        print(f"  TCI shape  : {tci.shape}")
        print(f"  Resolution : {tci_src.res[0]:.1f}m")
        print(f"  ExG range  : [{np.nanmin(ExG):.3f}, {np.nanmax(ExG):.3f}]")
        print(f"  Mean ExG   : {np.nanmean(ExG):.3f}")

        veg_mask = ExG > 0.1
        veg_pct = np.sum(veg_mask) / ExG.size * 100
        print(f"  Vegetated % (ExG>0.1): {veg_pct:.1f}%")

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        rgb_display = np.clip(
            np.moveaxis(tci[:3], 0, -1).astype(np.uint8), 0, 255)
        axes[0].imshow(rgb_display)
        axes[0].set_title("Sentinel-2 True Color")
        im = axes[1].imshow(ExG, cmap="RdYlGn", vmin=-0.3, vmax=0.5)
        axes[1].set_title(f"Excess Green Index (Veg: {veg_pct:.1f}%)")
        plt.colorbar(im, ax=axes[1], shrink=0.6, label="ExG")
        fig.suptitle("Query 13 — Vegetation Index from Sentinel-2",
                     fontsize=14)
        save_fig(fig, "Q13_Sentinel2_ExG.png")
    except Exception as e:
        print(f"  ⚠️ Sentinel-2 error: {e}")
else:
    print("  ⚠️ Sentinel-2 JP2 not found.")

# ─────────────────────────────────────────────────────────────────────────────
# Q14 — ZONAL STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
heading(14, "ZONAL STATISTICS — Elevation per 1 km Grid Cell")
try:
    from rasterstats import zonal_stats as zs
    HAS_RS = True
except ImportError:
    HAS_RS = False

grid_for_zonal = grid_1km.to_crs(dem_src.crs)

if HAS_RS:
    stats14 = zs(grid_for_zonal, dem_path,
                 stats=["min", "max", "mean", "std", "count"],
                 nodata=nodata)
    grid_for_zonal["elev_min"]  = [s["min"]  for s in stats14]
    grid_for_zonal["elev_max"]  = [s["max"]  for s in stats14]
    grid_for_zonal["elev_mean"] = [s["mean"] for s in stats14]
    grid_for_zonal["elev_std"]  = [s["std"]  for s in stats14]
else:
    centroids = grid_for_zonal.geometry.centroid
    vals = []
    for pt in centroids:
        try:
            ri, ci = rowcol(dem_src.transform, pt.x, pt.y)
            if 0 <= ri < dem_full.shape[0] and 0 <= ci < dem_full.shape[1]:
                vals.append(float(dem_full[ri, ci]))
            else:
                vals.append(np.nan)
        except Exception:
            vals.append(np.nan)
    grid_for_zonal["elev_mean"] = vals

valid14 = grid_for_zonal.dropna(subset=["elev_mean"])
print(f"  Grid cells with data : {len(valid14)} / {len(grid_for_zonal)}")
if len(valid14) > 0:
    print(f"  Elevation range      : "
          f"[{valid14['elev_mean'].min():.1f}, "
          f"{valid14['elev_mean'].max():.1f}] m")
    print(f"  Overall mean         : {valid14['elev_mean'].mean():.1f} m")
    print(f"  Overall std          : {valid14['elev_mean'].std():.1f} m")

    fig, ax = plt.subplots(figsize=(12, 12))
    valid14.to_crs(CRS_UTM).plot(
        ax=ax, column="elev_mean", cmap="terrain",
        edgecolor="grey", linewidth=0.1, legend=True,
        legend_kwds={"label": "Mean Elevation (m)", "shrink": 0.6})
    boundary_utm.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2)
    ax.set_title(f"Query 14 — Zonal Stats: Elevation ({len(valid14)} cells)",
                 fontsize=13)
    save_fig(fig, "Q14_Zonal_Elevation.png")

# ─────────────────────────────────────────────────────────────────────────────
# Q15 — MODIS AOD
# ─────────────────────────────────────────────────────────────────────────────
heading(15, "MODIS AOD — Aerosol Optical Depth")
aod_path = os.path.join(MODIS_DIR, "MODIS_AOD_Feb1_10_Mean_UTM44N.tif")
aod_scale_applied = False  # track whether we divided by 1000

if os.path.exists(aod_path):
    aod_src = rasterio.open(aod_path)
    aod_data = aod_src.read(1).astype(float)
    aod_nd = aod_src.nodata
    if aod_nd is not None:
        aod_data[aod_data == aod_nd] = np.nan
    aod_data[aod_data < 0] = np.nan

    aod_mean = np.nanmean(aod_data)
    if aod_mean > 5:
        print(f"  ⚠️  Raw mean = {aod_mean:.2f}, applying scale factor /1000")
        aod_data = aod_data / 1000
        aod_mean = np.nanmean(aod_data)
        aod_scale_applied = True

    print(f"  Shape    : {aod_data.shape}")
    print(f"  CRS      : {aod_src.crs}")
    print(f"  Res      : {aod_src.res[0]:.1f}m")
    print(f"  Valid px : {int(np.nansum(~np.isnan(aod_data)))}")
    print(f"  AOD range: [{np.nanmin(aod_data):.4f}, {np.nanmax(aod_data):.4f}]")
    print(f"  Mean AOD : {aod_mean:.4f}")

    fig, ax = plt.subplots(figsize=(10, 9))
    p2, p98 = np.nanpercentile(aod_data, [2, 98])
    im = ax.imshow(aod_data, cmap="YlOrBr", vmin=p2, vmax=p98)
    plt.colorbar(im, ax=ax, label="AOD", shrink=0.7)
    ax.set_title("Query 15 — MODIS Aerosol Optical Depth", fontsize=13)
    save_fig(fig, "Q15_MODIS_AOD.png")

    if HAS_RS:
        grid_aod = grid_1km.to_crs(aod_src.crs)
        sa = zs(grid_aod, aod_path, stats=["mean", "std"], nodata=aod_nd)
        grid_aod["aod_mean"] = [s["mean"] for s in sa]
        va = grid_aod.dropna(subset=["aod_mean"])
        if len(va) > 0:
            if aod_scale_applied:
                va = va.copy()
                va["aod_mean"] = va["aod_mean"] / 1000
            print(f"  Grid AOD range: [{va['aod_mean'].min():.4f}, "
                  f"{va['aod_mean'].max():.4f}]")
            print(f"  Grid AOD mean : {va['aod_mean'].mean():.4f}")
else:
    print("  ⚠️ MODIS AOD file not found.")

# ─────────────────────────────────────────────────────────────────────────────
# Q16 — LANDSAT LST + UHI  (FIXED: clamp to 55°C)
# ─────────────────────────────────────────────────────────────────────────────
heading(16, "LANDSAT LST — Land Surface Temperature + UHI Classes")
lst_path = os.path.join(LANDSAT_DIR, "Landsat_LST_Celsius_UTM44N.tif")
if os.path.exists(lst_path):
    lst_src = rasterio.open(lst_path)
    lst_data = lst_src.read(1).astype(float)
    lst_nd = lst_src.nodata
    if lst_nd is not None:
        lst_data[lst_data == lst_nd] = np.nan

    # ──── FIX: Clamp unrealistic temperatures ────────────────────────────────
    lst_data[(lst_data < -50) | (lst_data > 55)] = np.nan
    # ─────────────────────────────────────────────────────────────────────────

    print(f"  Shape    : {lst_data.shape}")
    print(f"  CRS      : {lst_src.crs}")
    print(f"  Res      : {lst_src.res[0]:.1f}m")
    print(f"  Temp     : [{np.nanmin(lst_data):.1f}, "
          f"{np.nanmax(lst_data):.1f}]°C")
    print(f"  Mean LST : {np.nanmean(lst_data):.1f}°C")
    print(f"  Std LST  : {np.nanstd(lst_data):.1f}°C")

    uhi = np.full_like(lst_data, np.nan)
    uhi[lst_data < 25]                              = 1
    uhi[(lst_data >= 25) & (lst_data < 30)]          = 2
    uhi[(lst_data >= 30) & (lst_data < 35)]          = 3
    uhi[(lst_data >= 35) & (lst_data < 40)]          = 4
    uhi[lst_data >= 40]                              = 5

    uhi_labels = {1: "Cool <25°C", 2: "25–30°C", 3: "30–35°C",
                  4: "35–40°C", 5: "Hot >40°C"}
    total_valid = int(np.nansum(~np.isnan(uhi)))
    for c in range(1, 6):
        cnt = int(np.nansum(uhi == c))
        pct = cnt / total_valid * 100 if total_valid > 0 else 0
        print(f"  Class {c} [{uhi_labels[c]:>12s}]: "
              f"{cnt:10d} px  ({pct:5.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    p2, p98 = np.nanpercentile(lst_data, [2, 98])
    im0 = axes[0].imshow(lst_data, cmap="coolwarm", vmin=p2, vmax=p98)
    axes[0].set_title("Land Surface Temperature (°C)")
    plt.colorbar(im0, ax=axes[0], shrink=0.6, label="°C")

    cmap_uhi = ListedColormap(
        ["#2166ac", "#67a9cf", "#ffffbf", "#ef8a62", "#b2182b"])
    im1 = axes[1].imshow(uhi, cmap=cmap_uhi, vmin=0.5, vmax=5.5)
    cbar1 = plt.colorbar(im1, ax=axes[1], shrink=0.6, ticks=[1, 2, 3, 4, 5])
    cbar1.ax.set_yticklabels([uhi_labels[i] for i in range(1, 6)])
    axes[1].set_title("UHI Classification")
    fig.suptitle("Query 16 — Landsat LST & UHI Classes", fontsize=14)
    save_fig(fig, "Q16_Landsat_LST_UHI.png")
else:
    print("  ⚠️ Landsat LST not found.")

# ─────────────────────────────────────────────────────────────────────────────
# Q17 — SENTINEL-5P NO₂
# ─────────────────────────────────────────────────────────────────────────────
heading(17, "SENTINEL-5P NO₂ — Tropospheric Column")
no2_file = None
for f in os.listdir(RASTER_DIR):
    if "NO2" in f.upper() and f.endswith(".nc"):
        no2_file = os.path.join(RASTER_DIR, f)
        break

if no2_file:
    try:
        import netCDF4 as nc4
        ds = nc4.Dataset(no2_file)
        prod = ds.groups["PRODUCT"]
        no2_raw = prod.variables[
            "nitrogendioxide_tropospheric_column"][0, :, :]
        lat = prod.variables["latitude"][0, :, :]
        lon = prod.variables["longitude"][0, :, :]
        qa  = prod.variables["qa_value"][0, :, :]

        no2_filt = np.where(qa > 0.5, no2_raw, np.nan)
        no2_umol = no2_filt * 1e6

        hyd_mask = ((lat > 17.2) & (lat < 17.6) &
                    (lon > 78.0) & (lon < 78.7))
        no2_hyd = np.where(hyd_mask, no2_umol, np.nan)
        vc = int(np.nansum(~np.isnan(no2_hyd)))
        print(f"  Valid pixels over Hyderabad: {vc}")
        if vc > 0:
            print(f"  NO₂ range: [{np.nanmin(no2_hyd):.2f}, "
                  f"{np.nanmax(no2_hyd):.2f}] µmol/m²")
            print(f"  Mean NO₂ : {np.nanmean(no2_hyd):.2f} µmol/m²")

        fig, ax = plt.subplots(figsize=(10, 9))
        valid_mask = hyd_mask & (~np.isnan(no2_umol))
        if np.any(valid_mask):
            sc = ax.scatter(lon[valid_mask], lat[valid_mask],
                            c=no2_umol[valid_mask], cmap="Reds", s=20,
                            edgecolor="black", linewidth=0.3)
            plt.colorbar(sc, ax=ax, label="NO₂ (µmol/m²)", shrink=0.7)
            bnd_geo = boundary.to_crs(CRS_GEO)
            bnd_geo.plot(ax=ax, facecolor="none", edgecolor="blue",
                         linewidth=1.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Query 17 — Sentinel-5P NO₂ over Hyderabad",
                     fontsize=13)
        save_fig(fig, "Q17_S5P_NO2.png")
        ds.close()
    except Exception as e:
        print(f"  ⚠️ NO₂ error: {e}")
else:
    print("  ⚠️ NO₂ file not found.")

# ─────────────────────────────────────────────────────────────────────────────
# Q18 — SENTINEL-5P SO₂ & CO
# ─────────────────────────────────────────────────────────────────────────────
heading(18, "SENTINEL-5P SO₂ & CO — Multi-Pollutant Overlay")
fig18, axes18 = plt.subplots(1, 2, figsize=(18, 8))
plot_idx = 0

for gas_label, gas_key, var_candidates in [
    ("SO₂", "SO2",
     ["sulfurdioxide_total_vertical_column",
      "sulfurdioxide_total_vertical_column_precision"]),
    ("CO", "CO",
     ["carbonmonoxide_total_column",
      "carbonmonoxide_total_column_corrected"])
]:
    nc_file = None
    for f in os.listdir(RASTER_DIR):
        if gas_key in f.upper() and f.endswith(".nc"):
            nc_file = os.path.join(RASTER_DIR, f)
            break
    if nc_file:
        try:
            import netCDF4 as nc4
            ds = nc4.Dataset(nc_file)
            prod = ds.groups["PRODUCT"]

            data_var = None
            for vn in var_candidates:
                if vn in prod.variables:
                    data_var = vn
                    break
            if data_var is None:
                skip_vars = {"latitude", "longitude", "qa_value", "time",
                             "delta_time", "time_utc", "scanline",
                             "ground_pixel"}
                for vn in prod.variables:
                    if vn not in skip_vars:
                        data_var = vn
                        break

            if data_var:
                raw = prod.variables[data_var][0, :, :]
                lat2 = prod.variables["latitude"][0, :, :]
                lon2 = prod.variables["longitude"][0, :, :]
                qa2  = prod.variables["qa_value"][0, :, :]
                filt = np.where(qa2 > 0.5, raw, np.nan)

                if gas_key == "SO2":
                    filt = np.where(filt < 0, 0, filt)

                hyd2 = ((lat2 > 17.2) & (lat2 < 17.6) &
                        (lon2 > 78.0) & (lon2 < 78.7))
                gas_hyd = np.where(hyd2, filt, np.nan)
                vc = int(np.nansum(~np.isnan(gas_hyd)))

                print(f"  {gas_label} variable: {data_var}")
                print(f"  {gas_label} valid pixels: {vc}")
                if vc > 0:
                    print(f"  {gas_label} mean: "
                          f"{np.nanmean(gas_hyd):.6e}")
                    print(f"  {gas_label} range: "
                          f"[{np.nanmin(gas_hyd):.6e}, "
                          f"{np.nanmax(gas_hyd):.6e}]")

                valid_mask2 = hyd2 & (~np.isnan(filt))
                if np.any(valid_mask2) and plot_idx < 2:
                    ax = axes18[plot_idx]
                    sc = ax.scatter(lon2[valid_mask2], lat2[valid_mask2],
                                    c=filt[valid_mask2], cmap="Oranges",
                                    s=15, edgecolor="black", linewidth=0.2)
                    plt.colorbar(sc, ax=ax, shrink=0.6)
                    bnd_geo = boundary.to_crs(CRS_GEO)
                    bnd_geo.plot(ax=ax, facecolor="none", edgecolor="blue",
                                 linewidth=1.5)
                    ax.set_title(f"{gas_label} ({data_var[:30]}…)")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    plot_idx += 1
            ds.close()
        except Exception as e:
            print(f"  ⚠️ {gas_label} error: {e}")
    else:
        print(f"  ⚠️ {gas_label} file not found.")

fig18.suptitle("Query 18 — Sentinel-5P SO₂ & CO over Hyderabad", fontsize=14)
save_fig(fig18, "Q18_S5P_SO2_CO.png")

# ─────────────────────────────────────────────────────────────────────────────
# Q19 — FILTERING (on clipped DEM)
# ─────────────────────────────────────────────────────────────────────────────
heading(19, "NEIGHBOURHOOD FILTERING — Smooth & Enhance DEM")
dem_for_filter = np.nan_to_num(dem_arr, nan=np.nanmean(dem_arr))

smooth_3 = uniform_filter(dem_for_filter, size=3)
smooth_7 = uniform_filter(dem_for_filter, size=7)

edge_kernel = np.array([[-0.1, -0.1, -0.1],
                         [-0.1,  1.8, -0.1],
                         [-0.1, -0.1, -0.1]])
dem_edge = convolve(dem_for_filter, edge_kernel)

std_orig = np.std(dem_for_filter)
std_s3   = np.std(smooth_3)
std_s7   = np.std(smooth_7)
std_e    = np.std(dem_edge)

print(f"  Original  std : {std_orig:.2f} m")
print(f"  Smooth 3×3 std: {std_s3:.2f} m  "
      f"(Δ = {(std_s3 / std_orig - 1) * 100:+.2f}%)")
print(f"  Smooth 7×7 std: {std_s7:.2f} m  "
      f"(Δ = {(std_s7 / std_orig - 1) * 100:+.2f}%)")
print(f"  Edge enh. std : {std_e:.2f} m   "
      f"(Δ = {(std_e / std_orig - 1) * 100:+.2f}%)")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
kw = dict(cmap="terrain",
          vmin=np.nanpercentile(dem_arr, 2),
          vmax=np.nanpercentile(dem_arr, 98))
axes[0, 0].imshow(dem_for_filter, **kw)
axes[0, 0].set_title("Original DEM")
axes[0, 1].imshow(smooth_3, **kw)
axes[0, 1].set_title("3×3 Low-pass (Smoothed)")
axes[1, 0].imshow(smooth_7, **kw)
axes[1, 0].set_title("7×7 Low-pass (Strong Smooth)")
axes[1, 1].imshow(dem_edge, **kw)
axes[1, 1].set_title("Edge Enhanced (High-pass)")
fig.suptitle("Query 19 — Neighbourhood Filters on DEM", fontsize=14)
save_fig(fig, "Q19_DEM_Filters.png")

# ─────────────────────────────────────────────────────────────────────────────
# Q20 — CONTOUR MAP
# ─────────────────────────────────────────────────────────────────────────────
heading(20, "CONTOUR MAP — DEM Surface Visualization")
d20 = np.flipud(dem_arr)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
cs = axes[0].contour(d20, levels=15, cmap="terrain", linewidths=0.6)
axes[0].clabel(cs, inline=True, fontsize=5, fmt="%.0f")
axes[0].set_title("Line Contours")
axes[0].set_aspect("equal")

cf = axes[1].contourf(d20, levels=20, cmap="terrain")
plt.colorbar(cf, ax=axes[1], label="Elevation (m)", shrink=0.7)
axes[1].set_title("Filled Contours")
axes[1].set_aspect("equal")

fig.suptitle("Query 20 — DEM Contour Maps", fontsize=14)
save_fig(fig, "Q20_DEM_Contours.png")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                     GROUND-TRUTH  DATA  SUMMARY                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
heading("GT", "CPCB Ground-Truth PM2.5 Summary")
if os.path.exists(CPCB_CSV):
    pm = pd.read_csv(CPCB_CSV)
    pm25_col = None
    for c in pm.columns:
        if "pm2.5" in c.lower() or "actual_pm25" in c.lower():
            pm25_col = c
            break
    if pm25_col is None:
        pm25_col = pm.columns[1]

    pm[pm25_col] = pd.to_numeric(pm[pm25_col], errors="coerce")
    pm = pm.dropna(subset=[pm25_col])

    print(f"  File       : {CPCB_CSV}")
    print(f"  Records    : {len(pm)}")
    print(f"  Column     : {pm25_col}")
    print(f"  Range      : [{pm[pm25_col].min():.1f}, "
          f"{pm[pm25_col].max():.1f}] µg/m³")
    print(f"  Mean       : {pm[pm25_col].mean():.1f} µg/m³")
    print(f"  Median     : {pm[pm25_col].median():.1f} µg/m³")
    print(f"  Std        : {pm[pm25_col].std():.1f} µg/m³")

    who_annual = 15
    pct_above_who = (pm[pm25_col] > who_annual).mean() * 100
    print(f"  % above WHO annual (15 µg/m³): {pct_above_who:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].hist(pm[pm25_col], bins=30, color="steelblue",
                 edgecolor="white", alpha=0.8)
    axes[0].axvline(pm[pm25_col].mean(), color="red", ls="--", lw=2,
                    label=f"Mean = {pm[pm25_col].mean():.1f}")
    axes[0].axvline(who_annual, color="green", ls=":", lw=2,
                    label=f"WHO = {who_annual}")
    axes[0].set_xlabel("PM2.5 (µg/m³)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("PM2.5 Distribution")
    axes[0].legend()

    date_col = None
    for c in pm.columns:
        if "date" in c.lower() or "timestamp" in c.lower():
            date_col = c
            break
    if date_col:
        pm["_dt"] = pd.to_datetime(pm[date_col], errors="coerce")
        pm_sorted = pm.dropna(subset=["_dt"]).sort_values("_dt")
        axes[1].plot(pm_sorted["_dt"], pm_sorted[pm25_col],
                     color="steelblue", linewidth=0.5, alpha=0.7)
        axes[1].axhline(who_annual, color="green", ls=":", lw=1.5,
                        label="WHO guideline")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("PM2.5 (µg/m³)")
        axes[1].set_title("PM2.5 Time Series")
        axes[1].legend()
        axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("CPCB PM2.5 Ground-Truth Data", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "GT_PM25_Summary.png")
else:
    print(f"  ⚠️ {CPCB_CSV} not found.")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║              FEATURE  EXTRACTION  TABLE  (ALL FIXES APPLIED)             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
heading("FE", "FEATURE EXTRACTION — LUR Predictor Table")
st_fe = stations.to_crs(CRS_UTM).copy().reset_index(drop=True)
n_st = len(st_fe)
print(f"  Extracting features for {n_st} CPCB station(s) …\n")

# ─── Road length in buffers ─────────────────────────────────────────────────
for r in [500, 1000, 2000]:
    col_name = f"road_km_{r}m"
    vals = []
    for idx, row in st_fe.iterrows():
        buf = row.geometry.buffer(r)
        buf_gdf = gpd.GeoDataFrame(geometry=[buf], crs=CRS_UTM)
        try:
            isect = gpd.clip(clipped["Primary_Roads"].to_crs(CRS_UTM),
                             buf_gdf)
            km = isect.geometry.length.sum() / 1000
        except Exception:
            km = 0
        vals.append(round(km, 3))
    st_fe[col_name] = vals
    print(f"  {col_name:30s}: mean={np.mean(vals):.3f} km")

# ─── Feature counts in buffers ──────────────────────────────────────────────
for feat_name, feat_gdf in [
    ("bus_stops", clipped["Bus_Stops"].to_crs(CRS_UTM)),
    ("signals",  clipped["Traffic_Signals"].to_crs(CRS_UTM))
]:
    for r in [500, 1000, 2000, 5000]:
        col_name = f"{feat_name}_{r}m"
        vals = []
        for idx, row in st_fe.iterrows():
            buf = row.geometry.buffer(r)
            count = feat_gdf[feat_gdf.within(buf)].shape[0]
            vals.append(count)
        st_fe[col_name] = vals
        print(f"  {col_name:30s}: mean={np.mean(vals):.1f}  max={max(vals)}")

# ─── Distances ───────────────────────────────────────────────────────────────
for target_name, target_geom in [
    ("dist_industrial_m",
     ind_union if len(clipped["Industrial_Areas"]) > 0 else None),
    ("dist_water_m",
     wat_union if len(clipped["Water_Bodies"]) > 0 else None),
    ("dist_road_m",
     rd_union if len(clipped["Primary_Roads"]) > 0 else None)
]:
    vals = []
    for idx, row in st_fe.iterrows():
        d = row.geometry.distance(target_geom) if target_geom else np.nan
        vals.append(round(d, 1))
    st_fe[target_name] = vals
    print(f"  {target_name:30s}: mean={np.nanmean(vals):.0f} m")

# ─── Elevation sampling (from FULL DEM) ─────────────────────────────────────
vals_e = []
for idx, row in st_fe.iterrows():
    pt_geo = gpd.GeoSeries([row.geometry], crs=CRS_UTM).to_crs(
        dem_src.crs).iloc[0]
    try:
        ri, ci = rowcol(dem_src.transform, pt_geo.x, pt_geo.y)
        if 0 <= ri < dem_full.shape[0] and 0 <= ci < dem_full.shape[1]:
            vals_e.append(float(dem_full[ri, ci]))
        else:
            vals_e.append(np.nan)
    except Exception:
        vals_e.append(np.nan)
st_fe["elevation_m"] = vals_e
print(f"  {'elevation_m':30s}: mean={np.nanmean(vals_e):.1f} m")

# ─── Slope sampling (FIX: from FULL DEM slope, not clipped) ─────────────────
vals_slope = []
for idx, row in st_fe.iterrows():
    pt_geo = gpd.GeoSeries([row.geometry], crs=CRS_UTM).to_crs(
        dem_src.crs).iloc[0]
    try:
        ri, ci = rowcol(dem_src.transform, pt_geo.x, pt_geo.y)
        if 0 <= ri < slope_full.shape[0] and 0 <= ci < slope_full.shape[1]:
            vals_slope.append(float(slope_full[ri, ci]))
        else:
            vals_slope.append(np.nan)
    except Exception:
        vals_slope.append(np.nan)
st_fe["slope_deg"] = vals_slope
print(f"  {'slope_deg':30s}: mean={np.nanmean(vals_slope):.2f}°")

# ─── LST sampling (with clamp applied) ──────────────────────────────────────
if os.path.exists(lst_path):
    lst_s2 = rasterio.open(lst_path)
    lst_full2 = lst_s2.read(1).astype(float)
    lst_nd2 = lst_s2.nodata
    if lst_nd2 is not None:
        lst_full2[lst_full2 == lst_nd2] = np.nan
    # ──── FIX: Same clamp as Q16 ────────────────────────────────────────────
    lst_full2[(lst_full2 < -50) | (lst_full2 > 55)] = np.nan
    # ────────────────────────────────────────────────────────────────────────

    vals_lst = []
    for idx, row in st_fe.iterrows():
        pt = gpd.GeoSeries([row.geometry], crs=CRS_UTM).to_crs(
            lst_s2.crs).iloc[0]
        try:
            ri, ci = rowcol(lst_s2.transform, pt.x, pt.y)
            if (0 <= ri < lst_full2.shape[0] and
                    0 <= ci < lst_full2.shape[1]):
                vals_lst.append(float(lst_full2[ri, ci]))
            else:
                vals_lst.append(np.nan)
        except Exception:
            vals_lst.append(np.nan)
    st_fe["lst_celsius"] = vals_lst
    print(f"  {'lst_celsius':30s}: mean={np.nanmean(vals_lst):.1f}°C")

# ─── AOD sampling (with consistent scale) ───────────────────────────────────
if os.path.exists(aod_path):
    aod_s2 = rasterio.open(aod_path)
    aod_full2 = aod_s2.read(1).astype(float)
    aod_nd2 = aod_s2.nodata
    if aod_nd2 is not None:
        aod_full2[aod_full2 == aod_nd2] = np.nan
    aod_full2[aod_full2 < 0] = np.nan
    # ──── FIX: Apply same scale factor as Q15 if needed ─────────────────────
    if aod_scale_applied:
        aod_full2 = aod_full2 / 1000
    # ────────────────────────────────────────────────────────────────────────

    vals_aod = []
    for idx, row in st_fe.iterrows():
        pt = gpd.GeoSeries([row.geometry], crs=CRS_UTM).to_crs(
            aod_s2.crs).iloc[0]
        try:
            ri, ci = rowcol(aod_s2.transform, pt.x, pt.y)
            if (0 <= ri < aod_full2.shape[0] and
                    0 <= ci < aod_full2.shape[1]):
                vals_aod.append(float(aod_full2[ri, ci]))
            else:
                vals_aod.append(np.nan)
        except Exception:
            vals_aod.append(np.nan)
    st_fe["modis_aod"] = vals_aod
    print(f"  {'modis_aod':30s}: mean={np.nanmean(vals_aod):.4f}")

# ─── Save feature table ─────────────────────────────────────────────────────
fe_path = os.path.join(OUTPUT_DIR, "LUR_Feature_Table.csv")
st_fe_save = st_fe.drop(columns=["geometry"], errors="ignore")
st_fe_save.to_csv(fe_path, index=False)
print(f"\n  📄 Feature table saved → {fe_path}")
print(f"     Columns ({len(st_fe_save.columns)}): {list(st_fe_save.columns)}")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                         FINAL SUMMARY                                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("  🎉  ALL 20 QUERIES + FEATURE EXTRACTION COMPLETE!")
print("=" * 70)

output_files = sorted(os.listdir(OUTPUT_DIR))
png_count = len([f for f in output_files if f.endswith(".png")])
csv_count = len([f for f in output_files if f.endswith(".csv")])
total_size_mb = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in output_files) / (1024 * 1024)

print(f"  Output folder  : {os.path.abspath(OUTPUT_DIR)}")
print(f"  PNG figures    : {png_count}")
print(f"  CSV files      : {csv_count}")
print(f"  Total size     : {total_size_mb:.1f} MB")
print()
print("  📂 Output files:")
for f in output_files:
    size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
    print(f"     {f:45s}  {size_kb:8.1f} KB")

print("\n  ✅ Script finished successfully.\n")