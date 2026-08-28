"""
risk_engine.py
==============

Core geospatial / hydrodynamic engine for the Global GLOF (Glacial Lake Outburst
Flood) Risk Assessment web app.

Responsibilities
----------------
1. ``load_global_glacier_data()``   -- fetch + cache a worldwide glacial-lake
   inventory (Zenodo Global GLOF Database / GLIMS style schema).
2. Hydrodynamic scaling            -- empirical volume & peak-discharge relations.
3. ``assess_global_location_risk`` -- R-tree accelerated spatial query that finds
   the nearest hazardous lake, computes the runout ratio ``H / L`` toward the
   nearest glaciated basin (Andes, Alps, Himalayas, Alaska, ...) and returns a
   categorical global Risk Score.

The module is deliberately dependency-light at import time: heavy network calls
only happen inside ``load_global_glacier_data`` and every function degrades
gracefully to a bundled synthetic inventory when offline.
"""

from __future__ import annotations

import io
import math
import os
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOCAL_CACHE = os.path.join(DATA_DIR, "global_glacial_lakes.csv")

# Public, open, worldwide glacial-lake inventories. The loader tries each URL in
# order and falls back to the bundled synthetic sample if all are unreachable.
REMOTE_SOURCES = [
    # Zenodo - Global Glacial Lake Database (schema-compatible CSV export).
    "https://zenodo.org/record/8188392/files/global_glacial_lakes.csv",
    # GLIMS / NSIDC mirror (placeholder path - kept for documentation).
    "https://www.glims.org/download/glims_lakes_global.csv",
]

REQUIRED_FIELDS = [
    "lake_id",
    "country",
    "latitude",
    "longitude",
    "elevation_m",
    "surface_area_m2",
]

# Representative outlet / valley-floor elevations (m) and centroids for the
# world's principal glaciated basins. Used to estimate the elevation drop (H)
# and the horizontal runout distance (L).
GLACIATED_BASINS: Dict[str, Dict[str, float]] = {
    "Himalayas":        {"lat": 28.00, "lon": 86.90, "outlet_elevation_m": 1500.0},
    "Karakoram":        {"lat": 35.88, "lon": 76.51, "outlet_elevation_m": 2500.0},
    "Central Asia":     {"lat": 42.20, "lon": 78.50, "outlet_elevation_m": 1600.0},
    "European Alps":    {"lat": 46.00, "lon": 8.00,  "outlet_elevation_m": 500.0},
    "Andes":            {"lat": -13.50, "lon": -72.00, "outlet_elevation_m": 3000.0},
    "Patagonia":        {"lat": -49.30, "lon": -73.00, "outlet_elevation_m": 200.0},
    "Alaska/Pacific NW": {"lat": 61.00, "lon": -145.00, "outlet_elevation_m": 100.0},
    "Iceland":          {"lat": 64.40, "lon": -17.30, "outlet_elevation_m": 50.0},
    "Scandinavia":      {"lat": 61.60, "lon": 7.00,  "outlet_elevation_m": 300.0},
}

EARTH_RADIUS_M = 6_371_008.8


# --------------------------------------------------------------------------- #
# 1. Global inventory loader
# --------------------------------------------------------------------------- #

def _synthetic_inventory(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic worldwide glacial-lake inventory (offline fallback)."""
    rng = np.random.default_rng(seed)
    rows = []
    basin_country = {
        "Himalayas": ["Nepal", "India", "Bhutan", "China"],
        "Karakoram": ["Pakistan", "China"],
        "Central Asia": ["Kyrgyzstan", "Kazakhstan", "Tajikistan"],
        "European Alps": ["Switzerland", "France", "Italy", "Austria"],
        "Andes": ["Peru", "Chile", "Bolivia", "Argentina", "Ecuador"],
        "Patagonia": ["Chile", "Argentina"],
        "Alaska/Pacific NW": ["United States", "Canada"],
        "Iceland": ["Iceland"],
        "Scandinavia": ["Norway", "Sweden"],
    }
    lid = 0
    for basin, meta in GLACIATED_BASINS.items():
        k = n // len(GLACIATED_BASINS)
        lat = meta["lat"] + rng.normal(0, 1.4, k)
        lon = meta["lon"] + rng.normal(0, 1.4, k)
        # Lognormal surface areas: most lakes small, a few very large.
        area = rng.lognormal(mean=11.0, sigma=1.3, size=k)  # ~6e4 m^2 median
        elev = meta["outlet_elevation_m"] + rng.uniform(800, 2600, k)
        for i in range(k):
            lid += 1
            rows.append(
                {
                    "lake_id": f"GL{lid:05d}",
                    "country": rng.choice(basin_country[basin]),
                    "latitude": round(float(lat[i]), 5),
                    "longitude": round(float(lon[i]), 5),
                    "elevation_m": round(float(elev[i]), 1),
                    "surface_area_m2": round(float(area[i]), 1),
                    "basin": basin,
                }
            )
    return pd.DataFrame(rows)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary inventory dataframe to the required schema."""
    colmap = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    out = pd.DataFrame()
    out["lake_id"] = df[pick("lake_id", "glof_id", "id", "glims_id")] \
        if pick("lake_id", "glof_id", "id", "glims_id") else [f"GL{i:05d}" for i in range(len(df))]
    out["country"] = df[pick("country", "nation", "region")] if pick("country", "nation", "region") else "Unknown"
    out["latitude"] = pd.to_numeric(df[pick("latitude", "lat", "y")], errors="coerce")
    out["longitude"] = pd.to_numeric(df[pick("longitude", "lon", "lng", "x")], errors="coerce")
    out["elevation_m"] = pd.to_numeric(
        df[pick("elevation_m", "elevation", "elev", "dem_m")], errors="coerce"
    ) if pick("elevation_m", "elevation", "elev", "dem_m") else np.nan
    area_col = pick("surface_area_m2", "area_m2", "lake_area_m2", "area")
    out["surface_area_m2"] = pd.to_numeric(df[area_col], errors="coerce") if area_col else np.nan

    out = out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    out["elevation_m"] = out["elevation_m"].fillna(out["elevation_m"].median())
    out["surface_area_m2"] = out["surface_area_m2"].fillna(out["surface_area_m2"].median())
    return out


@lru_cache(maxsize=1)
def load_global_glacier_data() -> gpd.GeoDataFrame:
    """
    Fetch and cache a worldwide glacial-lake inventory as a GeoDataFrame.

    Resolution order:
        1. Local CSV cache (``data/global_glacial_lakes.csv``).
        2. Remote open inventories in ``REMOTE_SOURCES``.
        3. Bundled deterministic synthetic inventory.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: ``lake_id, country, latitude, longitude, elevation_m,
        surface_area_m2, basin, water_volume_m3, peak_discharge_m3s, geometry``
        CRS = EPSG:4326.
    """
    df: Optional[pd.DataFrame] = None

    # 1. local cache
    if os.path.exists(LOCAL_CACHE):
        try:
            df = _normalise(pd.read_csv(LOCAL_CACHE))
        except Exception:
            df = None

    # 2. remote sources
    if df is None or df.empty:
        for url in REMOTE_SOURCES:
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                df = _normalise(pd.read_csv(io.StringIO(resp.text)))
                if not df.empty:
                    os.makedirs(DATA_DIR, exist_ok=True)
                    df.to_csv(LOCAL_CACHE, index=False)
                    break
            except Exception:
                df = None
                continue

    # 3. synthetic fallback
    if df is None or df.empty:
        df = _synthetic_inventory()
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(LOCAL_CACHE, index=False)

    if "basin" not in df.columns:
        df["basin"] = [
            nearest_basin(lat, lon)[0] for lat, lon in zip(df["latitude"], df["longitude"])
        ]

    # Hydrodynamic attributes
    df["water_volume_m3"] = water_volume_from_area(df["surface_area_m2"])
    df["peak_discharge_m3s"] = peak_discharge_from_volume(df["water_volume_m3"])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    return gdf


# --------------------------------------------------------------------------- #
# 2. Hydrodynamic scaling relations
# --------------------------------------------------------------------------- #

def water_volume_from_area(surface_area_m2):
    """
    Empirical lake water volume from surface area.

        V = 0.035 * (A ** 1.29)          [m^3]

    Accepts scalars, numpy arrays or pandas Series.
    """
    area = np.asarray(surface_area_m2, dtype="float64")
    vol = 0.035 * np.power(np.clip(area, 0, None), 1.29)
    if np.isscalar(surface_area_m2) or getattr(surface_area_m2, "ndim", 1) == 0:
        return float(vol)
    return vol


def peak_discharge_from_volume(water_volume_m3):
    """
    Empirical peak breach discharge from impounded volume.

        Q = 0.00013 * (V ** 0.60)        [m^3 / s]
    """
    vol = np.asarray(water_volume_m3, dtype="float64")
    q = 0.00013 * np.power(np.clip(vol, 0, None), 0.60)
    if np.isscalar(water_volume_m3) or getattr(water_volume_m3, "ndim", 1) == 0:
        return float(q)
    return q


# --------------------------------------------------------------------------- #
# Geodesy helpers
# --------------------------------------------------------------------------- #

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def nearest_basin(lat: float, lon: float):
    """Return ``(basin_name, distance_m, basin_meta)`` for the closest glaciated basin."""
    best_name, best_dist, best_meta = None, float("inf"), None
    for name, meta in GLACIATED_BASINS.items():
        d = float(haversine_m(lat, lon, meta["lat"], meta["lon"]))
        if d < best_dist:
            best_name, best_dist, best_meta = name, d, meta
    return best_name, best_dist, best_meta


# --------------------------------------------------------------------------- #
# 3. Spatial risk assessment (R-tree accelerated)
# --------------------------------------------------------------------------- #

def _risk_category(hl_ratio: float, distance_m: float, peak_q: float) -> str:
    """
    Combine runout ratio (H/L), proximity and breach discharge into a
    categorical global risk score.

    Higher H/L  -> more potential energy per unit runout -> more destructive.
    """
    score = 0

    # Runout / mobility term
    if hl_ratio >= 0.20:
        score += 2
    elif hl_ratio >= 0.10:
        score += 1

    # Proximity term
    if distance_m <= 5_000:
        score += 2
    elif distance_m <= 20_000:
        score += 1

    # Magnitude term
    if peak_q >= 5_000:
        score += 2
    elif peak_q >= 1_000:
        score += 1

    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def assess_global_location_risk(
    user_lat: float,
    user_lon: float,
    user_elevation: float,
    search_radius_km: float = 50.0,
    gdf: Optional[gpd.GeoDataFrame] = None,
) -> dict:
    """
    Rapidly scan the global glacial-lake inventory with an R-tree spatial index
    and assess GLOF exposure at an arbitrary point on Earth.

    Parameters
    ----------
    user_lat, user_lon : float
        Location of interest (WGS-84 degrees).
    user_elevation : float
        Ground elevation at the location of interest (m). Used for the
        elevation drop ``H`` between the hazardous lake and the user.
    search_radius_km : float
        Radius within which lakes are considered "local" hazards.
    gdf : GeoDataFrame, optional
        Pre-loaded inventory; defaults to :func:`load_global_glacier_data`.

    Returns
    -------
    dict
        {
          risk_score, nearest_basin, basin_distance_km,
          nearest_lake_id, nearest_lake_country, nearest_lake_distance_km,
          elevation_drop_H_m, runout_length_L_m, runout_ratio_HL,
          lake_surface_area_m2, lake_water_volume_m3, lake_peak_discharge_m3s,
          lakes_within_radius
        }
    """
    if gdf is None:
        gdf = load_global_glacier_data()

    if gdf.empty:
        return {"risk_score": "Low", "error": "empty inventory"}

    # --- R-tree spatial index over lake point geometries --------------------
    geoms = list(gdf.geometry.values)
    tree = STRtree(geoms)
    user_pt = Point(user_lon, user_lat)

    # Degrees buffer that safely encloses the search radius (+ a margin).
    deg_buffer = (search_radius_km * 1000.0) / 111_320.0 * 1.3
    query_env = user_pt.buffer(deg_buffer)

    idx = tree.query(query_env)  # candidate indices (fast, bbox-level)
    if len(idx) == 0:
        # Fall back to the whole inventory if nothing is nearby.
        idx = np.arange(len(gdf))

    cand = gdf.iloc[np.asarray(idx)].copy()
    cand["distance_m"] = haversine_m(
        user_lat, user_lon, cand["latitude"].values, cand["longitude"].values
    )
    cand = cand.sort_values("distance_m")

    within = cand[cand["distance_m"] <= search_radius_km * 1000.0]
    lakes_within_radius = int(len(within))

    # Nearest *hazardous* lake: prefer those above the user (positive H),
    # otherwise just the nearest lake.
    above = cand[cand["elevation_m"] > user_elevation]
    nearest = (above.iloc[0] if not above.empty else cand.iloc[0])

    nb_name, nb_dist, _ = nearest_basin(user_lat, user_lon)

    L = float(max(nearest["distance_m"], 1.0))                 # runout length
    H = float(max(nearest["elevation_m"] - user_elevation, 0.0))  # elevation drop
    hl = H / L

    result = {
        "risk_score": _risk_category(hl, L, float(nearest["peak_discharge_m3s"])),
        "nearest_basin": nb_name,
        "basin_distance_km": round(nb_dist / 1000.0, 1),
        "nearest_lake_id": str(nearest["lake_id"]),
        "nearest_lake_country": str(nearest["country"]),
        "nearest_lake_lat": float(nearest["latitude"]),
        "nearest_lake_lon": float(nearest["longitude"]),
        "nearest_lake_distance_km": round(L / 1000.0, 2),
        "elevation_drop_H_m": round(H, 1),
        "runout_length_L_m": round(L, 1),
        "runout_ratio_HL": round(hl, 4),
        "lake_surface_area_m2": float(nearest["surface_area_m2"]),
        "lake_water_volume_m3": float(nearest["water_volume_m3"]),
        "lake_peak_discharge_m3s": float(nearest["peak_discharge_m3s"]),
        "lakes_within_radius": lakes_within_radius,
    }
    return result


if __name__ == "__main__":  # quick smoke test
    g = load_global_glacier_data()
    print(f"Loaded {len(g):,} lakes across {g['basin'].nunique()} basins")
    demo = assess_global_location_risk(27.88, 86.87, 4200.0)  # Khumbu, Nepal
    for k, v in demo.items():
        print(f"  {k:26s} {v}")
