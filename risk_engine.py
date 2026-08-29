"""
risk_engine.py
==============

Core geospatial / hydrodynamic engine for the Global GLOF (Glacial Lake Outburst
Flood) Risk Assessment web app.

Responsibilities
----------------
1. ``load_global_glacier_data()``   -- fetch + cache the worldwide GLOF inventory
   (Zenodo *Glacier Lake Outburst Flood Database V3.0*, Veh et al. 2023), with
   real ground elevations from the Copernicus GLO-90 DEM (via the Open-Meteo
   elevation API).
2. Hydrodynamic scaling            -- empirical volume & peak-discharge relations,
   used only where the database does not report a measured value.
3. ``assess_global_location_risk`` -- R-tree accelerated spatial query that finds
   the nearest documented outburst site, computes the runout ratio ``H / L``
   toward the nearest glaciated basin (Andes, Alps, Himalayas, Alaska, ...) and
   returns a categorical global Risk Score.

Heavy work (spreadsheet parse + DEM lookups) happens once in
``build_inventory()``; the result is cached to ``data/global_glof_inventory.csv``
and committed to the repo, so normal runs are instant and offline-safe. A
deterministic synthetic inventory is the last-resort fallback.

Data source
-----------
Veh, G., Lützow, N., Kharlamova, V., Petrakov, D., Hugonnet, R., Korup, O.
(2023). *Trends, Breaching, and Drainage of the World's Glacier Lakes.*
J. Geophys. Res. Earth Surface. Database: https://doi.org/10.5281/zenodo.7330345
DEM: Copernicus GLO-90 via https://open-meteo.com/en/docs/elevation-api
"""

from __future__ import annotations

import os
import time
from datetime import datetime
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
INVENTORY_CSV = os.path.join(DATA_DIR, "global_glof_inventory.csv")
ODS_CACHE = os.path.join(DATA_DIR, "glofdatabase_V3.ods")

# Zenodo record 7330345 - Glacier Lake Outburst Flood Database V3.0
ZENODO_ODS_URL = (
    "https://zenodo.org/api/records/7330345/files/glofdatabase_V3.ods/content"
)
ELEVATION_API = "https://api.open-meteo.com/v1/elevation"

SCHEMA_FIELDS = [
    "lake_id",
    "country",
    "latitude",
    "longitude",
    "elevation_m",
    "surface_area_m2",
]

# Representative outlet / valley-floor elevations (m) and centroids for the
# world's principal glaciated basins. Used for the "nearest glaciated basin"
# read-out and as a coarse outlet elevation where the DEM lookup is missing.
GLACIATED_BASINS: Dict[str, Dict[str, float]] = {
    "Himalayas":         {"lat": 28.00, "lon": 86.90, "outlet_elevation_m": 1500.0},
    "Karakoram":         {"lat": 35.88, "lon": 76.51, "outlet_elevation_m": 2500.0},
    "Central Asia":      {"lat": 42.20, "lon": 78.50, "outlet_elevation_m": 1600.0},
    "European Alps":     {"lat": 46.00, "lon": 8.00,  "outlet_elevation_m": 500.0},
    "Andes":             {"lat": -13.50, "lon": -72.00, "outlet_elevation_m": 3000.0},
    "Patagonia":         {"lat": -49.30, "lon": -73.00, "outlet_elevation_m": 200.0},
    "Alaska/Pacific NW": {"lat": 61.00, "lon": -145.00, "outlet_elevation_m": 100.0},
    "NW North America":  {"lat": 60.00, "lon": -140.00, "outlet_elevation_m": 100.0},
    "Iceland":           {"lat": 64.40, "lon": -17.30, "outlet_elevation_m": 50.0},
    "Greenland":         {"lat": 67.00, "lon": -50.00, "outlet_elevation_m": 20.0},
    "Scandinavia":       {"lat": 61.60, "lon": 7.00,  "outlet_elevation_m": 300.0},
}

EARTH_RADIUS_M = 6_371_008.8


# --------------------------------------------------------------------------- #
# 1. Global inventory - build + load
# --------------------------------------------------------------------------- #

def _fetch_elevations(lats, lons, batch: int = 100, pause: float = 0.3) -> np.ndarray:
    """Copernicus GLO-90 ground elevation (m) for each (lat, lon) via Open-Meteo."""
    lats = np.asarray(lats, dtype="float64")
    lons = np.asarray(lons, dtype="float64")
    out = np.full(len(lats), np.nan)
    for start in range(0, len(lats), batch):
        sl = slice(start, start + batch)
        try:
            r = requests.get(
                ELEVATION_API,
                params={
                    "latitude": ",".join(f"{v:.5f}" for v in lats[sl]),
                    "longitude": ",".join(f"{v:.5f}" for v in lons[sl]),
                },
                timeout=30,
            )
            r.raise_for_status()
            out[sl] = np.asarray(r.json()["elevation"], dtype="float64")
        except Exception:
            pass
        time.sleep(pause)
    return out


def _clean_glof_dataframe(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate + tidy the regional sheets of the GLOF Database V3.0."""
    frames = []
    for region, df in sheets.items():
        df = df.copy()
        df["region"] = region
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    # Rows 0-1 of every sheet are human-readable column descriptions, not data.
    raw = raw[pd.to_numeric(raw["ID"], errors="coerce").notna()]

    lon = pd.to_numeric(raw["Longitude"], errors="coerce")
    lat = pd.to_numeric(raw["Latitude"], errors="coerce")

    vol_106 = pd.to_numeric(raw.get("Mean_Lake_Volume_VL"), errors="coerce")  # 10^6 m^3
    qp = pd.to_numeric(raw.get("Peak_discharge_Qp"), errors="coerce")          # m^3 / s

    date = pd.to_datetime(raw.get("Date"), errors="coerce")

    def _txt(series):
        return (series.astype("string")
                .str.replace(r"[^\x00-\x7f]+", " ", regex=True)  # drop mojibake
                .str.replace(r"\s+", " ", regex=True)
                .str.strip())

    out = pd.DataFrame({
        "lake_id": "GLOF-" + raw["region"].str.replace(r"\W+", "", regex=True)
                   + "-" + raw["ID"].astype("Int64").astype(str),
        "country": _txt(raw["Country"]),
        "region": raw["region"],
        "lake_name": _txt(raw.get("Lake")),
        "dam_type": _txt(raw.get("Lake_type")).str.lower(),
        "mechanism": _txt(raw.get("Mechanism")).str.lower(),
        "latitude": lat,
        "longitude": lon,
        "outburst_date": date.dt.strftime("%Y-%m-%d"),
        "outburst_year": date.dt.year,
        "reported_lake_volume_m3": vol_106 * 1e6,
        "reported_peak_discharge_m3s": qp,
    })

    out = out.dropna(subset=["latitude", "longitude"])
    out = out[(out["latitude"].between(-90, 90)) & (out["longitude"].between(-180, 180))]
    out = out.drop_duplicates(subset=["lake_id"]).reset_index(drop=True)

    # Surface area: invert V = 0.035 * A^1.29  ->  A = (V / 0.035) ^ (1 / 1.29),
    # only where a measured pre-outburst volume exists.
    out["surface_area_m2"] = np.where(
        out["reported_lake_volume_m3"] > 0,
        np.power(out["reported_lake_volume_m3"] / 0.035, 1.0 / 1.29),
        np.nan,
    )
    return out


def build_inventory(force: bool = False, with_elevation: bool = True) -> pd.DataFrame:
    """
    Download the Zenodo GLOF Database V3.0, tidy it, attach real DEM elevations,
    and cache the result to :data:`INVENTORY_CSV`. Safe to re-run.
    """
    if os.path.exists(INVENTORY_CSV) and not force:
        return pd.read_csv(INVENTORY_CSV)

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ODS_CACHE) or force:
        resp = requests.get(ZENODO_ODS_URL, timeout=120)
        resp.raise_for_status()
        with open(ODS_CACHE, "wb") as fh:
            fh.write(resp.content)

    sheets = pd.read_excel(ODS_CACHE, engine="odf", sheet_name=None)
    df = _clean_glof_dataframe(sheets)

    if with_elevation:
        df["elevation_m"] = _fetch_elevations(df["latitude"].values, df["longitude"].values)
    else:
        df["elevation_m"] = np.nan

    df.to_csv(INVENTORY_CSV, index=False)
    return df


def _synthetic_inventory(n: int = 660, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic inventory - last-resort offline fallback."""
    rng = np.random.default_rng(seed)
    basin_country = {
        "Himalayas": ["Nepal", "India", "Bhutan", "China"],
        "Karakoram": ["Pakistan", "China"],
        "Central Asia": ["Kyrgyzstan", "Kazakhstan", "Tajikistan"],
        "European Alps": ["Switzerland", "France", "Italy", "Austria"],
        "Andes": ["Peru", "Chile", "Bolivia", "Argentina"],
        "Patagonia": ["Chile", "Argentina"],
        "Alaska/Pacific NW": ["United States", "Canada"],
        "Iceland": ["Iceland"],
        "Scandinavia": ["Norway", "Sweden"],
    }
    rows, lid = [], 0
    for basin, countries in basin_country.items():
        meta = GLACIATED_BASINS[basin]
        k = n // len(basin_country)
        lat = meta["lat"] + rng.normal(0, 1.4, k)
        lon = meta["lon"] + rng.normal(0, 1.4, k)
        area = rng.lognormal(mean=11.0, sigma=1.3, size=k)
        elev = meta["outlet_elevation_m"] + rng.uniform(800, 2600, k)
        for i in range(k):
            lid += 1
            rows.append({
                "lake_id": f"SYNTH-{lid:05d}",
                "country": rng.choice(countries),
                "region": basin,
                "lake_name": None,
                "dam_type": rng.choice(["moraine", "ice", "bedrock"]),
                "mechanism": None,
                "latitude": round(float(lat[i]), 5),
                "longitude": round(float(lon[i]), 5),
                "outburst_date": None,
                "outburst_year": np.nan,
                "reported_lake_volume_m3": np.nan,
                "reported_peak_discharge_m3s": np.nan,
                "surface_area_m2": round(float(area[i]), 1),
                "elevation_m": round(float(elev[i]), 1),
            })
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def load_global_glacier_data() -> gpd.GeoDataFrame:
    """
    Load the worldwide GLOF inventory as a GeoDataFrame.

    Resolution order:
        1. Cached CSV ``data/global_glof_inventory.csv`` (shipped with the repo).
        2. Freshly built from the Zenodo GLOF Database V3.0 (+ DEM elevations).
        3. Deterministic synthetic inventory (offline fallback).

    Returns a GeoDataFrame (EPSG:4326) with columns::

        lake_id country region lake_name dam_type mechanism
        latitude longitude elevation_m outburst_date outburst_year
        surface_area_m2 reported_lake_volume_m3 reported_peak_discharge_m3s
        basin water_volume_m3 peak_discharge_m3s discharge_is_reported geometry
    """
    df: Optional[pd.DataFrame] = None

    if os.path.exists(INVENTORY_CSV):
        try:
            df = pd.read_csv(INVENTORY_CSV)
        except Exception:
            df = None

    if df is None or df.empty:
        try:
            df = build_inventory()
        except Exception:
            df = None

    if df is None or df.empty:
        df = _synthetic_inventory()

    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    # Nearest glaciated basin per site (for display / filtering / fallback elev).
    df["basin"] = [nearest_basin(la, lo)[0] for la, lo in zip(df["latitude"], df["longitude"])]

    # Fill missing ground elevations with the basin's typical outlet elevation.
    basin_elev = df["basin"].map(lambda b: GLACIATED_BASINS.get(b, {}).get("outlet_elevation_m", 1000.0))
    df["elevation_m"] = pd.to_numeric(df.get("elevation_m"), errors="coerce").fillna(basin_elev)

    # Water volume: measured where available, else scaling law from area.
    rep_vol = pd.to_numeric(df.get("reported_lake_volume_m3"), errors="coerce")
    area = pd.to_numeric(df.get("surface_area_m2"), errors="coerce")
    df["water_volume_m3"] = rep_vol.where(rep_vol > 0, water_volume_from_area(area))

    # Peak discharge: measured where available, else scaling law from volume.
    rep_q = pd.to_numeric(df.get("reported_peak_discharge_m3s"), errors="coerce")
    df["discharge_is_reported"] = rep_q > 0
    df["peak_discharge_m3s"] = rep_q.where(rep_q > 0, peak_discharge_from_volume(df["water_volume_m3"]))

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
    """Combine proximity, runout ratio (H/L) and breach discharge into a score.

    Proximity gates everything: a documented outburst far away is not a local
    risk no matter how large it was.
    """
    if distance_m > 60_000:
        return "Low"

    score = 0

    if distance_m <= 5_000:
        score += 2
    elif distance_m <= 20_000:
        score += 1

    if hl_ratio >= 0.20:
        score += 2
    elif hl_ratio >= 0.10:
        score += 1

    # Flood magnitude only matters if the flood could plausibly reach here.
    if distance_m <= 30_000 and np.isfinite(peak_q):
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
    Scan the global GLOF inventory with an R-tree spatial index and assess
    outburst-flood exposure at an arbitrary point on Earth.

    Parameters
    ----------
    user_lat, user_lon : float
        Location of interest (WGS-84 degrees).
    user_elevation : float
        Ground elevation at the location of interest (m), for the elevation drop
        ``H`` between the hazardous lake and the user.
    search_radius_km : float
        Radius within which sites are considered "local" hazards.
    gdf : GeoDataFrame, optional
        Pre-loaded inventory; defaults to :func:`load_global_glacier_data`.

    Returns
    -------
    dict
    """
    if gdf is None:
        gdf = load_global_glacier_data()

    if gdf.empty:
        return {"risk_score": "Low", "error": "empty inventory"}

    # --- R-tree spatial index over site point geometries -------------------
    # Query a generous buffer so we always have real neighbours to rank; fall
    # back to the whole inventory only when nothing is anywhere near.
    tree = STRtree(list(gdf.geometry.values))
    user_pt = Point(user_lon, user_lat)

    query_km = max(search_radius_km, 400.0)
    deg_buffer = (query_km * 1000.0) / 111_320.0 * 1.3
    idx = tree.query(user_pt.buffer(deg_buffer))
    if len(idx) == 0:
        idx = np.arange(len(gdf))

    cand = gdf.iloc[np.asarray(idx)].copy()
    cand["distance_m"] = haversine_m(
        user_lat, user_lon, cand["latitude"].values, cand["longitude"].values
    )
    cand = cand.sort_values("distance_m").reset_index(drop=True)

    within = cand[cand["distance_m"] <= search_radius_km * 1000.0]
    lakes_within_radius = int(len(within))

    # The nearest documented site (any elevation) - used for the map line and
    # the "nearest site" read-out.
    nearest = cand.iloc[0]

    # Runout geometry needs a site that sits ABOVE the location and is close
    # enough that its flood could plausibly reach it. If there is no such site,
    # there is no credible documented-GLOF runout path (H/L -> 0).
    runout_cap_m = max(search_radius_km, 150.0) * 1000.0
    runout_pool = cand[(cand["distance_m"] <= runout_cap_m)
                       & (cand["elevation_m"] > user_elevation)]
    runout = runout_pool.iloc[0] if not runout_pool.empty else None

    if runout is not None:
        L = float(max(runout["distance_m"], 1.0))
        H = float(max(runout["elevation_m"] - user_elevation, 0.0))
    else:
        L = float(max(nearest["distance_m"], 1.0))
        H = float(max(nearest["elevation_m"] - user_elevation, 0.0))
    hl = H / L
    peak_q = float(nearest["peak_discharge_m3s"]) if np.isfinite(nearest["peak_discharge_m3s"]) else np.nan
    nearest_m = float(nearest["distance_m"])

    # Nearest glaciated basin: the region label of the nearest documented site
    # (data-driven and accurate), with the coarse centroid list as a fallback.
    nb_name = str(nearest["region"]) if pd.notna(nearest.get("region")) else nearest_basin(user_lat, user_lon)[0]

    return {
        "risk_score": _risk_category(hl, nearest_m, peak_q),
        "nearest_basin": nb_name,
        "basin_distance_km": round(float(nearest["distance_m"]) / 1000.0, 1),
        "nearest_lake_id": str(nearest["lake_id"]),
        "nearest_lake_name": (None if pd.isna(nearest.get("lake_name")) else str(nearest.get("lake_name"))),
        "nearest_lake_country": str(nearest["country"]),
        "nearest_lake_region": str(nearest.get("region", nb_name)),
        "nearest_lake_dam_type": (None if pd.isna(nearest.get("dam_type")) else str(nearest.get("dam_type"))),
        "nearest_lake_last_outburst": (None if pd.isna(nearest.get("outburst_date")) else str(nearest.get("outburst_date"))),
        "nearest_lake_lat": float(nearest["latitude"]),
        "nearest_lake_lon": float(nearest["longitude"]),
        "nearest_lake_distance_km": round(float(nearest["distance_m"]) / 1000.0, 2),
        "runout_from_site_above": runout is not None,
        "elevation_drop_H_m": round(H, 1),
        "runout_length_L_m": round(L, 1),
        "runout_ratio_HL": round(hl, 4),
        "lake_surface_area_m2": (float(nearest["surface_area_m2"]) if np.isfinite(nearest["surface_area_m2"]) else None),
        "lake_water_volume_m3": (float(nearest["water_volume_m3"]) if np.isfinite(nearest["water_volume_m3"]) else None),
        "lake_peak_discharge_m3s": (peak_q if np.isfinite(peak_q) else None),
        "discharge_is_reported": bool(nearest.get("discharge_is_reported", False)),
        "lakes_within_radius": lakes_within_radius,
    }


# --------------------------------------------------------------------------- #
# 4. Location hazard index + global ranking
# --------------------------------------------------------------------------- #

# Impounding-dam material -> relative breach vulnerability (0-100). More
# specific keys first: substrings are matched in order.
DAM_VULNERABILITY = {
    "supraglacial": 88, "ice volc": 82, "combined": 80, "moraine": 95,
    "water pocket": 72, "bedrock": 30, "landslide": 62, "glacier": 78,
    "ice": 85, "rock": 35, "unknown": 60,
}

# Component weights for the headline index (includes runout mobility, which
# needs the user's ground elevation) ...
INDEX_WEIGHTS = {
    "Proximity": 0.26, "Site density": 0.17, "Runout mobility": 0.20,
    "Flood magnitude": 0.19, "Dam vulnerability": 0.10, "Recency": 0.08,
}
# ... and for the location-only exposure sub-score that is ranked against every
# documented site (no mobility term - there is no "user" at a catalogue point).
EXPOSURE_WEIGHTS = {
    "Proximity": 0.34, "Site density": 0.22, "Flood magnitude": 0.22,
    "Dam vulnerability": 0.12, "Recency": 0.10,
}
RANKING_RADIUS_KM = 50.0


def _score_proximity(dist_km):
    return 100.0 * np.exp(-np.asarray(dist_km, dtype="float64") / 25.0)


def _score_density(count):
    return np.clip(np.asarray(count, dtype="float64") * 4.0, 0.0, 100.0)


def _score_magnitude(q_m3s):
    q = np.clip(np.asarray(q_m3s, dtype="float64"), 1.0, None)
    return np.clip(100.0 * np.log10(q) / 4.0, 0.0, 100.0)  # 1 -> 0, 1e4 -> 100


def _score_mobility(h_l):
    return np.clip(np.asarray(h_l, dtype="float64") * 300.0, 0.0, 100.0)  # 0.33 -> 100


def _score_recency(years_since):
    ys = np.asarray(years_since, dtype="float64")
    return np.clip(100.0 - (ys - 10.0) * (90.0 / 190.0), 10.0, 100.0)


def _score_dam(dam_type) -> float:
    if dam_type is None or (isinstance(dam_type, float) and np.isnan(dam_type)):
        return 60.0
    key = str(dam_type).lower()
    for token, val in DAM_VULNERABILITY.items():
        if token in key:
            return float(val)
    return 60.0


def hazard_tier(index_value: float) -> str:
    for lo, name in ((80, "Severe"), (60, "High"), (40, "Moderate"), (20, "Low"), (0, "Minimal")):
        if index_value >= lo:
            return name
    return "Minimal"


@lru_cache(maxsize=1)
def _site_exposure_distribution() -> np.ndarray:
    """Location-exposure sub-score for every documented GLOF site (for ranking)."""
    gdf = load_global_glacier_data()
    lat = gdf["latitude"].to_numpy("float64")
    lon = gdf["longitude"].to_numpy("float64")
    q = pd.to_numeric(gdf["peak_discharge_m3s"], errors="coerce").to_numpy("float64")
    yr = pd.to_numeric(gdf["outburst_year"], errors="coerce").to_numpy("float64")
    dam_scores = np.array([_score_dam(x) for x in gdf["dam_type"].tolist()])
    n = len(lat)

    nearest_km = np.full(n, np.nan)
    density = np.zeros(n)
    for s in range(0, n, 256):
        e = min(s + 256, n)
        d = haversine_m(lat[s:e, None], lon[s:e, None], lat[None, :], lon[None, :]) / 1000.0
        for i in range(e - s):
            d[i, s + i] = np.inf
        nearest_km[s:e] = d.min(axis=1)
        density[s:e] = (d <= RANKING_RADIUS_KM).sum(axis=1)

    this_year = datetime.now().year
    years_since = np.where(np.isfinite(yr), this_year - yr, 300.0)
    w = EXPOSURE_WEIGHTS
    return (
        w["Proximity"] * _score_proximity(nearest_km)
        + w["Site density"] * _score_density(density)
        + w["Flood magnitude"] * _score_magnitude(np.where(np.isfinite(q), q, 0.0))
        + w["Dam vulnerability"] * dam_scores
        + w["Recency"] * _score_recency(years_since)
    )


def location_hazard_index(
    user_lat: float,
    user_lon: float,
    user_elevation: float,
    search_radius_km: float = 50.0,
    gdf: Optional[gpd.GeoDataFrame] = None,
) -> dict:
    """
    Quantitative GLOF Hazard Index (0-100) for a location, its severity tier,
    a component breakdown, and how it ranks against every documented GLOF site.

    Returns the full :func:`assess_global_location_risk` payload plus::

        hazard_index, hazard_tier, index_components,
        exposure_percentile, exposure_rank, inventory_size,
        nearest_site_km, sites_within_50km
    """
    if gdf is None:
        gdf = load_global_glacier_data()

    a = assess_global_location_risk(user_lat, user_lon, user_elevation, search_radius_km, gdf)

    d_km = haversine_m(
        user_lat, user_lon,
        gdf["latitude"].to_numpy("float64"), gdf["longitude"].to_numpy("float64"),
    ) / 1000.0
    nearest_km = float(np.min(d_km))
    density = int((d_km <= RANKING_RADIUS_KM).sum())

    q = a["lake_peak_discharge_m3s"] or 1.0
    last = a["nearest_lake_last_outburst"]
    try:
        years_since = max(datetime.now().year - int(str(last)[:4]), 0) if last else 300.0
    except (ValueError, TypeError):
        years_since = 300.0

    comp = {
        "Proximity": float(_score_proximity(nearest_km)),
        "Site density": float(_score_density(density)),
        "Runout mobility": float(_score_mobility(a["runout_ratio_HL"])),
        "Flood magnitude": float(_score_magnitude(q)),
        "Dam vulnerability": _score_dam(a["nearest_lake_dam_type"]),
        "Recency": float(_score_recency(years_since)),
    }

    index = sum(INDEX_WEIGHTS[k] * comp[k] for k in INDEX_WEIGHTS)
    exposure = sum(EXPOSURE_WEIGHTS[k] * comp[k] for k in EXPOSURE_WEIGHTS)

    dist = _site_exposure_distribution()
    percentile = float((dist < exposure).mean() * 100.0)
    rank = int((dist >= exposure).sum()) + 1

    return {
        **a,
        "hazard_index": round(index, 1),
        "hazard_tier": hazard_tier(index),
        "index_components": {k: round(v, 1) for k, v in comp.items()},
        "exposure_score": round(exposure, 1),
        "exposure_percentile": round(percentile, 1),
        "exposure_rank": rank,
        "inventory_size": int(len(dist)),
        "nearest_site_km": round(nearest_km, 1),
        "sites_within_50km": density,
    }


if __name__ == "__main__":  # build + smoke test:  python risk_engine.py [--build]
    import sys

    if "--build" in sys.argv:
        built = build_inventory(force="--force" in sys.argv)
        print(f"Built inventory: {len(built):,} rows -> {INVENTORY_CSV}")

    g = load_global_glacier_data()
    print(f"Loaded {len(g):,} documented GLOF sites across {g['basin'].nunique()} basins")
    print(f"  with reported peak discharge: {int(g['discharge_is_reported'].sum()):,}")
    print(f"  with DEM elevation:           {int(g['elevation_m'].notna().sum()):,}")
    demo = location_hazard_index(27.88, 86.87, 4200.0)  # Khumbu, Nepal
    for k in ("hazard_index", "hazard_tier", "exposure_percentile", "exposure_rank",
              "inventory_size", "nearest_site_km", "sites_within_50km"):
        print(f"  {k:22s} {demo[k]}")
    print("  components:", demo["index_components"])
