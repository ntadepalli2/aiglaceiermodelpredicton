# 🏔️ Global Glacial Lake Outburst Flood (GLOF) Risk Assessment

A Streamlit web app for **worldwide** GLOF hazard screening. Enter any latitude /
longitude (or search a town / address), pick a glaciated region, and the app
scans a global glacial-lake inventory with an R-tree spatial index to estimate
breach discharge, runout mobility (`H/L`) and a categorical risk score — then
drafts an AI emergency advisory tailored to that mountain range.

## Project structure

```
glof-risk-app/
├── app.py                     # Streamlit UI: sidebar inputs + 3 dashboard tabs
├── risk_engine.py             # Data loading, hydrodynamics, spatial risk query
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example   # copy to secrets.toml for live AI advisories
└── data/                      # local cache of the downloaded lake inventory
```

## Quick start

```bash
cd glof-risk-app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, `geopandas` / `shapely` install most easily via `conda`
(`conda install -c conda-forge geopandas`) or by using pip wheels on Python
3.11+.

### Enabling live AI advisories

The **AI Safety Advisory** tab works offline with a built-in template. For
live, range-specific advisories set an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # or add it to .streamlit/secrets.toml
pip install anthropic
```

## How it works

### 1. Global inventory — `load_global_glacier_data()`

Fetches and caches a worldwide glacial-lake inventory with the schema
`lake_id, country, latitude, longitude, elevation_m, surface_area_m2`.
Resolution order: local CSV cache → open remote inventories (Zenodo Global GLOF
Database / GLIMS) → a deterministic bundled synthetic inventory so the app always
runs.

### 2. Hydrodynamic scaling

| Quantity | Relation |
|---|---|
| Water volume | `V = 0.035 · (surface_area_m2 ^ 1.29)`  [m³] |
| Peak discharge | `Q = 0.00013 · (V ^ 0.60)`  [m³/s] |

### 3. Spatial risk — `assess_global_location_risk(user_lat, user_lon, user_elevation)`

* Builds a **`shapely.strtree.STRtree`** R-tree over all lake points for fast
  bounding-box candidate selection.
* Computes great-circle distance to each candidate, identifies the nearest lake
  **above** the site, and the nearest of the world's principal glaciated basins
  (Andes, Alps, Himalayas, Karakoram, Central Asia, Alaska/Pacific NW, Patagonia,
  Iceland, Scandinavia).
* Runout mobility `H/L` = elevation drop ÷ horizontal distance.
* Returns a `Low` / `Medium` / `High` **Global Risk Score** combining `H/L`,
  proximity and peak breach discharge.

## Global open data sources

* **Zenodo Global GLOF Inventory** — open global database of 3,000+ historic
  outburst events across 27 countries.
* **GLIMS Glacier Database** — NASA/NSIDC-backed global satellite inventory
  mapping 200,000+ glaciers worldwide.
* **Copernicus Global DEM (GLO-30, 30 m)** — free worldwide elevation dataset for
  terrain-drop (`H/L`) calculation anywhere on Earth.

## AI technologies in glacier disaster prediction

* **Satellite computer vision & SAR models** — CNNs process optical and Synthetic
  Aperture Radar imagery to monitor high-altitude lake growth, snout retreat and
  moraine-dam deformation before a breach.
* **Supervised ML for GLOF risk** — Random Forests / SVMs trained on historical
  disaster databases score outburst probability from lake volume, slope, dam
  material and glacier distance.
* **Physics-Informed Neural Networks (PINNs)** — hybrid models coupling fluid
  dynamics with ML to simulate avalanche-driven displacement waves, dam erosion
  and downstream flood routing in real time.
* **Multimodal AI & Early Warning Systems** — aggregate seismic sensors,
  automated weather stations and satellite radar to synthesize threat levels and
  auto-notify disaster-management teams.

### Current challenges

* **Cascading triggers** — permafrost degradation, temperature spikes and deep
  ice fractures chain together in ways remote sensors miss until collapse.
* **Cloud cover & coverage gaps** — optical satellites lose the glacier during
  monsoon cloud and storms, exactly when flood risk peaks.

## Disclaimer

Research / decision-support tool using empirical scaling laws and a simplified
mobility model. Not a substitute for site-specific hydrological modelling or
official national disaster-management guidance.
