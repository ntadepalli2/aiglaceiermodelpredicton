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

Loads **~3,060 documented historical GLOF sites** from the real
[Zenodo *Glacier Lake Outburst Flood Database V3.0*](https://doi.org/10.5281/zenodo.7330345)
(Veh et al. 2023), covering the Andes, European Alps, NW North America, High
Mountain Asia, Scandinavia, Iceland and Greenland. Each site carries:
`country, region, lake_name, dam_type, mechanism, latitude, longitude,
outburst_date, reported_lake_volume_m3, reported_peak_discharge_m3s`.

`build_inventory()` does the one-time heavy lifting:

1. downloads the `.ods` database from Zenodo,
2. concatenates + cleans the regional sheets,
3. attaches a **real ground elevation** to every site from the **Copernicus
   GLO-90 DEM** via the [Open-Meteo elevation API](https://open-meteo.com/en/docs/elevation-api),
4. back-calculates `surface_area_m2` from the reported pre-outburst volume where
   one exists (inverse of the V–A scaling law).

The result is cached to `data/global_glof_inventory.csv` (committed to the repo,
so normal runs are instant and offline-safe). Resolution order: cached CSV →
fresh build from Zenodo → deterministic synthetic fallback.

Regenerate the cache with:

```bash
python risk_engine.py --build --force
```

### 2. Hydrodynamic scaling

Reported measurements are used where available (755 sites have a measured peak
discharge, 174 a measured volume). Elsewhere the engine falls back to published
empirical relations:

| Quantity | Relation |
|---|---|
| Water volume | `V = 0.035 · (surface_area_m2 ^ 1.29)`  [m³] |
| Peak discharge | `Q = 0.00013 · (V ^ 0.60)`  [m³/s] |

### 3. Spatial risk — `assess_global_location_risk(user_lat, user_lon, user_elevation)`

* Builds a **`shapely.strtree.STRtree`** R-tree over all site points for fast
  bounding-box candidate selection.
* Computes great-circle distance to each candidate, identifies the nearest
  documented site **above** the location, and the nearest of the world's
  principal glaciated basins.
* Runout mobility `H/L` = (DEM elevation drop) ÷ (horizontal distance).
* Returns a `Low` / `Medium` / `High` **Global Risk Score** combining `H/L`,
  proximity and peak breach discharge.

## Global open data sources

* **[Zenodo GLOF Database V3.0](https://doi.org/10.5281/zenodo.7330345)** — the
  live inventory behind this app: ~3,060 documented outburst events with
  coordinates, dam material, dates and reported magnitudes.
  *Veh, G. et al. (2023), J. Geophys. Res. Earth Surface.*
* **[Copernicus GLO-90 DEM](https://open-meteo.com/en/docs/elevation-api)** —
  free ~90 m global elevation, queried per site for the terrain drop (`H/L`).
* **GLIMS Glacier Database** — NASA/NSIDC global satellite glacier inventory
  (200,000+ glaciers); a candidate future layer for lake-growth monitoring.

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
