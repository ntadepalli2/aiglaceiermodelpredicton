"""
app.py
======

Global Glacial Lake Outburst Flood (GLOF) Risk Assessment - Streamlit dashboard.

Layout
------
Sidebar : global coordinate inputs (lat / lon / search radius) OR city search,
          plus a quick-select dropdown for key glaciated regions.
Main    : three tabs -
            1. Global Hazard Map     (Folium / Leaflet + clustered lake markers)
            2. Glacial Analytics     (tables + charts derived from risk_engine)
            3. AI Safety Advisory    (LLM-generated 3-bullet disaster advisory)

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from geopy.geocoders import Nominatim
from geopy.distance import geodesic

import risk_engine as re

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Global GLOF Risk Assessment",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

REGION_PRESETS = {
    "— none —":            None,
    "Himalayas":           (28.00, 86.90, 6),
    "Andes":               (-13.50, -72.00, 5),
    "European Alps":       (46.00, 8.00, 6),
    "Alaska / Pacific NW": (61.00, -145.00, 4),
    "Central Asia":        (42.20, 78.50, 5),
}

# Reported peak-discharge bands (m³/s) for marker colouring.
Q_BINS = [0, 100, 500, 2000, np.inf]
Q_LABELS = ["<100 m³/s", "100–500 m³/s", "500–2000 m³/s", ">2000 m³/s"]
Q_COLORS = {
    "<100 m³/s": "#2c7fb8",
    "100–500 m³/s": "#41b6c4",
    "500–2000 m³/s": "#fe9929",
    ">2000 m³/s": "#d7301f",
    "not reported": "#9e9e9e",
}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading global GLOF inventory (Zenodo GLOF DB V3.0)…")
def get_inventory() -> pd.DataFrame:
    gdf = re.load_global_glacier_data()
    df = pd.DataFrame(gdf).copy()
    q_class = pd.cut(df["reported_peak_discharge_m3s"], bins=Q_BINS, labels=Q_LABELS)
    df["q_class"] = q_class.cat.add_categories("not reported").fillna("not reported")
    return df


@st.cache_data(show_spinner=False)
def geocode(query: str):
    """Return (lat, lon, address) for a free-text place, or None."""
    try:
        geo = Nominatim(user_agent="global_glof_risk_app")
        loc = geo.geocode(query, timeout=10)
        if loc:
            return loc.latitude, loc.longitude, loc.address
    except Exception:
        pass
    return None


@st.cache_data(show_spinner="Scoring location against the global GLOF record…")
def hazard_index(la: float, lo: float, el: float, rad: float) -> dict:
    return re.location_hazard_index(la, lo, el, rad, gdf=re.load_global_glacier_data())


# --------------------------------------------------------------------------- #
# AI emergency-response module  (AI Safety Advisory tab)
# --------------------------------------------------------------------------- #

def build_advisory_prompt(context: dict) -> str:
    """
    Build the LLM prompt from the computed GLOF Hazard Index and its components.
    """
    comps = context.get("components", {})
    comp_lines = "\n".join(f"    - {k}: {v}/100" for k, v in comps.items())
    return (
        "You are an emergency-management advisor specialising in glacial lake "
        "outburst floods (GLOFs). A quantitative GLOF Hazard Index (0-100) has "
        "already been computed for a specific location. Interpret it for the "
        "reader - do not restate the numbers mechanically.\n\n"
        f"Location / region: {context.get('country_region', 'Unknown')}\n"
        f"Nearest glaciated basin: {context.get('nearest_glacier', 'Unknown')}\n"
        f"GLOF Hazard Index: {context.get('hazard_index', 'n/a')}/100. "
        f"This is the PRIMARY signal - tier: {context.get('hazard_tier', 'n/a')} "
        f"(bands: 0-20 Minimal, 20-40 Low, 40-60 Moderate, 60-80 High, 80-100 Severe).\n"
        f"For context only: the location's exposure exceeds {context.get('exposure_percentile', 'n/a')}% "
        f"of the {context.get('inventory_size', 'n/a')} catalogued outburst sites "
        f"(most sites sit in known hazard zones, so a normal town scores near 0% here - "
        f"do NOT treat a low percentage as alarming, and do NOT invert it).\n"
        f"Nearest documented outburst site: {context.get('runout_distance_km', 'n/a')} km away, "
        f"dam material {context.get('dam_type', 'unknown')}, "
        f"last recorded outburst {context.get('last_outburst', 'none on record')}\n"
        f"Peak breach discharge (reported or scaled): "
        f"{context.get('peak_discharge_m3s', 'n/a')} m3/s\n"
        f"Index component scores:\n{comp_lines}\n\n"
        "Write a location-specific advisory as EXACTLY three short paragraphs "
        "(2-3 sentences each, no headings, no bullet points):\n"
        "1. What this index and global ranking mean in plain language for someone "
        "living in or travelling to this location.\n"
        "2. The local warning signs that matter most here and the single most "
        "important evacuation action for this terrain.\n"
        "3. One preparedness priority for local authorities, tied to the "
        "highest-scoring (worst) index component.\n"
    )


def _secret(name: str, default: str = "") -> str:
    """Read a config value from the environment first, then ``st.secrets``."""
    val = os.environ.get(name, "")
    if not val:
        try:
            val = st.secrets.get(name, default)
        except Exception:
            val = default
    return val or default


SYSTEM_PROMPT = "You are a careful glacial-hazard emergency-management advisor."


def _chat_completion(base_url: str, api_key: str, model: str, prompt: str,
                     extra_headers: dict | None = None) -> str:
    """POST to any OpenAI-compatible /chat/completions endpoint and return the text."""
    payload = {
        "model": model,
        "max_tokens": _int_secret("LLM_MAX_TOKENS", 600),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    effort = _secret("LLM_REASONING_EFFORT")  # e.g. "low" for gpt-oss / o-series
    if effort:
        payload["reasoning_effort"] = effort

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    headers.update(extra_headers or {})

    resp = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                         headers=headers, json=payload, timeout=45)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


class LLMUnavailable(Exception):
    """No key configured, budget exhausted, or the provider call failed."""


def generate_ai_advisory(prompt: str, context: dict | None = None,
                         strict: bool = False) -> str:
    """
    Connect to an LLM API client and return the location advisory.

    Provider is auto-detected from whichever key is configured (env var or
    ``st.secrets``):

    * ``OPENROUTER_API_KEY``  -> OpenRouter (OpenAI-compatible chat completions)
    * ``ANTHROPIC_API_KEY``   -> Anthropic Messages API
    * ``OPENAI_API_KEY`` (+ optional ``OPENAI_BASE_URL``) -> any OpenAI-compatible
      endpoint (OpenAI, Groq, Google Gemini, Cerebras, Mistral, ...)

    Optional tuning secrets: ``LLM_MODEL``, ``LLM_MAX_TOKENS``,
    ``LLM_REASONING_EFFORT``.

    With no key configured a deterministic offline template is returned so the
    app stays usable without network / credentials.
    """
    try:
        or_key = _secret("OPENROUTER_API_KEY")
        if or_key:
            return _chat_completion(
                "https://openrouter.ai/api/v1", or_key,
                _secret("LLM_MODEL", "anthropic/claude-sonnet-5"), prompt,
                extra_headers={
                    "HTTP-Referer": "https://github.com/ntadepalli2/aiglaceiermodelpredicton",
                    "X-Title": "Global GLOF Risk Assessment",
                },
            )

        ant_key = _secret("ANTHROPIC_API_KEY")
        if ant_key:
            import anthropic

            client = anthropic.Anthropic(api_key=ant_key)
            msg = client.messages.create(
                model=_secret("LLM_MODEL", "claude-sonnet-5"),
                max_tokens=_int_secret("LLM_MAX_TOKENS", 600),
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in msg.content if b.type == "text").strip()

        oai_key = _secret("OPENAI_API_KEY")
        if oai_key:
            return _chat_completion(
                _secret("OPENAI_BASE_URL", "https://api.openai.com/v1"), oai_key,
                _secret("LLM_MODEL", "gpt-4o-mini"), prompt,
            )
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise LLMUnavailable(str(exc)) from exc
        return (f"_LLM request failed ({exc}). Showing offline guidance._\n\n"
                + _offline_advisory(context))

    if strict:
        raise LLMUnavailable("no LLM API key configured")
    return _offline_advisory(context)


# --------------------------------------------------------------------------- #
# LLM cost control
#
# The advisory is generated automatically, so on a public deployment every
# visitor would otherwise spend tokens. Three layers keep that bounded:
#   1. a server-wide cache, so one advisory per ~11 km cell is shared by ALL
#      visitors rather than regenerated per session;
#   2. a per-session cap, so one visitor clicking around the map cannot drain
#      the quota;
#   3. a server-wide daily cap, after which everyone gets the offline template.
# Failures and exhaustion raise, so they are never cached in place of a real
# advisory.
# --------------------------------------------------------------------------- #

def _int_secret(name: str, default: int) -> int:
    try:
        return int(_secret(name, str(default)) or default)
    except ValueError:
        return default


@st.cache_resource
def _llm_budget() -> dict:
    """Server-wide daily call counter, shared across every user session."""
    return {"day": None, "calls": 0}


def _consume_llm_budget() -> bool:
    cap = _int_secret("LLM_DAILY_CALL_CAP", 250)
    budget = _llm_budget()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if budget["day"] != today:
        budget["day"], budget["calls"] = today, 0
    if budget["calls"] >= cap:
        return False
    budget["calls"] += 1
    return True


@st.cache_data(show_spinner=False, max_entries=500, ttl=7 * 24 * 3600)
def cached_advisory(cache_key: tuple, prompt: str, context: dict) -> str:
    """One LLM call per distinct location cell, shared across all sessions."""
    if not _consume_llm_budget():
        raise LLMUnavailable("daily advisory budget reached")
    return generate_ai_advisory(prompt, context, strict=True)


def advisory_cache_key(lat: float, lon: float, elevation: float, tier: str) -> tuple:
    """Coarse key: ~11 km cell and 100 m elevation band, so nearby clicks reuse."""
    return (round(lat, 1), round(lon, 1), round(float(elevation) / 100.0) * 100, tier)


def _offline_advisory(context: dict | None = None) -> str:
    ctx = context or {}
    tier = ctx.get("hazard_tier", "Moderate")
    idx = ctx.get("hazard_index", "—")
    pct = ctx.get("exposure_percentile", "—")
    worst = max(ctx.get("components", {"—": 0}).items(), key=lambda kv: kv[1])[0]
    return (
        f"**GLOF Hazard Index {idx}/100 — {tier}.** This location ranks above "
        f"{pct}% of documented GLOF sites worldwide. A {tier.lower()} rating means "
        f"outburst-flood exposure here is "
        + ("negligible; standard mountain-river caution is enough."
           if tier in ("Minimal", "Low")
           else "material: treat glacier-fed channels as active hazard corridors.")
        + "\n\n"
        "Watch for a sudden drop or surge in river level, unusually turbid or "
        "debris-laden water, and rumbling from upstream. If any appear, move "
        "immediately away from the channel and climb 30–50 m above the valley "
        "floor — never cross bridges during a surge.\n\n"
        f"Preparedness priority for authorities: the weakest factor here is "
        f"**{worst}** — pair upstream water-level/seismic sensors with SMS-and-siren "
        f"alerting, and keep signposted evacuation routes to high ground clear."
    )


# --------------------------------------------------------------------------- #
# Sidebar - inputs
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def dem_elevation(la: float, lo: float) -> float:
    try:
        return float(re._fetch_elevations([la], [lo])[0])
    except Exception:
        return float("nan")


def _set_location(la: float, lo: float, *, pull_elev: bool = True) -> None:
    """Point the app at a new location (used by presets, search and map clicks)."""
    st.session_state["sel_lat"] = float(round(la, 4))
    st.session_state["sel_lon"] = float(round(lo, 4))
    st.session_state["_recenter"] = True
    if pull_elev:
        z = dem_elevation(round(la, 4), round(lo, 4))
        if np.isfinite(z):
            st.session_state["sel_elev"] = float(round(z, 0))


st.sidebar.title("🏔️ GLOF Risk Inputs")

# --- one-time defaults -----------------------------------------------------
for k, v in {"sel_lat": 20.0, "sel_lon": 0.0, "sel_elev": 3000.0}.items():
    st.session_state.setdefault(k, v)

# --- apply a pending location BEFORE any location widget is instantiated ---
_pending = st.session_state.pop("_pending_loc", None)
if _pending:
    _set_location(*_pending)

region = st.sidebar.selectbox("Quick-select glaciated region", list(REGION_PRESETS))
if region != st.session_state.get("_last_region"):
    st.session_state["_last_region"] = region
    if REGION_PRESETS[region]:
        pa, po, _pz = REGION_PRESETS[region]
        _set_location(pa, po)

st.sidebar.markdown("### Location")
mode = st.sidebar.radio("Input mode", ["Coordinates", "City / address search"], horizontal=True)

geo_result = None
if mode == "City / address search":
    query = st.sidebar.text_input("Town, city or address", placeholder="e.g. Pokhara, Nepal")
    if query:
        geo_result = geocode(query)
        if geo_result:
            st.sidebar.success(f"📍 {geo_result[2][:60]}")
            if query != st.session_state.get("_last_geo"):
                st.session_state["_last_geo"] = query
                _set_location(geo_result[0], geo_result[1])
        else:
            st.sidebar.error("Location not found.")
else:
    st.sidebar.caption("💡 Or click any point on the Global Hazard Map.")

lat = st.sidebar.number_input(
    "Latitude", -90.0, 90.0, step=0.001, format="%.4f", key="sel_lat")
lon = st.sidebar.number_input(
    "Longitude", -180.0, 180.0, step=0.001, format="%.4f", key="sel_lon")

if st.sidebar.button("📡 Fetch ground elevation from DEM"):
    z = dem_elevation(lat, lon)
    if np.isfinite(z):
        st.session_state["sel_elev"] = float(round(z, 0))

elevation = st.sidebar.number_input(
    "Ground elevation (m)", min_value=-400.0, max_value=8000.0, step=10.0, key="sel_elev",
    help="Ground elevation at your location. A map click pulls this from the DEM automatically.",
)
radius_km = st.sidebar.slider("Search radius (km)", 5, 300, 50, 5)

st.sidebar.caption(
    "Data: [Zenodo GLOF Database V3.0](https://doi.org/10.5281/zenodo.7330345) "
    "(Veh et al. 2023) · ground elevations from the Copernicus GLO-90 DEM via "
    "[Open-Meteo](https://open-meteo.com/en/docs/elevation-api). "
    "Set `OPENROUTER_API_KEY` (or `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) for live AI advisories."
)

# --------------------------------------------------------------------------- #
# Compute risk
# --------------------------------------------------------------------------- #

inventory = get_inventory()
assessment = re.assess_global_location_risk(lat, lon, elevation, radius_km)

RISK_COLOR = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("Global Glacial Lake Outburst Flood (GLOF) Risk Assessment")
st.markdown(
    f"**Inventory:** {len(inventory):,} documented GLOF sites "
    f"([Zenodo GLOF Database V3.0](https://doi.org/10.5281/zenodo.7330345)) · "
    f"**Location:** {lat:.3f}, {lon:.3f} · "
    f"**Global Risk Level:** {RISK_COLOR.get(assessment['risk_score'], '')} "
    f"**{assessment['risk_score']}**"
)

tab_map, tab_analytics, tab_ai = st.tabs(
    ["🗺️ Global Hazard Map", "📊 Glacial Analytics", "🤖 AI Safety Advisory"]
)

# --------------------------------------------------------------------------- #
# Tab 1 - Global Hazard Map
# --------------------------------------------------------------------------- #

with tab_map:
    c1, c2 = st.columns([3, 1])

    with c2:
        st.subheader("Location metrics")
        st.metric("Nearest documented GLOF",
                  f'{assessment["nearest_lake_distance_km"]:,g} km',
                  assessment["nearest_basin"], delta_color="off")
        h_help = ("from nearest site above the location"
                  if assessment.get("runout_from_site_above")
                  else "no documented site above this location nearby")
        st.metric("Elevation Drop (H)", f'{assessment["elevation_drop_H_m"]} m',
                  h_help, delta_color="off")
        st.metric("Global Risk Level",
                  f'{RISK_COLOR.get(assessment["risk_score"], "")} {assessment["risk_score"]}',
                  f'H/L {assessment["runout_ratio_HL"]}', delta_color="off")
        st.metric("Documented sites within radius", assessment["lakes_within_radius"])

        q_filter = st.multiselect(
            "Show peak-discharge class", list(Q_COLORS), default=list(Q_COLORS)
        )
        max_markers = st.slider("Max markers", 200, 5000, 2000, 100)

    with c1:
        st.caption("🖱️ **Click anywhere on the map** to assess that point "
                   "(elevation is pulled from the DEM automatically).")
        at_default = (round(lat, 3), round(lon, 3)) == (20.0, 0.0)
        map_zoom = 2 if at_default else 8

        # Keyless light basemap. CARTO "positron" and Stamen now require an API
        # key ("API KEY REQUIRED" is watermarked onto their tiles), so use
        # Esri's free World Light Gray Base, with OpenStreetMap as a fallback.
        fmap = folium.Map(
            location=[lat, lon],
            zoom_start=map_zoom,
            tiles=None,
            world_copy_jump=True,
        )
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                  "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles © Esri — Esri, DeLorme, NAVTEQ",
            name="Light gray",
            max_zoom=16,
            control=True,
        ).add_to(fmap)
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(fmap)

        view = inventory[inventory["q_class"].astype(str).isin(q_filter)]
        # Prioritise sites near the current location, then cap for performance.
        view = view.assign(
            _d=re.haversine_m(lat, lon, view["latitude"].values, view["longitude"].values)
        ).sort_values("_d").head(max_markers)

        def _fmt(v, unit, dp=0):
            return f"{v:,.{dp}f} {unit}" if pd.notna(v) else "not reported"

        cluster = MarkerCluster(name="GLOF sites").add_to(fmap)
        for _, r in view.iterrows():
            color = Q_COLORS.get(str(r["q_class"]), "#9e9e9e")
            name = r["lake_name"] if pd.notna(r["lake_name"]) else "(unnamed lake)"
            popup = folium.Popup(
                html=(
                    f"<b>{name}</b><br>"
                    f"🏳️ {r['country']} &nbsp;|&nbsp; {r['region']}<br>"
                    f"Dam type: {r['dam_type'] if pd.notna(r['dam_type']) else '—'}<br>"
                    f"Last outburst: {r['outburst_date'] if pd.notna(r['outburst_date']) else '—'}"
                    + (f" ({r['mechanism']})" if pd.notna(r['mechanism']) else "") + "<br>"
                    f"Reported peak discharge: {_fmt(r['reported_peak_discharge_m3s'], 'm³/s', 1)}<br>"
                    f"Reported lake volume: {_fmt(r['reported_lake_volume_m3'], 'm³')}<br>"
                    f"Ground elevation (DEM): {_fmt(r['elevation_m'], 'm')}<br>"
                    f"<span style='color:#888'>{r['lake_id']}</span>"
                ),
                max_width=280,
            )
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=popup,
            ).add_to(cluster)

        # User location + geodesic line to nearest documented site above it
        folium.Marker(
            [lat, lon],
            tooltip="Selected location",
            icon=folium.Icon(color="black", icon="user", prefix="fa"),
        ).add_to(fmap)

        nlat, nlon = assessment["nearest_lake_lat"], assessment["nearest_lake_lon"]
        near_label = assessment["nearest_lake_name"] or assessment["nearest_lake_id"]
        folium.PolyLine(
            [[lat, lon], [nlat, nlon]],
            color="#d7301f", weight=3, dash_array="8",
            tooltip=f'{geodesic((lat, lon), (nlat, nlon)).km:.1f} km to {near_label}',
        ).add_to(fmap)
        folium.CircleMarker(
            [nlat, nlon], radius=8, color="#d7301f", fill=True, fill_opacity=1.0,
            tooltip=f'Nearest documented GLOF site: {near_label}',
        ).add_to(fmap)

        folium.LayerControl().add_to(fmap)
        fmap.add_child(folium.LatLngPopup())

        # Follow the selection only right after it changes programmatically,
        # so manual panning between clicks is preserved.
        recenter = st.session_state.pop("_recenter", False)
        st_kwargs = {"center": [lat, lon], "zoom": map_zoom} if recenter else {}
        map_state = st_folium(
            fmap, height=620, use_container_width=True,
            key="hazard_map", returned_objects=["last_clicked"], **st_kwargs,
        )

    clicked = (map_state or {}).get("last_clicked")
    if clicked:
        cll = (round(clicked["lat"], 4), round(((clicked["lng"] + 180) % 360) - 180, 4))
        if cll != st.session_state.get("_last_click"):
            st.session_state["_last_click"] = cll
            st.session_state["_pending_loc"] = cll
            st.rerun()

    st.caption(
        "Each marker is a **documented** historical outburst site (Zenodo GLOF "
        "Database V3.0). Colour = reported peak discharge; grey = magnitude not "
        "reported. Red dashed line = geodesic to the nearest documented outburst "
        "site."
    )

# --------------------------------------------------------------------------- #
# Tab 2 - Glacial Analytics
# --------------------------------------------------------------------------- #

with tab_analytics:
    def _m(v, unit, dp=0):
        return f"{v:,.{dp}f} {unit}" if v is not None and pd.notna(v) else "not reported"

    st.subheader("Nearest documented GLOF site")
    a, b, c, d = st.columns(4)
    a.metric("Lake", assessment["nearest_lake_name"] or "(unnamed)")
    b.metric("Dam type", assessment["nearest_lake_dam_type"] or "—")
    c.metric("Last recorded outburst", assessment["nearest_lake_last_outburst"] or "—")
    d.metric("Runout ratio H/L", assessment["runout_ratio_HL"])

    e, f, g, h = st.columns(4)
    e.metric("Surface area (est.)", _m(assessment["lake_surface_area_m2"], "m²"))
    f.metric("Water volume", _m(assessment["lake_water_volume_m3"], "m³"))
    g.metric(
        "Peak discharge Q", _m(assessment["lake_peak_discharge_m3s"], "m³/s", 1),
        "reported" if assessment["discharge_is_reported"] else "estimated (V-scaling)",
    )
    h.metric("Elevation drop H", f'{assessment["elevation_drop_H_m"]} m')

    st.caption(
        f"{int(inventory['discharge_is_reported'].sum()):,} of {len(inventory):,} sites "
        f"have a reported peak discharge; {int(inventory['reported_lake_volume_m3'].notna().sum()):,} "
        f"have a reported pre-outburst volume. Areas/volumes are back-calculated from the "
        f"V = 0.035·A¹·²⁹ scaling law where not reported."
    )
    st.divider()

    left, right = st.columns(2)
    with left:
        st.markdown("**Documented outbursts per region**")
        st.bar_chart(inventory["region"].value_counts())
        st.markdown("**Impounding-dam material**")
        st.bar_chart(inventory["dam_type"].value_counts().head(8))

    with right:
        st.markdown("**Outbursts by decade**")
        dec = inventory["outburst_year"].dropna()
        dec = (dec // 10 * 10).astype(int)
        dec = dec[dec >= 1850]
        st.bar_chart(dec.value_counts().sort_index())

        st.markdown("**Reported peak discharge (log10 m³/s)**")
        q = inventory["reported_peak_discharge_m3s"].dropna()
        q = np.log10(q[q > 0])
        hist = np.histogram(q, bins=20)
        st.bar_chart(pd.Series(hist[0], index=np.round(hist[1][:-1], 2)))

    st.divider()
    st.markdown("**Documented outburst sites near the selected location**")
    near = inventory.assign(
        distance_km=re.haversine_m(lat, lon, inventory["latitude"].values,
                                   inventory["longitude"].values) / 1000.0
    )
    near = near[near["distance_km"] <= radius_km]
    show_cols = ["lake_name", "country", "region", "distance_km", "elevation_m",
                 "dam_type", "outburst_date", "mechanism", "reported_peak_discharge_m3s"]
    if near.empty:
        st.info("No documented outburst sites within the current search radius — widen it in the sidebar.")
    else:
        st.dataframe(
            near.sort_values("distance_km")[show_cols].head(25),
            use_container_width=True, hide_index=True,
            column_config={
                "distance_km": st.column_config.NumberColumn("dist (km)", format="%.1f"),
                "elevation_m": st.column_config.NumberColumn("elev (m)", format="%.0f"),
                "reported_peak_discharge_m3s": st.column_config.NumberColumn("Qp (m³/s)", format="%.0f"),
            },
        )

    st.download_button(
        "⬇️ Download full inventory (CSV)",
        inventory.to_csv(index=False).encode(),
        file_name="global_glof_inventory.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------- #
# Tab 3 - AI Safety Advisory
# --------------------------------------------------------------------------- #

with tab_ai:
    st.subheader("AI Safety Advisory")

    hz = hazard_index(round(lat, 3), round(lon, 3), float(elevation), float(radius_km))
    idx, tier = hz["hazard_index"], hz["hazard_tier"]
    TIER_ICON = {"Minimal": "🟢", "Low": "🟢", "Moderate": "🟠", "High": "🔴", "Severe": "🔴"}

    m1, m2, m3 = st.columns(3)
    m1.metric("GLOF Hazard Index", f"{idx} / 100", f'{TIER_ICON.get(tier, "")} {tier}',
              delta_color="off")
    m2.metric("Nearest documented GLOF", f'{hz["nearest_site_km"]:.1f} km',
              hz["nearest_lake_region"], delta_color="off")
    m3.metric("Sites within 50 km", f'{hz["sites_within_50km"]:,}',
              f'exceeds {hz["exposure_percentile"]:.0f}% of catalogued sites',
              delta_color="off")
    st.progress(min(idx / 100.0, 1.0), text=f"{tier} — index {idx}/100")

    worst = max(hz["index_components"].items(), key=lambda kv: kv[1])
    st.markdown("**How the index breaks down** (0–100 per factor)")
    st.bar_chart(pd.Series(hz["index_components"]).sort_values())

    st.caption(
        f"Nearest documented GLOF site: **{hz['nearest_site_km']} km** away "
        f"({hz['nearest_lake_name'] or 'unnamed lake'}, "
        f"{hz['nearest_lake_dam_type'] or 'dam type n/a'} dam, "
        f"last outburst {hz['nearest_lake_last_outburst'] or '—'}). "
        f"{hz['sites_within_50km']} documented sites within 50 km. "
        f"Dominant factor: **{worst[0]}** ({worst[1]}/100)."
    )
    st.divider()

    country_region = (geo_result[2] if geo_result else
                      f'{hz["nearest_lake_country"]} / {hz["nearest_lake_region"]}')
    q = hz["lake_peak_discharge_m3s"]
    context = {
        "country_region": country_region,
        "nearest_glacier": hz["nearest_basin"],
        "runout_distance_km": hz["nearest_lake_distance_km"],
        "hazard_index": idx,
        "hazard_tier": tier,
        "exposure_percentile": hz["exposure_percentile"],
        "inventory_size": hz["inventory_size"],
        "peak_discharge_m3s": (round(q, 1) if q is not None else "n/a"),
        "dam_type": hz["nearest_lake_dam_type"] or "unknown",
        "last_outburst": hz["nearest_lake_last_outburst"] or "none on record",
        "components": hz["index_components"],
    }

    adv_key = advisory_cache_key(lat, lon, elevation, tier)
    session_cap = _int_secret("LLM_SESSION_CALL_CAP", 10)

    if st.session_state.get("adv_key") != adv_key:
        if st.session_state.get("adv_calls", 0) >= session_cap:
            st.session_state["advisory"] = _offline_advisory(context)
            st.session_state["advisory_is_ai"] = False
        else:
            try:
                with st.spinner("Assessing location and drafting advisory…"):
                    st.session_state["advisory"] = cached_advisory(
                        adv_key, build_advisory_prompt(context), context
                    )
                st.session_state["advisory_is_ai"] = True
            except LLMUnavailable:
                st.session_state["advisory"] = _offline_advisory(context)
                st.session_state["advisory_is_ai"] = False
            st.session_state["adv_calls"] = st.session_state.get("adv_calls", 0) + 1
        st.session_state["adv_key"] = adv_key

    st.markdown("### Advisory for this location")
    st.markdown(st.session_state["advisory"])
    if not st.session_state.get("advisory_is_ai", False):
        st.info(
            "Showing the built-in assessment. Live AI advisories are unavailable "
            "right now (no API key configured, or the shared daily limit for this "
            "deployment has been reached). The hazard index above is unaffected."
        )
    # Regenerate deliberately bypasses the shared cache, so it still costs one
    # call and is charged against both budgets.
    if st.button("↻ Regenerate"):
        if st.session_state.get("adv_calls", 0) >= session_cap or not _consume_llm_budget():
            st.session_state["advisory"] = _offline_advisory(context)
            st.session_state["advisory_is_ai"] = False
        else:
            try:
                with st.spinner("Redrafting advisory…"):
                    st.session_state["advisory"] = generate_ai_advisory(
                        build_advisory_prompt(context), context, strict=True
                    )
                st.session_state["advisory_is_ai"] = True
            except LLMUnavailable:
                st.session_state["advisory"] = _offline_advisory(context)
                st.session_state["advisory_is_ai"] = False
            st.session_state["adv_calls"] = st.session_state.get("adv_calls", 0) + 1
        st.rerun()

    st.caption(
        "The GLOF Hazard Index (0–100) blends proximity to documented outburst "
        "sites, local site density, runout mobility (H/L), reported flood magnitude, "
        "dam-material vulnerability and outburst recency, benchmarked against every "
        "site in the Zenodo GLOF Database V3.0. Decision-support only — validate "
        "against official national disaster-management guidance. Without an LLM API "
        "key a built-in offline assessment is shown."
    )
