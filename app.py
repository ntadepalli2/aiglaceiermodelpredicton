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
    df = pd.DataFrame(gdf.drop(columns="geometry"))
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


# --------------------------------------------------------------------------- #
# AI emergency-response module  (AI Safety Advisory tab)
# --------------------------------------------------------------------------- #

def build_advisory_prompt(context: dict) -> str:
    """
    Construct an LLM prompt from the geographic risk context.

    Parameters
    ----------
    context : dict with keys
        country_region, nearest_glacier, runout_distance_km, risk_rating,
        elevation_drop_m, peak_discharge_m3s, dam_type, last_outburst
    """
    return (
        "You are an emergency-management advisor specialising in glacial lake "
        "outburst floods (GLOFs).\n\n"
        "Location context (nearest documented outburst site, Zenodo GLOF DB V3.0):\n"
        f"- Country / region: {context.get('country_region', 'Unknown')}\n"
        f"- Nearest glaciated basin: {context.get('nearest_glacier', 'Unknown')}\n"
        f"- Distance to nearest documented GLOF site: "
        f"{context.get('runout_distance_km', 'n/a')} km\n"
        f"- Impounding-dam material at that site: {context.get('dam_type', 'unknown')}\n"
        f"- Most recent recorded outburst there: {context.get('last_outburst', 'n/a')}\n"
        f"- Elevation drop (H) from lake to site: "
        f"{context.get('elevation_drop_m', 'n/a')} m\n"
        f"- Peak breach discharge (reported or V-scaled): "
        f"{context.get('peak_discharge_m3s', 'n/a')} m³/s\n"
        f"- Computed risk rating: {context.get('risk_rating', 'Unknown')}\n\n"
        "Produce EXACTLY three concise bullet points, tailored to this specific "
        "mountain range's terrain and infrastructure, covering:\n"
        "  1. Local warning signs residents and trekkers should watch for.\n"
        "  2. Evacuation tactics and safe-ground guidance for this valley type.\n"
        "  3. Infrastructure / preparedness actions for local authorities.\n"
        "Keep each bullet under 40 words. Do not add a preamble or conclusion."
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
        "max_tokens": int(_secret("LLM_MAX_TOKENS", "800") or 800),
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


def generate_ai_advisory(prompt: str) -> str:
    """
    Connect to an LLM API client and return a 3-bullet disaster advisory.

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
                max_tokens=int(_secret("LLM_MAX_TOKENS", "800") or 800),
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
        return (f"_LLM request failed ({exc}). Showing offline guidance._\n\n"
                + _offline_advisory())

    return _offline_advisory()


def _offline_advisory() -> str:
    return (
        "- **Warning signs:** sudden drop or surge in river level, unusually turbid or "
        "debris-laden water, rumbling from upstream, fresh cracks or seepage on the moraine dam.\n"
        "- **Evacuation:** move immediately perpendicular to the river channel and climb at "
        "least 30–50 m above the valley floor; never cross bridges during a surge; follow "
        "marked high-ground routes rather than the road along the river.\n"
        "- **Authorities:** install upstream water-level and seismic sensors with SMS/siren "
        "alerting, keep evacuation routes and assembly points signposted and clear, and "
        "pre-position rescue caches in downstream settlements."
    )


# --------------------------------------------------------------------------- #
# Sidebar - inputs
# --------------------------------------------------------------------------- #

st.sidebar.title("🏔️ GLOF Risk Inputs")

region = st.sidebar.selectbox("Quick-select glaciated region", list(REGION_PRESETS))
preset = REGION_PRESETS[region]

st.sidebar.markdown("### Location")
mode = st.sidebar.radio("Input mode", ["Coordinates", "City / address search"], horizontal=True)

geo_result = None
if mode == "City / address search":
    query = st.sidebar.text_input("Town, city or address", placeholder="e.g. Pokhara, Nepal")
    if query:
        geo_result = geocode(query)
        if geo_result:
            st.sidebar.success(f"📍 {geo_result[2][:60]}")
        else:
            st.sidebar.error("Location not found.")

default_lat, default_lon, default_zoom = (20.0, 0.0, 2)
if preset:
    default_lat, default_lon, default_zoom = preset
if geo_result:
    default_lat, default_lon, default_zoom = geo_result[0], geo_result[1], 9

lat = st.sidebar.number_input("Latitude", -90.0, 90.0, float(round(default_lat, 4)), 0.001, format="%.4f")
lon = st.sidebar.number_input("Longitude", -180.0, 180.0, float(round(default_lon, 4)), 0.001, format="%.4f")


@st.cache_data(show_spinner=False)
def dem_elevation(la: float, lo: float) -> float:
    try:
        return float(re._fetch_elevations([la], [lo])[0])
    except Exception:
        return float("nan")

if "elev" not in st.session_state:
    st.session_state["elev"] = 3000.0
if st.sidebar.button("📡 Fetch ground elevation from DEM"):
    z = dem_elevation(lat, lon)
    if np.isfinite(z):
        st.session_state["elev"] = float(round(z, 0))

elevation = st.sidebar.number_input(
    "Ground elevation (m)", min_value=-400.0, max_value=8000.0, step=50.0, key="elev",
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
        st.metric("Nearest Glacial Basin", assessment["nearest_basin"],
                  f'{assessment["basin_distance_km"]} km')
        st.metric("Elevation Drop (H)", f'{assessment["elevation_drop_H_m"]} m',
                  f'runout L {assessment["nearest_lake_distance_km"]} km')
        st.metric("Global Risk Level",
                  f'{RISK_COLOR.get(assessment["risk_score"], "")} {assessment["risk_score"]}',
                  f'H/L {assessment["runout_ratio_HL"]}')
        st.metric("Documented sites within radius", assessment["lakes_within_radius"])

        q_filter = st.multiselect(
            "Show peak-discharge class", list(Q_COLORS), default=list(Q_COLORS)
        )
        max_markers = st.slider("Max markers", 200, 5000, 2000, 100)

    with c1:
        # Keyless light basemap. CARTO "positron" and Stamen now require an API
        # key ("API KEY REQUIRED" is watermarked onto their tiles), so use
        # Esri's free World Light Gray Base, with OpenStreetMap as a fallback.
        fmap = folium.Map(
            location=[default_lat, default_lon],
            zoom_start=default_zoom,
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
        st_folium(fmap, height=620, use_container_width=True, returned_objects=[])

    st.caption(
        "Each marker is a **documented** historical outburst site (Zenodo GLOF "
        "Database V3.0). Colour = reported peak discharge; grey = magnitude not "
        "reported. Red dashed line = geodesic to the nearest documented site above "
        "the selected location."
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
    st.subheader("AI-generated disaster advisory")

    country_region = (geo_result[2] if geo_result else
                      f'{assessment["nearest_lake_country"]} / {assessment["nearest_lake_region"]}')

    q = assessment["lake_peak_discharge_m3s"]
    context = {
        "country_region": country_region,
        "nearest_glacier": assessment["nearest_basin"],
        "runout_distance_km": assessment["nearest_lake_distance_km"],
        "risk_rating": assessment["risk_score"],
        "elevation_drop_m": assessment["elevation_drop_H_m"],
        "peak_discharge_m3s": (round(q, 1) if q is not None else "n/a"),
        "dam_type": assessment["nearest_lake_dam_type"] or "unknown",
        "last_outburst": assessment["nearest_lake_last_outburst"] or "none on record",
    }

    cc1, cc2 = st.columns(2)
    cc1.metric("Risk rating", f'{RISK_COLOR.get(assessment["risk_score"], "")} {assessment["risk_score"]}')
    cc2.metric("Nearest range", assessment["nearest_basin"])

    prompt = build_advisory_prompt(context)
    with st.expander("View the generated prompt"):
        st.code(prompt, language="text")

    if st.button("🧠 Generate advisory", type="primary"):
        with st.spinner("Contacting LLM…"):
            advisory = generate_ai_advisory(prompt)
        st.session_state["advisory"] = advisory

    if "advisory" in st.session_state:
        st.markdown(st.session_state["advisory"])

    st.caption(
        "Advisories are decision-support only and must be validated against official "
        "national disaster-management guidance. Without an LLM API key a built-in "
        "offline template is shown."
    )
