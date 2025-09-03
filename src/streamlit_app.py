import os
import time
import requests
import pandas as pd
import streamlit as st
import pydeck as pdk
import numpy as np

API_BASE = os.environ.get("VELIB_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Velib Trend — Stations", layout="wide")
st.title("Velib Trend — Live Stations")

with st.sidebar:
    st.markdown("Settings")
    fetch_all = st.checkbox("Fetch all stations", value=True)
    validate = st.checkbox("Validate & coerce types", value=True)
    limit = st.slider("Limit (used if not fetching all)", min_value=1, max_value=200, value=50)
    refresh = st.button("Refresh")

@st.cache_data(ttl=60)
def load_data(fetch_all: bool, validate: bool, limit: int) -> pd.DataFrame:
    params = {}
    if fetch_all:
        params["all"] = "true"
    else:
        params["limit"] = str(limit)
    if validate:
        params["validate"] = "true"
    url = f"{API_BASE}/stations"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    recs = payload.get("records", [])
    df = pd.DataFrame(recs)
    # Reorder common columns if present
    preferred = [
        "stationcode",
        "name",
        "lat",
        "lon",
        "numbikesavailable",
        "numdocksavailable",
        "mechanical",
        "ebike",
        "capacity",
        "duedate",
        "nom_arrondissement_communes",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    if df.empty:
        return df
    return df[cols]

if refresh:
    load_data.clear()

status = st.empty()
start = time.time()
try:
    df = load_data(fetch_all=fetch_all, validate=validate, limit=limit)
    elapsed = time.time() - start
    status.success(f"Loaded {len(df)} stations in {elapsed:.2f}s from {API_BASE}")

    # Compute capacity and availability ratio (simple and safe)
    if not df.empty:
        # Coerce numerics
        df["numbikesavailable"] = pd.to_numeric(df.get("numbikesavailable"), errors="coerce")
        df["numdocksavailable"] = pd.to_numeric(df.get("numdocksavailable"), errors="coerce")
        if "capacity" in df.columns:
            df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
        else:
            df["capacity"] = (df["numbikesavailable"].fillna(0) + df["numdocksavailable"].fillna(0))

        # Safe ratio: avoid division by zero/NA
        safe_cap = df["capacity"].replace(0, np.nan)
        df["avail_ratio"] = (df["numbikesavailable"] / safe_cap).replace([np.inf, -np.inf], np.nan)
        df["avail_ratio"] = df["avail_ratio"].fillna(0.0).clip(lower=0.0, upper=1.0)

        # Color mapping (handles NaN as 0.0)
        def ratio_to_color(r):
            try:
                val = float(r)
            except Exception:
                val = 0.0
            if val < 0.3:
                return [230, 57, 70]  # red
            if val < 0.6:
                return [253, 203, 110]  # yellow
            return [76, 175, 80]  # green

        df["color"] = df["avail_ratio"].apply(ratio_to_color)

        # Ensure coords numeric, then drop missing
        df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
        df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
        map_df = df.dropna(subset=["lat", "lon"]).copy()

        # Center the map on the mean lat/lon (guard against all-NaN)
        if not map_df.empty:
            center_lat = float(np.nanmean(map_df["lat"]))
            center_lon = float(np.nanmean(map_df["lon"]))

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=60,
                pickable=True,
                radius_min_pixels=3,
                radius_max_pixels=30,
            )

            tooltip = {
                "html": "<b>{name}</b><br/>Bikes: {numbikesavailable}<br/>Docks: {numdocksavailable}<br/>Ratio: {avail_ratio}",
                "style": {"backgroundColor": "#1E1E1E", "color": "white"},
            }

            view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11)
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="light")
            st.pydeck_chart(deck, use_container_width=True)

    st.markdown("### Sample of stations")
    st.dataframe(df.head(50), use_container_width=True)
except Exception as e:
    status.error(f"Error loading data: {e}")
