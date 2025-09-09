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
    st.markdown("### Options")
    validate = st.checkbox("Validate & coerce types", value=True)
    refresh = st.button("Refresh data")

    st.markdown("### Mode")
    if "mode" not in st.session_state:
        st.session_state.mode = "bike"
    mcol1, mcol2 = st.columns(2)
    rerun_needed = False
    if mcol1.button("Find a bike", type="primary" if st.session_state.mode == "bike" else "secondary"):
        if st.session_state.mode != "bike":
            st.session_state.mode = "bike"
            rerun_needed = True
    if mcol2.button("Find a dock", type="primary" if st.session_state.mode == "dock" else "secondary"):
        if st.session_state.mode != "dock":
            st.session_state.mode = "dock"
            rerun_needed = True
    if rerun_needed:
        try:
            st.rerun()
        except AttributeError:  # fallback for older Streamlit
            st.experimental_rerun()
    mode = st.session_state.mode

@st.cache_data(ttl=60)
def load_data(validate: bool) -> pd.DataFrame:
    # Always fetch all stations
    params = {"all": "true"}
    if validate:
        params["validate"] = "true"
    url = f"{API_BASE}/stations"
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    payload = r.json()
    records = payload.get("records", [])
    df = pd.DataFrame(records)
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
    if df.empty:
        return df
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[ordered]

if refresh:
    load_data.clear()

status = st.empty()
start = time.time()
try:
    df = load_data(validate=validate)
    elapsed = time.time() - start
    status.success(f"Loaded {len(df)} stations (all) in {elapsed:.2f}s from {API_BASE}")

    # Simple color-coded circles (fixed radius) by availability percentage (bike or dock mode)
    if not df.empty:
        df["numbikesavailable"] = pd.to_numeric(df.get("numbikesavailable"), errors="coerce")
        df["numdocksavailable"] = pd.to_numeric(df.get("numdocksavailable"), errors="coerce")
        if "capacity" in df.columns:
            df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
        else:
            df["capacity"] = (df["numbikesavailable"].fillna(0) + df["numdocksavailable"].fillna(0))
        df["capacity"] = df["capacity"].fillna(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = df["numbikesavailable"] / df["capacity"].replace(0, np.nan)
        pct = pct.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1)
        df["pct_bikes"] = pct
        with np.errstate(divide='ignore', invalid='ignore'):
            pctd = df["numdocksavailable"] / df["capacity"].replace(0, np.nan)
        pctd = pctd.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1)
        df["pct_docks"] = pctd

        # Select metric based on mode
        mode = st.session_state.get("mode", "bike")
        metric_col = "pct_bikes" if mode == "bike" else "pct_docks"
        df["metric"] = df[metric_col]

        def pct_to_color(row):
            cap = row.get("capacity", 0)
            if pd.isna(cap) or cap == 0:
                return [0, 0, 0, 220]  # black
            v = float(row["metric"])
            if v < 0.3:
                return [230, 57, 70, 220]  # red
            if v < 0.6:
                return [253, 180, 70, 220]  # orange-ish
            return [70, 160, 60, 220]      # green

        # Apply row-wise for clarity
        df["color"] = df.apply(pct_to_color, axis=1)

        df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
        df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
        map_df = df.dropna(subset=["lat", "lon"]).copy()
        if not map_df.empty:
            center_lat = float(np.nanmean(map_df["lat"]))
            center_lon = float(np.nanmean(map_df["lon"]))

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=55,
                pickable=True,
                radius_min_pixels=4,
                radius_max_pixels=25,
            )
            if mode == "bike":
                metric_label = "Bikes"
            else:
                metric_label = "Docks"
            tooltip = {
                "html": "<b>{name}</b><br/>Bikes: {numbikesavailable}<br/>Docks: {numdocksavailable}<br/>Mode: " + metric_label,
                "style": {"backgroundColor": "#1E1E1E", "color": "white"},
            }
            view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11)
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="light")
            st.pydeck_chart(deck, use_container_width=True)

            # Legend + quick stats
            avg_bikes = map_df["numbikesavailable"].fillna(0).mean()
            avg_docks = map_df["numdocksavailable"].fillna(0).mean()
            total_capacity = map_df["capacity"].fillna(0).sum()
            if total_capacity > 0:
                fleet_util = (map_df["numbikesavailable"].fillna(0).sum() / total_capacity) * 100
            else:
                fleet_util = 0.0

            col_a, col_b = st.columns([1,2])
            with col_a:
                st.markdown("#### Légende")
                if mode == "bike":
                    st.markdown("""
<div style='line-height:1.2'>
<span style='display:inline-block;width:14px;height:14px;background:#46A03C;border-radius:50%;margin-right:6px;'></span> ≥ 60% vélos<br>
<span style='display:inline-block;width:14px;height:14px;background:#FDB446;border-radius:50%;margin-right:6px;'></span> 30–59% vélos<br>
<span style='display:inline-block;width:14px;height:14px;background:#E63946;border-radius:50%;margin-right:6px;'></span> < 30% vélos<br>
<span style='display:inline-block;width:14px;height:14px;background:#000;border-radius:50%;margin-right:6px;'></span> capacité inconnue
</div>
""", unsafe_allow_html=True)
                else:
                    st.markdown("""
<div style='line-height:1.2'>
<span style='display:inline-block;width:14px;height:14px;background:#46A03C;border-radius:50%;margin-right:6px;'></span> ≥ 60% docks libres<br>
<span style='display:inline-block;width:14px;height:14px;background:#FDB446;border-radius:50%;margin-right:6px;'></span> 30–59% docks libres<br>
<span style='display:inline-block;width:14px;height:14px;background:#E63946;border-radius:50%;margin-right:6px;'></span> < 30% docks libres<br>
<span style='display:inline-block;width:14px;height:14px;background:#000;border-radius:50%;margin-right:6px;'></span> capacité inconnue
</div>
""", unsafe_allow_html=True)
            with col_b:
                st.markdown("#### Stats rapides")
                st.markdown(f"Stations: **{len(map_df)}**  | Utilisation flotte: **{fleet_util:.1f}%**  | Moy. vélos/station: **{avg_bikes:.1f}**  | Moy. docks/station: **{avg_docks:.1f}**")

except Exception as e:
    status.error(f"Error loading data: {e}")
