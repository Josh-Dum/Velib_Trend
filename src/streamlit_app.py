import os
import time
import requests
import pandas as pd
import streamlit as st
import pydeck as pdk
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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
                "html": "<b>{name}</b><br/>Station: {stationcode}<br/>Bikes: {numbikesavailable}<br/>Docks: {numdocksavailable}",
                "style": {"backgroundColor": "#1E1E1E", "color": "white"},
            }
            view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11)
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="light")
            st.pydeck_chart(deck, use_container_width=True)
            
            # Station selection for predictions
            st.markdown("---")
            st.markdown("### 🔮 Historique & Prédictions")
            st.markdown("💡 **Astuce**: Survolez les stations sur la carte pour voir leur code, puis entrez-le ci-dessous")
            
            col_input, col_button = st.columns([3, 1])
            with col_input:
                station_code_input = st.text_input(
                    "Code de la station",
                    placeholder="Ex: 16107",
                    help="Survolez une station sur la carte pour voir son code"
                )
            with col_button:
                st.markdown("<br>", unsafe_allow_html=True)  # Align button
                predict_button = st.button("📊 Voir l'historique + prédictions", type="primary")
            
            if predict_button and station_code_input:
                # Find station in dataframe
                station_data = map_df[map_df['stationcode'].astype(str) == str(station_code_input)]
                
                if station_data.empty:
                    st.error(f"❌ Station {station_code_input} introuvable")
                else:
                    selected_station = station_data.iloc[0]
                    station_code = str(selected_station['stationcode'])
                    station_name = selected_station['name']
                    current_bikes = selected_station['numbikesavailable']
                    current_docks = selected_station['numdocksavailable']
                    capacity = selected_station['capacity']
                    
                    # Display station header
                    st.markdown(f"#### 🚲 **{station_name}**")
                    st.caption(f"Code: {station_code} • Capacité: {capacity} places")
                    
                    # Current status
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🚴 Vélos actuels", int(current_bikes))
                    with col2:
                        st.metric("🅿️ Places libres", int(current_docks))
                    with col3:
                        occupancy = (current_bikes / capacity * 100) if capacity > 0 else 0
                        st.metric("📊 Taux d'occupation", f"{occupancy:.0f}%")
                    
                    # Fetch historical data + predictions
                    with st.spinner("⏳ Loading historical data and predictions... (5-10 seconds)"):
                        try:
                            pred_url = f"{API_BASE}/predict/{station_code}"
                            pred_response = requests.get(pred_url, timeout=30)
                            pred_response.raise_for_status()
                            pred_data = pred_response.json()
                            
                            # Extract data
                            historical = pred_data.get('historical_24h', [])
                            predictions_dict = pred_data.get('predictions', {})
                            
                            # Check if we have data
                            if not historical:
                                st.warning("⚠️ No historical data available for this station")
                            else:
                                # Create visualization with plotly
                                fig = go.Figure()
                                
                                # Convert historical data to Paris time
                                hist_times = []
                                hist_bikes = []
                                for h in historical:
                                    try:
                                        # Ensure we have valid data
                                        if not isinstance(h, dict) or 'time' not in h or 'bikes' not in h:
                                            continue
                                        
                                        dt_utc = datetime.fromisoformat(h['time'].replace('Z', '+00:00'))
                                        dt_paris = dt_utc.astimezone(ZoneInfo('Europe/Paris'))
                                        bikes_val = int(h['bikes'])
                                        
                                        hist_times.append(dt_paris)
                                        hist_bikes.append(bikes_val)
                                    except Exception as e:
                                        # Skip invalid data points
                                        continue
                                
                                # Add historical trace
                                if hist_times and all(isinstance(t, datetime) for t in hist_times):
                                    fig.add_trace(go.Scatter(
                                        x=hist_times,
                                        y=hist_bikes,
                                        mode='lines+markers',
                                        name='Historical (24h)',
                                        line=dict(color='#1f77b4', width=2),
                                        marker=dict(size=6),
                                        hovertemplate='<b>%{x|%H:%M}</b><br>Bikes: %{y}<extra></extra>'
                                    ))
                                elif hist_times:
                                    st.warning(f"⚠️ Invalid historical time data types detected")
                                
                                # Add predictions if available
                                if predictions_dict:
                                    pred_times = []
                                    pred_bikes = []
                                    
                                    for key in ['T+1h', 'T+2h', 'T+3h']:
                                        if key in predictions_dict:
                                            try:
                                                p = predictions_dict[key]
                                                # Ensure we have valid data
                                                if not isinstance(p, dict) or 'time' not in p or 'bikes' not in p:
                                                    continue
                                                
                                                dt_utc = datetime.fromisoformat(p['time'].replace('Z', '+00:00'))
                                                dt_paris = dt_utc.astimezone(ZoneInfo('Europe/Paris'))
                                                bikes_val = int(p['bikes'])
                                                
                                                pred_times.append(dt_paris)
                                                pred_bikes.append(bikes_val)
                                            except Exception as e:
                                                # Skip invalid prediction
                                                continue
                                    
                                    if pred_times and all(isinstance(t, datetime) for t in pred_times):
                                        fig.add_trace(go.Scatter(
                                            x=pred_times,
                                            y=pred_bikes,
                                            mode='lines+markers',
                                            name='Predictions',
                                            line=dict(color='#ff7f0e', width=3, dash='dash'),
                                            marker=dict(size=10, symbol='diamond'),
                                            hovertemplate='<b>%{x|%H:%M}</b><br>Predicted: %{y} bikes<extra></extra>'
                                        ))
                                    elif pred_times:
                                        st.warning(f"⚠️ Invalid prediction time data types detected")
                                
                                # Add current time vertical line (only if we have valid traces)
                                if len(fig.data) > 0:
                                    try:
                                        current_time = datetime.now(ZoneInfo('Europe/Paris'))
                                        fig.add_vline(
                                            x=current_time,
                                            line_dash="dot",
                                            line_color="red",
                                            line_width=2,
                                            annotation_text="🕐 Now",
                                            annotation_position="top right"
                                        )
                                    except Exception as vline_error:
                                        # Silently skip vline - graph works without it
                                        pass
                                
                                # Update layout
                                fig.update_layout(
                                    title=f"📈 Availability: {station_name}",
                                    xaxis_title="Time (Paris)",
                                    yaxis_title="Number of available bikes",
                                    hovermode='x unified',
                                    showlegend=True,
                                    height=500,
                                    template="plotly_white"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show prediction details below chart
                                if predictions_dict:
                                    st.markdown("### 🎯 Prediction Details")
                                    col1, col2, col3 = st.columns(3)
                                    
                                    for i, (col, key) in enumerate(zip([col1, col2, col3], ['T+1h', 'T+2h', 'T+3h'])):
                                        if key in predictions_dict:
                                            with col:
                                                pred = predictions_dict[key]
                                                try:
                                                    dt_utc = datetime.fromisoformat(pred['time'].replace('Z', '+00:00'))
                                                    dt_paris = dt_utc.astimezone(ZoneInfo('Europe/Paris'))
                                                    time_str = dt_paris.strftime("%H:%M")
                                                except Exception:
                                                    time_str = "??:??"
                                                
                                                try:
                                                    # Extract bikes value (convert to int)
                                                    bikes_raw = pred.get('bikes', 0)
                                                    if isinstance(bikes_raw, (int, float)):
                                                        bikes = int(bikes_raw)
                                                    else:
                                                        st.error(f"⚠️ Invalid bikes value: {bikes_raw} (type: {type(bikes_raw)})")
                                                        continue
                                                    
                                                    # Calculate change
                                                    current_bikes_int = int(current_bikes)
                                                    change = bikes - current_bikes_int
                                                    
                                                    # Debug: verify change type
                                                    if not isinstance(change, (int, float)):
                                                        st.error(f"⚠️ Change is not a number! change={change}, type={type(change)}")
                                                        st.write(f"bikes={bikes} (type: {type(bikes)})")
                                                        st.write(f"current_bikes_int={current_bikes_int} (type: {type(current_bikes_int)})")
                                                        continue
                                                    
                                                    # Format delta string separately to avoid formatting issues
                                                    if change != 0:
                                                        delta_str = f"{int(change):+d} bikes"
                                                    else:
                                                        delta_str = "stable"
                                                    
                                                    st.metric(
                                                        f"⏰ {time_str} (Paris)",
                                                        f"{bikes} bikes",
                                                        delta=delta_str,
                                                        delta_color="normal" if change == 0 else ("inverse" if change < 0 else "normal")
                                                    )
                                                except (ValueError, TypeError) as e:
                                                    st.error(f"❌ Error displaying prediction: {e}")
                                                    st.write(f"Debug - pred dict: {pred}")
                                                    st.write(f"Debug - current_bikes: {current_bikes} (type: {type(current_bikes)})")
                            
                            # Show metadata
                            st.markdown("---")
                            model_info = pred_data.get("model", {})
                            metadata = pred_data.get("metadata", {})
                            is_simulated = metadata.get("simulated_history", False)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.caption(f"🤖 Model: {model_info.get('version', 'unknown')}")
                            with col2:
                                st.caption(f"⚡ Inference time: {model_info.get('inference_time_ms', 0):.0f}ms")
                            with col3:
                                if is_simulated:
                                    st.caption("📊 Simulated data")
                                    st.warning("⚠️ Simulated history (S3 unavailable)")
                                else:
                                    st.caption("✅ Real S3 data")
                        
                        except requests.exceptions.Timeout:
                            st.error("⏱️ **Timeout**: Server taking too long to respond (>30s)")
                            st.info("💡 Try again in a few seconds")
                        except requests.exceptions.RequestException as e:
                            st.error(f"❌ **Connection error**: {str(e)}")
                            st.info("💡 Check that FastAPI is running on http://127.0.0.1:8000")
                        except Exception as e:
                            import traceback
                            st.error(f"❌ **Error**: {str(e)}")
                            st.code(traceback.format_exc())

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
