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

st.set_page_config(page_title="Velib Trend — Paris Bike Predictions", layout="wide")
st.title("🚴 Velib Trend")
st.caption("Real-time availability & AI-powered predictions for Paris bike stations")

# Sidebar with minimal controls
with st.sidebar:
    st.markdown("### 🔍 Mode")
    if "mode" not in st.session_state:
        st.session_state.mode = "bike"
    mode = st.radio(
        "What are you looking for?",
        ["bike", "dock"],
        format_func=lambda x: "🚴 Find a bike" if x == "bike" else "🅿️ Find a dock",
        key="mode_selector"
    )
    st.session_state.mode = mode
    
    st.markdown("---")
    
    # Collapsible advanced options
    with st.expander("⚙️ Advanced Options"):
        validate = st.checkbox("Validate data types", value=True)
        refresh = st.button("🔄 Refresh data")
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.caption("ML predictions powered by LSTM neural network")
    st.caption("Data updates hourly via AWS Lambda")

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

# Load data silently (no status message unless error)
try:
    df = load_data(validate=validate)

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
        
        # ============================================================
        # SEARCH BAR AT THE TOP (PROMINENT)
        # ============================================================
        st.markdown("---")
        st.markdown("## 🔍 Find a Station")
        
        # Create searchable options with station name and code
        station_options = {}
        station_lookup = {}  # Map display_text -> station data
        for _, row in map_df.iterrows():
            station_name = row.get('name', 'Unknown')
            station_code = str(row.get('stationcode', ''))
            # Format: "Station Name (Code)"
            display_text = f"{station_name} ({station_code})"
            station_options[display_text] = station_code
            station_lookup[display_text] = row
        
        # Sort by station name
        sorted_options = [""] + sorted(station_options.keys())
        
        selected_station = st.selectbox(
            "Search by station name",
            options=sorted_options,
            help="Start typing a station name (e.g., 'République', 'Bastille', 'Louvre')",
            placeholder="Type to search...",
            label_visibility="collapsed"
        )
        
        # Get the station code from selection
        station_code_input = station_options.get(selected_station, "") if selected_station else ""
        
        # ============================================================
        # MAP WITH ZOOM TO SELECTED STATION
        # ============================================================
        if not map_df.empty:
            # If a station is selected, zoom to it
            if station_code_input and selected_station in station_lookup:
                selected_row = station_lookup[selected_station]
                center_lat = float(selected_row["lat"])
                center_lon = float(selected_row["lon"])
                zoom_level = 15  # Zoomed in
            else:
                # Default: show all Paris
                center_lat = float(np.nanmean(map_df["lat"]))
                center_lon = float(np.nanmean(map_df["lon"]))
                zoom_level = 11  # Zoomed out

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
            view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom_level)
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="light")
            
            # Display the map
            st.pydeck_chart(deck, use_container_width=True)
            
            # Collapsible legend and stats
            with st.expander("ℹ️ Map Legend & Statistics"):
                col_legend, col_stats = st.columns([1, 2])
                with col_legend:
                    st.markdown("#### Color Code")
                    if mode == "bike":
                        st.markdown("""
<div style='line-height:1.4'>
<span style='display:inline-block;width:12px;height:12px;background:#46A03C;border-radius:50%;margin-right:6px;'></span> ≥ 60% bikes<br>
<span style='display:inline-block;width:12px;height:12px;background:#FDB446;border-radius:50%;margin-right:6px;'></span> 30–59% bikes<br>
<span style='display:inline-block;width:12px;height:12px;background:#E63946;border-radius:50%;margin-right:6px;'></span> < 30% bikes<br>
<span style='display:inline-block;width:12px;height:12px;background:#000;border-radius:50%;margin-right:6px;'></span> Unknown
</div>
""", unsafe_allow_html=True)
                    else:
                        st.markdown("""
<div style='line-height:1.4'>
<span style='display:inline-block;width:12px;height:12px;background:#46A03C;border-radius:50%;margin-right:6px;'></span> ≥ 60% docks free<br>
<span style='display:inline-block;width:12px;height:12px;background:#FDB446;border-radius:50%;margin-right:6px;'></span> 30–59% docks free<br>
<span style='display:inline-block;width:12px;height:12px;background:#E63946;border-radius:50%;margin-right:6px;'></span> < 30% docks free<br>
<span style='display:inline-block;width:12px;height:12px;background:#000;border-radius:50%;margin-right:6px;'></span> Unknown
</div>
""", unsafe_allow_html=True)
                
                with col_stats:
                    st.markdown("#### Network Statistics")
                    avg_bikes = map_df["numbikesavailable"].fillna(0).mean()
                    avg_docks = map_df["numdocksavailable"].fillna(0).mean()
                    total_capacity = map_df["capacity"].fillna(0).sum()
                    if total_capacity > 0:
                        fleet_util = (map_df["numbikesavailable"].fillna(0).sum() / total_capacity) * 100
                    else:
                        fleet_util = 0.0
                    
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("Stations", f"{len(map_df)}")
                    with col_s2:
                        st.metric("Fleet Usage", f"{fleet_util:.1f}%")
                    with col_s3:
                        st.metric("Avg Bikes/Station", f"{avg_bikes:.1f}")
            
            # ============================================================
            # STATION DETAILS & PREDICTIONS
            # ============================================================
            if station_code_input:
                st.markdown("---")
                st.markdown("## 📊 Station Details & Predictions")
                
                # Find station in dataframe
                station_data = map_df[map_df['stationcode'].astype(str) == str(station_code_input)]
                
                if station_data.empty:
                    st.error(f"❌ Station {station_code_input} not found")
                else:
                    selected_station_row = station_data.iloc[0]
                    station_code = str(selected_station_row['stationcode'])
                    station_name = selected_station_row['name']
                    current_bikes = selected_station_row['numbikesavailable']
                    current_docks = selected_station_row['numdocksavailable']
                    capacity = selected_station_row['capacity']
                    
                    # Display station header (cleaner)
                    st.markdown(f"### 🚲 {station_name}")
                    st.caption(f"Station Code: {station_code} • Capacity: {int(capacity)} spaces")
                    
                    # Current status
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🚴 Available Bikes", int(current_bikes))
                    with col2:
                        st.metric("🅿️ Free Docks", int(current_docks))
                    with col3:
                        occupancy = (current_bikes / capacity * 100) if capacity > 0 else 0
                        st.metric("📊 Occupancy", f"{occupancy:.0f}%")
                    
                    # Fetch historical data + predictions
                    with st.spinner("⏳ Loading predictions... (5-10 seconds)"):
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
                                
                                # Add historical trace with smooth curve
                                if hist_times and all(isinstance(t, datetime) for t in hist_times):
                                    fig.add_trace(go.Scatter(
                                        x=hist_times,
                                        y=hist_bikes,
                                        mode='lines+markers',
                                        name='Historical (24h)',
                                        line=dict(color='#1f77b4', width=2.5, shape='spline'),
                                        marker=dict(size=5, color='#1f77b4'),
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
                                        # Add smooth connection line from last historical point to first prediction
                                        if hist_times and hist_bikes:
                                            # Create smooth connecting segment with intermediate points
                                            from datetime import timedelta
                                            
                                            start_time = hist_times[-1]
                                            end_time = pred_times[0]
                                            start_bikes = hist_bikes[-1]
                                            end_bikes = pred_bikes[0]
                                            
                                            # Create intermediate points for smooth curve
                                            time_diff = (end_time - start_time).total_seconds()
                                            num_points = 5  # More points = smoother curve
                                            
                                            connect_times = [start_time]
                                            connect_bikes = [start_bikes]
                                            
                                            for i in range(1, num_points):
                                                ratio = i / num_points
                                                intermediate_time = start_time + timedelta(seconds=time_diff * ratio)
                                                # Linear interpolation for bikes
                                                intermediate_bikes = start_bikes + (end_bikes - start_bikes) * ratio
                                                connect_times.append(intermediate_time)
                                                connect_bikes.append(intermediate_bikes)
                                            
                                            connect_times.append(end_time)
                                            connect_bikes.append(end_bikes)
                                            
                                            fig.add_trace(go.Scatter(
                                                x=connect_times,
                                                y=connect_bikes,
                                                mode='lines',
                                                name='Connection',
                                                line=dict(color='#999999', width=2, dash='dot', shape='spline'),
                                                showlegend=False,
                                                hoverinfo='skip'
                                            ))
                                        
                                        # Add predictions trace with smooth curve
                                        fig.add_trace(go.Scatter(
                                            x=pred_times,
                                            y=pred_bikes,
                                            mode='lines+markers',
                                            name='AI Predictions',
                                            line=dict(color='#ff7f0e', width=3, shape='spline'),
                                            marker=dict(size=10, symbol='diamond', color='#ff7f0e'),
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
                                
                                # Update layout with better styling
                                fig.update_layout(
                                    title={
                                        'text': f"📈 24-Hour History & AI Predictions",
                                        'x': 0.5,
                                        'xanchor': 'center'
                                    },
                                    xaxis_title="Time (Paris local time)",
                                    yaxis_title="Available bikes",
                                    hovermode='x unified',
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    ),
                                    height=450,
                                    template="plotly_white",
                                    margin=dict(l=50, r=50, t=80, b=50)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show prediction details below chart
                                if predictions_dict:
                                    st.markdown("#### 🎯 Predictions")
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
                            
                            # Show metadata in collapsible section
                            with st.expander("🔧 Technical Details"):
                                model_info = pred_data.get("model", {})
                                metadata = pred_data.get("metadata", {})
                                is_simulated = metadata.get("simulated_history", False)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.caption(f"🤖 Model: {model_info.get('version', 'unknown')}")
                                with col2:
                                    st.caption(f"⚡ Inference: {model_info.get('inference_time_ms', 0):.0f}ms")
                                with col3:
                                    if is_simulated:
                                        st.caption("📊 Data: Simulated")
                                    else:
                                        st.caption("✅ Data: Real S3")
                                
                                if is_simulated:
                                    st.info("ℹ️ Using simulated historical data (S3 temporarily unavailable)")
                        
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

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("💡 Make sure FastAPI is running on http://127.0.0.1:8000")
