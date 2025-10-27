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
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = os.environ.get("VELIB_API_BASE", "http://127.0.0.1:8000")

# Configure page
st.set_page_config(
    page_title="Velib Trend — Paris Bike Predictions",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1f1f1f;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Station card styling */
    .station-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    /* Better button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Info boxes */
    .info-box {
        background: #f0f2f6;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header with gradient
st.markdown('<h1 class="main-title">🚴 Velib Trend</h1>', unsafe_allow_html=True)
st.markdown("**Real-time availability & AI-powered predictions for Paris bike-sharing**")
st.markdown("---")

# Sidebar with page navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/V%C3%A9lib%27_logo.svg/320px-V%C3%A9lib%27_logo.svg.png", width=150)
    
    st.markdown("### 🗺️ Navigation")
    page = st.radio(
        "Choose a feature:",
        ["🚴 Plan Journey", "🗺️ Explore Map"],
        key="page_selector"
    )
    
    st.markdown("---")
    
    # Mode selection (only show for Explore Map)
    if page == "🗺️ Explore Map":
        st.markdown("### 🔍 Display Mode")
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
            validate = st.checkbox("Validate data types", value=True, help="Ensures data quality but may be slower")
            refresh = st.button("🔄 Refresh data", help="Clear cache and reload fresh data")
    else:
        # Journey Planner page - set defaults
        validate = True
        refresh = False
    
    st.markdown("---")
    st.markdown("### 📊 About This App")
    st.markdown("""
    <div style='font-size: 0.85rem; line-height: 1.6;'>
    ✨ ML predictions powered by <b>LSTM neural network</b><br>
    📡 Data updates <b>hourly</b> via AWS Lambda<br>
    🎯 Model accuracy: <b>R²=0.815</b> @ T+1h<br>
    📍 Tracking <b>1,498 stations</b> across Paris
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 0.75rem; color: #666;'>
    Made with ❤️ using Streamlit & AWS<br>
    © 2025 Paris Pulse Project
    </div>
    """, unsafe_allow_html=True)

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
    
    # Filter out closed/uninstalled stations
    # Keep only stations where is_installed is "OUI", True, etc.
    if "is_installed" in df.columns:
        df = df[df["is_installed"].isin(["OUI", True, "true", 1])].copy()
    
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
    # Also clear session state to force reload
    if 'cached_df' in st.session_state:
        del st.session_state['cached_df']

# Use session state to persist data across reruns (avoids reloading on every interaction)
if 'cached_df' not in st.session_state or refresh:
    st.session_state['cached_df'] = load_data(validate=validate)

df = st.session_state['cached_df']

# ==================== PAGE ROUTING ====================
try:
    if page == "🗺️ Explore Map":
        # ==================== EXPLORE MAP PAGE ====================
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
        st.markdown('<div class="section-header">🔍 Find a Station</div>', unsafe_allow_html=True)
        
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
        
        # Search with help section
        col_search, col_help = st.columns([4, 1])
        with col_search:
            selected_station = st.selectbox(
                "Search by station name",
                options=sorted_options,
                help="Start typing a station name (e.g., 'République', 'Bastille', 'Louvre')",
                placeholder="Type to search...",
                label_visibility="collapsed"
            )
        with col_help:
            st.markdown("") # Spacing
            st.markdown("") # Spacing
            with st.expander("💡 Help"):
                st.markdown("""
                **Search Tips:**
                - Type any part of the station name
                - Try famous places: Louvre, Eiffel, République
                - Map will auto-zoom to your selection
                """)
        
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
                st.markdown('<div class="section-header">📊 Station Details & AI Predictions</div>', unsafe_allow_html=True)
                
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
                    
                    # Display station header with card styling
                    st.markdown(f"""
                    <div class="station-card">
                        <h2 style="margin: 0; font-size: 1.8rem;">🚲 {station_name}</h2>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                            Station Code: {station_code} • Total Capacity: {int(capacity)} spaces
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Current status with improved metrics
                    st.markdown("#### 📍 Current Availability")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🚴 Available Bikes", int(current_bikes))
                    with col2:
                        st.metric("🅿️ Free Docks", int(current_docks))
                    with col3:
                        occupancy = (current_bikes / capacity * 100) if capacity > 0 else 0
                        st.metric("📊 Occupancy", f"{occupancy:.0f}%")
                    with col4:
                        # Status indicator
                        if occupancy >= 60:
                            status = "🟢 High"
                        elif occupancy >= 30:
                            status = "🟡 Medium"
                        else:
                            status = "🔴 Low"
                        st.metric("🎯 Bike Status", status)
                    
                    # Fetch historical data + predictions
                    st.markdown("#### 🔮 AI-Powered Predictions")
                    with st.spinner("⏳ Loading ML predictions from AWS SageMaker... (5-10 seconds)"):
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
                                                    
                                                    # Verify change is a valid number
                                                    if not isinstance(change, (int, float)):
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
                                                except (ValueError, TypeError):
                                                    # Skip invalid predictions silently
                                                    pass
                                
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

    # ==================== JOURNEY PLANNER PAGE ====================
    elif page == "🚴 Plan Journey":
        from journey_planner import (
            geocode_address,
            plan_route,
            get_prediction_at_time,
            get_journey_verdict
        )
        
        st.markdown('<div class="section-header">🚴 Plan Your Journey</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <strong>🎯 How it works:</strong> Enter your start and destination, and our AI will:
        <ul style="margin-top: 0.5rem; margin-bottom: 0;">
            <li>Find the nearest Vélib' stations</li>
            <li>Predict bike/dock availability when you arrive</li>
            <li>Calculate your total journey time</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Input form with better design
        st.markdown("#### 📍 Journey Details")
        col1, col2 = st.columns(2)
        with col1:
            start_address = st.text_input(
                "📍 From (start location)",
                placeholder="e.g., 24 Rue de Rivoli, Paris",
                help="Enter your starting address - be as specific as possible"
            )
        with col2:
            dest_address = st.text_input(
                "🎯 To (destination)",
                placeholder="e.g., Gare du Nord, Paris",
                help="Enter your destination address - be as specific as possible"
            )
        
        st.markdown("")  # Spacing
        
        # Plan route button with better styling
        col_btn, col_spacer = st.columns([2, 3])
        with col_btn:
            plan_button = st.button("🔍 Plan My Route", type="primary", use_container_width=True)
        
        # Initialize route state in session_state
        if 'route_data' not in st.session_state:
            st.session_state['route_data'] = None
        
        # Process route planning
        if plan_button:
            if not start_address or not dest_address:
                st.warning("⚠️ Please enter both start and destination addresses")
            elif start_address.lower() == dest_address.lower():
                st.error("❌ Start and destination are the same. Please enter different addresses.")
            else:
                try:
                    # Step 1: Geocode addresses with progress
                    progress_text = st.empty()
                    progress_text.info("🌍 Step 1/4: Finding locations...")
                    
                    start_lat, start_lon = geocode_address(start_address)
                    dest_lat, dest_lon = geocode_address(dest_address)
                    
                    if not start_lat or not dest_lat:
                        progress_text.empty()
                        if not start_lat:
                            st.error(f"❌ Could not find location: '{start_address}'")
                        if not dest_lat:
                            st.error(f"❌ Could not find location: '{dest_address}'")
                        st.info("💡 **Tip:** Try being more specific (add 'Paris' or postal code like '75001')")
                    else:
                        progress_text.success("✅ Step 1/4: Locations found!")
                        
                        # Step 2: Plan route
                        progress_text.info("🚴 Step 2/4: Finding best stations and calculating route...")
                        route = plan_route(start_lat, start_lon, dest_lat, dest_lon, df)
                        
                        start_station = route['start_station']
                        end_station = route['end_station']
                        progress_text.success("✅ Step 2/4: Route calculated!")
                        
                        # Store route data in session state for map display
                        st.session_state['route_data'] = {
                            'start_lat': start_lat,
                            'start_lon': start_lon,
                            'dest_lat': dest_lat,
                            'dest_lon': dest_lon,
                            'start_station': start_station,
                            'end_station': end_station,
                            'route': route
                        }
                        
                        # Check if same station
                        if start_station['stationcode'] == end_station['stationcode']:
                            progress_text.empty()
                            st.warning("⚠️ **Close Proximity**: Start and destination are very close - same station recommended!")
                            st.info(f"🚴 **Suggested Station:** {start_station['name']}")
                        else:
                            # Step 3: Get predictions in parallel
                            progress_text.info("🔮 Step 3/4: Getting ML predictions from AWS SageMaker... (may take 20-30s on first request)")
                            try:
                                # Use ThreadPoolExecutor to run both predictions simultaneously
                                with ThreadPoolExecutor(max_workers=2) as executor:
                                    future_start = executor.submit(
                                        get_prediction_at_time,
                                        start_station['stationcode'],
                                        route['arrival_at_start_min'],
                                        API_BASE
                                    )
                                    future_end = executor.submit(
                                        get_prediction_at_time,
                                        end_station['stationcode'],
                                        route['arrival_at_end_min'],
                                        API_BASE
                                    )
                                    
                                    # Wait for both to complete
                                    start_pred = future_start.result()
                                    end_pred = future_end.result()
                                
                                progress_text.success("✅ Step 3/4: Predictions received!")
                                
                                # Step 4: Get verdict
                                progress_text.info("🎯 Step 4/4: Analyzing your journey...")
                                verdict = get_journey_verdict(
                                    start_pred['bikes_predicted'],
                                    end_pred['docks_predicted']
                                )
                                
                                progress_text.success("✅ Step 4/4: Journey analysis complete!")
                                time.sleep(0.5)  # Brief pause to show completion
                                progress_text.empty()
                            
                            except requests.exceptions.Timeout:
                                progress_text.empty()
                                st.error("⏱️ **Timeout**: Prediction service is taking too long (>40s)")
                                st.warning("⚠️ **Route information** (without predictions):")
                                # Show route info without predictions
                                st.markdown("---")
                                st.markdown("## 📊 Your Journey (Route Only)")
                                
                                # Time breakdown
                                st.markdown("### ⏱️ Journey Breakdown")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("🚶 Walk to Start", f"{route['walk_to_start_min']:.0f} min", f"{route['walk_to_start_km']:.2f} km")
                                with col2:
                                    st.metric("🚴 Bike Ride", f"{route['bike_time_min']:.0f} min", f"{route['bike_distance_km']:.2f} km")
                                with col3:
                                    st.metric("🚶 Walk to Dest", f"{route['walk_from_end_min']:.0f} min", f"{route['walk_from_end_km']:.2f} km")
                                with col4:
                                    st.metric("⏱️ Total Time", f"{route['total_time_min']:.0f} min")
                                
                                # Station details (without predictions)
                                st.markdown("### 🚲 Start Station")
                                st.markdown(f"**{start_station['name']}**")
                                st.caption(f"📍 {route['walk_to_start_km']*1000:.0f}m from your location ({route['walk_to_start_min']:.0f} min walk)")
                                
                                st.markdown("### 🅿️ End Station")
                                st.markdown(f"**{end_station['name']}**")
                                st.caption(f"📍 {route['walk_from_end_km']*1000:.0f}m from destination ({route['walk_from_end_min']:.0f} min walk)")
                                
                                st.info("💡 **Tip**: Check current availability manually in the 'Explore Map' tab before starting your journey")
                                
                                # Skip to map display (will be added below)
                                verdict = None
                                start_pred = None
                                end_pred = None
                            
                            except requests.exceptions.RequestException as e:
                                progress_text.empty()
                                st.error(f"❌ **Connection error**: {str(e)}")
                                st.info("💡 Make sure FastAPI is running on http://127.0.0.1:8000")
                                verdict = None
                                start_pred = None
                                end_pred = None
                            
                            # Display results (only if predictions succeeded)
                            if verdict is not None:
                                st.markdown("---")
                                st.markdown('<div class="section-header">📊 Your Journey Plan</div>', unsafe_allow_html=True)
                                
                                # Verdict banner with gradient styling
                                if verdict['status'] == 'success':
                                    banner_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                                elif verdict['status'] == 'warning':
                                    banner_color = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
                                else:
                                    banner_color = "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
                                
                                st.markdown(f"""
                                <div style="background: {banner_color}; color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                                    <h3 style="margin: 0; font-size: 1.6rem;">{verdict['icon']} {verdict['verdict']}</h3>
                                    <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.95;">{verdict['details']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Time breakdown
                                st.markdown("#### ⏱️ Journey Timeline")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("🚶 Walk to Start", f"{route['walk_to_start_min']:.0f} min", f"{route['walk_to_start_km']:.2f} km")
                                with col2:
                                    st.metric("🚴 Bike Ride", f"{route['bike_time_min']:.0f} min", f"{route['bike_distance_km']:.2f} km")
                                with col3:
                                    st.metric("🚶 Walk to Dest", f"{route['walk_from_end_min']:.0f} min", f"{route['walk_from_end_km']:.2f} km")
                                with col4:
                                    st.metric("⏱️ Total Time", f"{route['total_time_min']:.0f} min", help="Total journey time including all segments")
                                
                                # Station details with card styling
                                col_stations = st.columns(2)
                                with col_stations[0]:
                                    st.markdown("#### 🚲 Pickup Station")
                                    confidence_emoji = "🟢" if start_pred['confidence'] == 'high' else "🟡" if start_pred['confidence'] == 'medium' else "🟠"
                                    st.markdown(f"""
                                    <div style="background: #f0f2f6; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea;">
                                        <strong style="font-size: 1.1rem;">{start_station['name']}</strong><br>
                                        <span style="color: #666; font-size: 0.9rem;">� {route['walk_to_start_km']*1000:.0f}m from start ({route['walk_to_start_min']:.0f} min walk)</span><br>
                                        <br>
                                        <strong>🔮 Predicted in {route['arrival_at_start_min']:.0f} min:</strong><br>
                                        <span style="font-size: 1.4rem; color: #667eea;">~{start_pred['bikes_predicted']:.0f} bikes</span><br>
                                        <span style="font-size: 0.9rem;">{confidence_emoji} {start_pred['confidence'].title()} confidence</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col_stations[1]:
                                    st.markdown("#### 🅿️ Return Station")
                                    confidence_emoji = "🟢" if end_pred['confidence'] == 'high' else "🟡" if end_pred['confidence'] == 'medium' else "🟠"
                                    st.markdown(f"""
                                    <div style="background: #f0f2f6; padding: 1rem; border-radius: 8px; border-left: 4px solid #764ba2;">
                                        <strong style="font-size: 1.1rem;">{end_station['name']}</strong><br>
                                        <span style="color: #666; font-size: 0.9rem;">📍 {route['walk_from_end_km']*1000:.0f}m from destination ({route['walk_from_end_min']:.0f} min walk)</span><br>
                                        <br>
                                        <strong>� Predicted in {route['arrival_at_end_min']:.0f} min:</strong><br>
                                        <span style="font-size: 1.4rem; color: #764ba2;">~{end_pred['docks_predicted']:.0f} docks</span><br>
                                        <span style="font-size: 0.9rem;">{confidence_emoji} {end_pred['confidence'].title()} confidence</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Google Maps integration button
                                st.markdown("---")
                                st.markdown("#### 🗺️ Navigate with Google Maps")
                                
                                # Create Google Maps URL with waypoints (API format)
                                # Note: travelmode applies to entire route (Google Maps limitation)
                                # Set to bicycling as compromise - users can manually adjust first/last segments to walking
                                origin_coords = f"{start_lat},{start_lon}"
                                dest_coords = f"{dest_lat},{dest_lon}"
                                waypoint_a = f"{start_station['lat']},{start_station['lon']}"
                                waypoint_b = f"{end_station['lat']},{end_station['lon']}"
                                
                                # Google Maps Directions API URL
                                google_maps_url = (
                                    f"https://www.google.com/maps/dir/?api=1"
                                    f"&origin={origin_coords}"
                                    f"&destination={dest_coords}"
                                    f"&waypoints={waypoint_a}|{waypoint_b}"
                                    f"&travelmode=bicycling"
                                )
                                
                                # Display button with description
                                st.markdown(f"""
                                <a href="{google_maps_url}" target="_blank" style="text-decoration: none;">
                                    <div style="background: linear-gradient(135deg, #4285f4 0%, #34a853 100%); 
                                                color: white; padding: 1rem 1.5rem; border-radius: 8px; 
                                                text-align: center; font-weight: 600; font-size: 1.1rem;
                                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                                                transition: transform 0.2s;">
                                        🗺️ Open Route in Google Maps
                                    </div>
                                </a>
                                """, unsafe_allow_html=True)
                                
                                st.caption("📍 Opens in Google Maps with bicycling mode. Adjust first/last segments to walking if needed.")
                
                except Exception as e:
                    import traceback
                    st.error(f"❌ Error planning journey: {str(e)}")
                    st.code(traceback.format_exc())
                    st.info("💡 Make sure FastAPI is running and try again")
        
        # ============================================================
        # UNIFIED MAP: Always shown, updates based on route state
        # ============================================================
        st.markdown("---")
        st.markdown("### 🗺️ Paris Vélib' Network")
        
        # Prepare ALL stations as base layer
        all_stations_map = df[['lat', 'lon', 'name', 'stationcode', 'numbikesavailable']].copy()
        
        # Check if we have a planned route
        route_info = st.session_state.get('route_data', None)
        
        if route_info is None:
            # No route planned: Show all stations color-coded by availability
            st.caption(f"Showing all {len(df):,} Vélib' stations in Paris. Plan a route to see your journey overlay.")
            
            # Color by availability
            def get_station_color(row):
                bikes = row.get('numbikesavailable', 0)
                if bikes >= 5:
                    return [0, 200, 0, 180]  # Green - good availability
                elif bikes >= 1:
                    return [255, 165, 0, 180]  # Orange - low availability
                else:
                    return [255, 0, 0, 180]  # Red - no bikes
            
            all_stations_map['color'] = df.apply(get_station_color, axis=1).tolist()
            all_stations_map['radius'] = 50
            all_stations_map['type'] = 'station'
            
            map_data = all_stations_map
            
            # Default view: Paris center
            view_state = pdk.ViewState(
                latitude=48.8566,
                longitude=2.3522,
                zoom=12,
                pitch=0
            )
            
            tooltip = {
                "html": "<b>{name}</b><br/>Code: {stationcode}<br/>Bikes: {numbikesavailable}",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
            
        else:
            # Route planned: Show all stations (gray) + route overlay (highlighted)
            st.caption("All Vélib' stations shown in gray. Your route is highlighted in color.")
            
            # Background stations (gray, small, semi-transparent)
            all_stations_map['type'] = 'background'
            all_stations_map['color'] = [[180, 180, 180, 80]] * len(all_stations_map)  # Gray, semi-transparent
            all_stations_map['radius'] = 40  # Small circles
            
            # Extract route info from session_state
            start_lat = route_info['start_lat']
            start_lon = route_info['start_lon']
            dest_lat = route_info['dest_lat']
            dest_lon = route_info['dest_lon']
            start_station = route_info['start_station']
            end_station = route_info['end_station']
            
            # Create route points (highlighted, large, opaque)
            route_points = pd.DataFrame({
                'lat': [start_lat, start_station['lat'], end_station['lat'], dest_lat],
                'lon': [start_lon, start_station['lon'], end_station['lon'], dest_lon],
                'type': ['start', 'start_station', 'end_station', 'destination'],
                'name': ['Your location', start_station['name'], end_station['name'], 'Destination'],
                'stationcode': ['', start_station.get('stationcode', ''), end_station.get('stationcode', ''), ''],
                'numbikesavailable': [0, start_station.get('numbikesavailable', 0), end_station.get('numbikesavailable', 0), 0]
            })
            
            # Color mapping for route points
            color_map = {
                'start': [255, 0, 0, 200],  # Red
                'start_station': [0, 200, 0, 255],  # Bright green
                'end_station': [0, 100, 255, 255],  # Bright blue
                'destination': [255, 0, 0, 200]  # Red
            }
            route_points['color'] = route_points['type'].map(color_map)
            route_points['radius'] = 150  # Large circles
            
            # Combine: ALL stations (background) + route points (foreground)
            map_data = pd.concat([all_stations_map, route_points], ignore_index=True)
            
            # Zoom to route
            view_state = pdk.ViewState(
                latitude=(start_lat + dest_lat) / 2,
                longitude=(start_lon + dest_lon) / 2,
                zoom=13,
                pitch=0
            )
            
            tooltip = {
                "html": "<b>{name}</b><br/>Type: {type}<br/>Code: {stationcode}",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
        
        # Create and display the map
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position='[lon, lat]',
            get_color='color',
            get_radius='radius',
            pickable=True,
            auto_highlight=True
        )
        
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style='light'
        )
        
        st.pydeck_chart(deck, use_container_width=True)
        
        # Legend
        if route_info is None:
            st.markdown("**Station availability:** 🟢 Good (5+ bikes) · 🟠 Low (1-4 bikes) · 🔴 Empty")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Your Route:**")
                st.markdown("🔴 You/Destination · 🟢 Start Station · 🔵 End Station")
            with col2:
                st.markdown("**Network:**")
                st.markdown(f"⚪ All {len(df):,} Vélib stations (gray)")

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("💡 Make sure FastAPI is running on http://127.0.0.1:8000")
