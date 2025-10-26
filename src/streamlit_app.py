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

st.set_page_config(page_title="Velib Trend — Paris Bike Predictions", layout="wide")
st.title("🚴 Velib Trend")
st.caption("Real-time availability & AI-powered predictions for Paris bike stations")

# Sidebar with page navigation
with st.sidebar:
    st.markdown("### 🗺️ Navigation")
    page = st.radio(
        "Choose a feature:",
        ["🗺️ Explore Map", "🚴 Plan Journey"],
        key="page_selector"
    )
    
    st.markdown("---")
    
    # Mode selection (only show for Explore Map)
    if page == "🗺️ Explore Map":
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
    else:
        # Journey Planner page - set defaults
        validate = True
        refresh = False
    
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

    # ==================== JOURNEY PLANNER PAGE ====================
    elif page == "🚴 Plan Journey":
        from journey_planner import (
            geocode_address,
            plan_route,
            get_prediction_at_time,
            get_journey_verdict
        )
        
        st.markdown("## 🚴 Plan Your Journey")
        st.markdown("Enter your start and destination addresses, and we'll find the best stations for you!")
        
        # Input form
        col1, col2 = st.columns(2)
        with col1:
            start_address = st.text_input(
                "📍 From (start location)",
                placeholder="e.g., 24 Rue de Rivoli, Paris",
                help="Enter your starting address"
            )
        with col2:
            dest_address = st.text_input(
                "🎯 To (destination)",
                placeholder="e.g., Gare du Nord, Paris",
                help="Enter your destination address"
            )
        
        # Plan route button
        plan_button = st.button("🔍 Plan My Route", type="primary", use_container_width=True)
        
        if plan_button:
            if not start_address or not dest_address:
                st.warning("⚠️ Please enter both start and destination addresses")
            elif start_address.lower() == dest_address.lower():
                st.error("❌ Start and destination are the same. Please enter different addresses.")
            else:
                try:
                    # Step 1: Geocode addresses
                    with st.spinner("🌍 Finding locations..."):
                        start_lat, start_lon = geocode_address(start_address)
                        dest_lat, dest_lon = geocode_address(dest_address)
                    
                    if not start_lat or not dest_lat:
                        if not start_lat:
                            st.error(f"❌ Could not find location: '{start_address}'")
                        if not dest_lat:
                            st.error(f"❌ Could not find location: '{dest_address}'")
                        st.info("💡 Try being more specific (add 'Paris' or postal code)")
                    else:
                        st.success(f"✅ Locations found!")
                        
                        # Step 2: Plan route
                        with st.spinner("🚴 Finding best stations and calculating route..."):
                            route = plan_route(start_lat, start_lon, dest_lat, dest_lon, df)
                        
                        start_station = route['start_station']
                        end_station = route['end_station']
                        st.success(f"✅ Route calculated!")
                        
                        # Check if same station
                        if start_station['stationcode'] == end_station['stationcode']:
                            st.warning("⚠️ Start and destination are very close - same station recommended!")
                            st.info(f"🚴 Station: **{start_station['name']}**")
                        else:
                            # Step 3: Get predictions in parallel
                            with st.spinner("🔮 Getting ML predictions... (first request may take 30-40s to fetch historical data, then cached for 30min)"):
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
                                    
                                    # Step 4: Get verdict
                                    verdict = get_journey_verdict(
                                        start_pred['bikes_predicted'],
                                        end_pred['docks_predicted']
                                    )
                                    
                                    st.success(f"✅ Route planned successfully!")
                                
                                except requests.exceptions.Timeout:
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
                                    st.error(f"❌ **Connection error**: {str(e)}")
                                    st.info("💡 Make sure FastAPI is running on http://127.0.0.1:8000")
                                    verdict = None
                                    start_pred = None
                                    end_pred = None
                            
                            # Display results (only if predictions succeeded)
                            if verdict is not None:
                                st.markdown("---")
                                st.markdown("## 📊 Your Journey")
                                
                                # Verdict banner
                                if verdict['status'] == 'success':
                                    st.success(f"### {verdict['icon']} {verdict['verdict']}")
                                elif verdict['status'] == 'warning':
                                    st.warning(f"### {verdict['icon']} {verdict['verdict']}")
                                else:
                                    st.error(f"### {verdict['icon']} {verdict['verdict']}")
                                
                                st.markdown(verdict['details'])
                                
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
                                
                                # Station details
                                st.markdown("### 🚲 Start Station")
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"**{start_station['name']}**")
                                    st.caption(f"📍 {route['walk_to_start_km']*1000:.0f}m from your location ({route['walk_to_start_min']:.0f} min walk)")
                                with col2:
                                    confidence_emoji = "🟢" if start_pred['confidence'] == 'high' else "🟡" if start_pred['confidence'] == 'medium' else "🟠"
                                    st.metric(
                                        f"🔮 In {route['arrival_at_start_min']:.0f} min",
                                        f"~{start_pred['bikes_predicted']:.0f} bikes",
                                        f"{confidence_emoji} {start_pred['confidence']} confidence"
                                    )
                                
                                st.markdown("### 🅿️ End Station")
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"**{end_station['name']}**")
                                    st.caption(f"📍 {route['walk_from_end_km']*1000:.0f}m from destination ({route['walk_from_end_min']:.0f} min walk)")
                                with col2:
                                    confidence_emoji = "🟢" if end_pred['confidence'] == 'high' else "🟡" if end_pred['confidence'] == 'medium' else "🟠"
                                    st.metric(
                                        f"🔮 In {route['arrival_at_end_min']:.0f} min",
                                        f"~{end_pred['docks_predicted']:.0f} docks",
                                        f"{confidence_emoji} {end_pred['confidence']} confidence"
                                    )
                            
                            # Map visualization (show regardless of prediction success)
                            st.markdown("### 🗺️ Route Map")
                            
                            # Create map data
                            map_data = pd.DataFrame({
                                'lat': [start_lat, start_station['lat'], end_station['lat'], dest_lat],
                                'lon': [start_lon, start_station['lon'], end_station['lon'], dest_lon],
                                'type': ['start', 'start_station', 'end_station', 'destination'],
                                'name': ['Your location', start_station['name'], end_station['name'], 'Destination']
                            })
                            
                            # Color mapping
                            color_map = {
                                'start': [255, 0, 0, 160],  # Red
                                'start_station': [0, 255, 0, 200],  # Green
                                'end_station': [0, 0, 255, 200],  # Blue
                                'destination': [255, 0, 0, 160]  # Red
                            }
                            map_data['color'] = map_data['type'].map(color_map)
                            
                            # Create pydeck map
                            view_state = pdk.ViewState(
                                latitude=(start_lat + dest_lat) / 2,
                                longitude=(start_lon + dest_lon) / 2,
                                zoom=13,
                                pitch=0
                            )
                            
                            layer = pdk.Layer(
                                'ScatterplotLayer',
                                data=map_data,
                                get_position='[lon, lat]',
                                get_color='color',
                                get_radius=100,
                                pickable=True
                            )
                            
                            tooltip = {
                                "html": "<b>{name}</b><br/>{type}",
                                "style": {"backgroundColor": "steelblue", "color": "white"}
                            }
                            
                            deck = pdk.Deck(
                                layers=[layer],
                                initial_view_state=view_state,
                                tooltip=tooltip,
                                map_style='mapbox://styles/mapbox/light-v10'
                            )
                            
                            st.pydeck_chart(deck)
                            
                            # Legend
                            st.markdown("**Legend:** 🔴 You/Destination · 🟢 Start Station · 🔵 End Station")
                
                except Exception as e:
                    import traceback
                    st.error(f"❌ Error planning journey: {str(e)}")
                    st.code(traceback.format_exc())
                    st.info("💡 Make sure FastAPI is running and try again")

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("💡 Make sure FastAPI is running on http://127.0.0.1:8000")
