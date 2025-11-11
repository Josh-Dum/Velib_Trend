import os
import time
import html
import textwrap
from typing import List, Dict, Tuple
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = os.environ.get("VELIB_API_BASE", "http://127.0.0.1:8000")

TIMELINE_FONT_STACK = "'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
AI_PREDICTION_ICON = textwrap.dedent(
        """
        <svg width="100" height="100" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <g stroke="currentColor" stroke-width="7" fill="none" stroke-linecap="round" stroke-linejoin="round">
                <path d="M50 20 Q55 40 80 50 Q55 60 50 80 Q45 60 20 50 Q45 40 50 20 Z"/>
                <path d="M78 17 Q80 24 89 28 Q80 32 78 39 Q76 32 67 28 Q76 24 78 17 Z"/>
                <path d="M77 64 Q79 69 85 71 Q79 73 77 78 Q75 73 69 71 Q75 69 77 64 Z"/>
            </g>
        </svg>
        """
).strip()

# ============================================================
# UTILITY FUNCTIONS - Moved to top for reusability
# ============================================================

def get_availability_color(value):
    """
    Return hex color based on availability count (absolute numbers).
    
    Color scheme (count-based):
    - Green (#5DBB63): 5+ available - good availability
    - Orange (#F39C12): 1-4 available - low availability
    - Red (#E74C3C): 0 available - no availability
    - Gray (#6B7280): Unknown/NA
    
    Args:
        value: Number of bikes or docks available
        
    Returns:
        str: Hex color code
    """
    if pd.isna(value):
        return "#6B7280"  # Gray for unknown
    val = int(value)
    if val >= 5:
        return "#5DBB63"  # Green
    elif val >= 1:
        return "#F39C12"  # Orange
    else:
        return "#E74C3C"  # Red


def pct_to_marker_color(row, metric_col="metric"):
    """
    DEPRECATED: Use count_to_marker_color instead.
    Convert availability percentage to RGBA color for map markers.
    """
    cap = row.get("capacity", 0)
    if pd.isna(cap) or cap == 0:
        return [44, 62, 80, 220]  # Dark gray (#2C3E50) for unknown
    
    v = float(row[metric_col])
    if v < 0.3:
        return [231, 76, 60, 220]  # Danger red (#E74C3C)
    if v < 0.6:
        return [243, 156, 18, 220]  # Warning amber (#F39C12)
    return [93, 187, 99, 220]  # Success green (#5DBB63)


def count_to_marker_color(row, mode="bike"):
    """
    Convert availability count to RGBA color for map markers (count-based).
    
    Color scheme (count-based):
    - Green (#5DBB63): 5+ available
    - Orange (#F39C12): 1-4 available
    - Red (#E74C3C): 0 available
    - Gray (#6B7280): Unknown/NA
    
    Args:
        row: DataFrame row with numbikesavailable/numdocksavailable
        mode: "bike" or "dock" to determine which metric to use
        
    Returns:
        list: RGBA color array [R, G, B, A]
    """
    if mode == "bike":
        value = row.get("numbikesavailable", 0)
    else:
        value = row.get("numdocksavailable", 0)
    
    if pd.isna(value):
        return [107, 114, 128, 220]  # Gray (#6B7280) for unknown
    
    val = int(value)
    if val >= 5:
        return [93, 187, 99, 220]  # Green (#5DBB63)
    elif val >= 1:
        return [243, 156, 18, 220]  # Orange (#F39C12)
    else:
        return [231, 76, 60, 220]  # Red (#E74C3C)


def add_color_columns(df):
    """
    Add color columns for tooltips to DataFrame (count-based).
    
    Args:
        df: DataFrame with numbikesavailable and numdocksavailable columns
        
    Returns:
        DataFrame: Original DataFrame with added bikes_color and docks_color columns
    """
    df["bikes_color"] = df["numbikesavailable"].apply(get_availability_color)
    df["docks_color"] = df["numdocksavailable"].apply(get_availability_color)
    return df


def format_minutes(minutes: float) -> str:
    """Format a minute duration into a friendly label."""
    rounded = int(round(minutes))
    if rounded <= 0:
        return "<1 min"
    return f"{rounded} min"


def format_distance(kilometers: float) -> str:
    """Format a distance in kilometers into metres or km text."""
    meters = kilometers * 1000
    if meters < 1000:
        return f"{meters:.0f} m"
    return f"{kilometers:.1f} km"


def build_timeline_html(events: List[Dict]) -> Tuple[str, str]:
    """Render timeline styling and markup for the journey timeline."""
    blocks = []
    event_count = len(events)
    for idx, event in enumerate(events):
        top_hidden = "hidden" if idx == 0 else ""
        bottom_hidden = "hidden" if idx == event_count - 1 else ""
        chips_container = ""
        if event.get("chips"):
            chips = "".join(f'<span class="timeline-chip">{chip}</span>' for chip in event["chips"])
            chips_container = f'<div class="timeline-chips">{chips}</div>'
        subtitle_html = f'<div class="timeline-subtitle">{event["subtitle"]}</div>' if event.get("subtitle") else ""
        body_html = f'<div class="timeline-body">{event["body"]}</div>' if event.get("body") else ""
        prediction_html = ""
        if event.get("prediction_html"):
            prediction_icon = event.get("prediction_icon", "🤖")
            prediction_label = html.escape(event.get("prediction_label", "AI forecast"))
            prediction_content = event["prediction_html"]
            prediction_html = textwrap.dedent(
                f"""
                <div class="timeline-prediction">
                    <div class="timeline-prediction-icon">{prediction_icon}</div>
                    <div class="timeline-prediction-text">
                        <div class="timeline-prediction-label">{prediction_label}</div>
                        <div class="timeline-prediction-value">{prediction_content}</div>
                    </div>
                </div>
                """
            ).strip()
        block_html = textwrap.dedent(
            f"""
            <div class="timeline-step">
                <div class="timeline-time">{event["time"]}</div>
                <div class="timeline-axis">
                    <div class="timeline-line {top_hidden}"></div>
                    <div class="timeline-dot">{event["icon"]}</div>
                    <div class="timeline-line {bottom_hidden}"></div>
                </div>
                <div class="timeline-content">
                    <div class="timeline-title">{event["title"]}</div>
                    {subtitle_html}
                    {body_html}
                    {prediction_html}
                    {chips_container}
                </div>
            </div>
            """
        ).strip()
        blocks.append(block_html)

    timeline_body = "\n".join(blocks)
    style_block = textwrap.dedent(
        """
        <style>
            :root {
                --primary-green: #5DBB63;
                --accent-blue: #3498DB;
                --text-primary: #262730;
                --text-secondary: #6B7280;
                --bg-card: #F0F2F6;
                --border-color: #E5E7EB;
            }
            html, body {
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: FONT_STACK_VALUE;
                color: var(--text-primary);
            }
            .timeline-container {
                background: #ffffff;
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 1.7rem 1.9rem;
                box-shadow: 0 20px 50px rgba(38, 39, 48, 0.06);
                margin-top: 1.3rem;
                font-family: FONT_STACK_VALUE;
            }
            .timeline-step {
                display: grid;
                grid-template-columns: 82px 34px 1fr;
                column-gap: 1.35rem;
                align-items: flex-start;
            }
            .timeline-step + .timeline-step {
                margin-top: 1.35rem;
            }
            .timeline-time {
                text-align: right;
                font-weight: 600;
                color: var(--text-primary);
                font-size: 0.95rem;
                padding-top: 0.25rem;
            }
            .timeline-axis {
                display: flex;
                flex-direction: column;
                align-items: center;
                height: 100%;
            }
            .timeline-line {
                width: 3px;
                background: linear-gradient(180deg, rgba(52, 152, 219, 0.35), rgba(93, 187, 99, 0.45));
                flex-grow: 1;
            }
            .timeline-line.hidden {
                visibility: hidden;
                flex-grow: 0;
            }
            .timeline-dot {
                width: 28px;
                height: 28px;
                border-radius: 50%;
                background: #ffffff;
                border: 3px solid var(--accent-blue);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.9rem;
                color: var(--accent-blue);
                box-shadow: 0 0 0 5px rgba(52, 152, 219, 0.14);
                margin: 0.25rem 0;
            }
            .timeline-content {
                background: var(--bg-card);
                border-radius: 12px;
                padding: 1rem 1.25rem;
                border: 1px solid rgba(52, 152, 219, 0.15);
                box-shadow: 0 12px 28px rgba(37, 99, 235, 0.07);
            }
            .timeline-title {
                font-weight: 600;
                font-size: 1.05rem;
                color: var(--text-primary);
            }
            .timeline-subtitle {
                font-size: 0.78rem;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-top: 0.15rem;
            }
            .timeline-body {
                font-size: 0.95rem;
                color: var(--text-primary);
                margin-top: 0.65rem;
                line-height: 1.55;
            }
            .timeline-chips {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.85rem;
            }
            .timeline-chip {
                background: rgba(52, 152, 219, 0.14);
                color: var(--accent-blue);
                font-weight: 600;
                font-size: 0.78rem;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                letter-spacing: 0.01em;
            }
            .timeline-prediction {
                margin-top: 0.85rem;
                display: inline-flex;
                align-items: center;
                gap: 0.65rem;
                padding: 0.55rem 1.05rem;
                border-radius: 14px;
                background: linear-gradient(135deg, rgba(52, 152, 219, 0.18), rgba(93, 187, 99, 0.26));
                border: 1px solid rgba(52, 152, 219, 0.18);
                box-shadow: 0 10px 22px rgba(52, 152, 219, 0.12);
                color: var(--text-primary);
            }
            .timeline-prediction-icon {
                width: 34px;
                height: 34px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                background: white;
                color: var(--accent-blue);
                box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.18);
            }
            .timeline-prediction-icon svg {
                display: block;
                width: 24px;
                height: 24px;
            }
            .timeline-prediction-text {
                display: flex;
                flex-direction: column;
                gap: 0.15rem;
            }
            .timeline-prediction-label {
                text-transform: uppercase;
                font-size: 0.68rem;
                letter-spacing: 0.15em;
                color: rgba(38, 39, 48, 0.6);
                font-weight: 600;
            }
            .timeline-prediction-value {
                font-size: 0.98rem;
                font-weight: 600;
                color: var(--accent-blue);
            }
            .timeline-prediction-number {
                font-size: 1.25rem;
                font-weight: 700;
                margin-right: 0.35rem;
                display: inline-block;
            }
            @media (max-width: 768px) {
                .timeline-container {
                    padding: 1.2rem 1.1rem;
                }
                .timeline-step {
                    grid-template-columns: 64px 30px 1fr;
                    column-gap: 1rem;
                }
                .timeline-content {
                    padding: 0.85rem 1rem;
                }
            }
        </style>
        """
    ).replace("FONT_STACK_VALUE", TIMELINE_FONT_STACK).strip()
    markup = textwrap.dedent(
        f"""
        <div class="timeline-container">
            {timeline_body}
        </div>
        """
    ).strip()
    return style_block, markup


# Configure page
st.set_page_config(
    page_title="Velib Trend — Paris Bike Predictions",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DESIGN SYSTEM - Professional Light Theme
# ============================================================
# Primary: #5DBB63 (Vélib green - brand color)
# Accent: #3498DB (Bright blue - interactive elements)
# Warning: #F39C12 (Amber - medium availability)
# Danger: #E74C3C (Red - low availability)
# Background: #FFFFFF (white), #F0F2F6 (light gray cards)
# Text: #262730 (dark gray), #6B7280 (secondary gray)
# ============================================================

# Custom CSS for professional light theme
st.markdown("""
<style>
    /* ========== LIGHT THEME COLORS ========== */
    :root {
        /* Brand colors */
        --primary-green: #5DBB63;
        --primary-green-dark: #4A9D50;
        --accent-blue: #3498DB;
        --accent-blue-dark: #2980B9;
        --warning-amber: #F39C12;
        --danger-red: #E74C3C;
        
        /* Backgrounds - Light theme */
        --bg-main: #FFFFFF;
        --bg-card: #F0F2F6;
        --bg-hover: #E5E7EB;
        
        /* Text - Light theme */
        --text-primary: #262730;
        --text-secondary: #6B7280;
        
        /* Borders - Light theme */
        --border-color: #E5E7EB;
    }
    
    /* ========== CENTER MAIN CONTENT ========== */
    .block-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }
    
    /* ========== MAIN TITLE ========== */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--primary-green) !important;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary) !important;
        font-weight: 400;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* ========== SECTION HEADERS ========== */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary) !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-green);
        padding-left: 1rem;
        text-align: left;
    }
    
    /* ========== CARDS & CONTAINERS ========== */
    .info-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-blue);
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .info-card * {
        color: var(--text-primary) !important;
    }
    
    .info-card strong {
        color: var(--primary-green) !important;
        font-weight: 600;
    }
    
    .info-card ul, .info-card li {
        color: var(--text-secondary) !important;
    }
    
    .success-banner {
        background: linear-gradient(135deg, #5DBB63 0%, #4A9D50 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-banner {
        background: linear-gradient(135deg, #F39C12 0%, #E67E22 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .danger-banner {
        background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* ========== METRICS ========== */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.9rem;
    }
    
    /* ========== BUTTONS ========== */
    .stButton>button {
        background-color: var(--accent-blue);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--accent-blue-dark);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }
    
    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: var(--primary-green) !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] p {
        color: var(--text-secondary) !important;
    }
    
    /* ========== DIVIDERS ========== */
    hr {
        border: none;
        border-top: 1px solid var(--border-color);
        margin: 1.5rem 0;
    }
    
    /* ========== FORM INPUTS ========== */
    label[data-testid="stWidgetLabel"] {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    input[type="text"] {
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
    }
    
    input[type="text"]:focus {
        border-color: var(--primary-green) !important;
        box-shadow: 0 0 0 2px rgba(93, 187, 99, 0.2) !important;
        outline: none !important;
    }
    
    textarea:focus {
        border-color: var(--primary-green) !important;
        box-shadow: 0 0 0 2px rgba(93, 187, 99, 0.2) !important;
        outline: none !important;
    }
    
    /* ========== MARKDOWN HEADERS ========== */
    h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /* ========== CENTER JOURNEY PLANNER ELEMENTS ========== */
    .centered-header {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header - clean and professional
st.markdown('<h1 class="main-title">Velib Trend</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time availability & AI-powered predictions for Paris bike-sharing</p>', unsafe_allow_html=True)
st.markdown("---")

validate = True  # Always request backend validation now that advanced options are gone

@st.cache_data(ttl=60)
def load_data(validate: bool) -> pd.DataFrame:
    """
    Load station data from FastAPI backend with caching.
    
    Args:
        validate: Whether to validate data types on the backend
        
    Returns:
        DataFrame with station information, filtered for installed stations only
        
    Raises:
        requests.HTTPError: If API request fails
    """
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
    
    # Reorder columns for better readability
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
    
    ordered = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]
    return df[ordered]

# Use session state to persist data across reruns (avoids reloading on every interaction)
if 'cached_df' not in st.session_state:
    st.session_state['cached_df'] = load_data(validate=validate)

df = st.session_state['cached_df']

# ============================================================
# UNIFIED SINGLE-PAGE LAYOUT - No Navigation
# ============================================================
try:
    # ============================================================
    # SECTION 1: JOURNEY PLANNER (Primary Feature)
    # ============================================================
    from journey_planner import (
        geocode_address,
        plan_route,
        get_prediction_at_time,
        get_journey_verdict
    )
    
    # Section header
    st.markdown("""
    <div class="info-card" style="text-align: center; border-left: none; max-width: 700px; margin: 2rem auto;">
    <strong style="font-size: 1.3rem;">🚴 Plan Your Journey</strong>
    <p style="margin: 1rem 0 0.5rem 0; color: var(--text-secondary); line-height: 1.6;">
    Plan your trip with confidence. Enter your start and end points to confirm that a bike is waiting for you at your departure station and an empty dock will be available when you arrive. Enjoy your ride!
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input form with better design
    col1, col2 = st.columns(2)
    with col1:
        start_address = st.text_input(
            "From (start location)",
            placeholder="e.g., 24 Rue de Rivoli, Paris",
            help="Enter your starting address - be as specific as possible"
        )
    with col2:
        dest_address = st.text_input(
            "To (destination)",
            placeholder="e.g., Gare du Nord, Paris",
            help="Enter your destination address - be as specific as possible"
        )
    
    st.markdown("")  # Spacing
    
    # Plan route button - centered
    col_left, col_btn, col_right = st.columns([1, 2, 1])
    with col_btn:
        plan_button = st.button("Plan My Route", type="primary", use_container_width=True)
    
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
                progress_text.info("🔎 Finding your locations… (~1s)")
                
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
                    progress_text.success("✅ Locations found")
                    
                    # Step 2: Plan route
                    progress_text.info("Calculating best stations… (~1s)")
                    route = plan_route(start_lat, start_lon, dest_lat, dest_lon, df)
                    
                    start_station = route['start_station']
                    end_station = route['end_station']
                    progress_text.success("✅ Route ready")
                    
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
                        progress_text.info("🔮 Fetching availability predictions… (~15-30s)")
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
                            
                            progress_text.success("✅ Predictions received")
                            
                            # Step 4: Get verdict
                            progress_text.info("🎯 Reviewing journey confidence… (~1s)")
                            verdict = get_journey_verdict(
                                start_pred['bikes_predicted'],
                                end_pred['docks_predicted']
                            )
                            
                            progress_text.success("✅ Journey analysis complete")
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
                                st.metric("🚶 Walk to Start", f"{route['walk_to_start_min']:.0f} min")
                            with col2:
                                st.metric("🚴 Bike Ride", f"{route['bike_time_min']:.0f} min")
                            with col3:
                                st.metric("🚶 Walk to Dest", f"{route['walk_from_end_min']:.0f} min")
                            with col4:
                                st.metric("⏱️ Total Time", f"{route['total_time_min']:.0f} min")
                            
                            # Station details (without predictions)
                            st.markdown("### 🚲 Start Station")
                            st.markdown(f"**{start_station['name']}**")
                            st.caption(f"📍 {route['walk_to_start_km']*1000:.0f}m from your location ({route['walk_to_start_min']:.0f} min walk)")
                            
                            st.markdown("### 🅿️ End Station")
                            st.markdown(f"**{end_station['name']}**")
                            st.caption(f"📍 {route['walk_from_end_km']*1000:.0f}m from destination ({route['walk_from_end_min']:.0f} min walk)")
                            
                            st.info("💡 **Tip**: Check current availability manually in the Explore Map section below before starting your journey")
                            
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
                            st.markdown('<div class="section-header">Your Journey Plan</div>', unsafe_allow_html=True)
                            
                            # Verdict banner with new design system colors
                            if verdict['status'] == 'success':
                                banner_class = "success-banner"
                            elif verdict['status'] == 'warning':
                                banner_class = "warning-banner"
                            else:
                                banner_class = "danger-banner"
                            
                            st.markdown(f"""
                            <div class="{banner_class}">
                                <h3 style="margin: 0; font-size: 1.6rem;">{verdict['icon']} {verdict['verdict']}</h3>
                                <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.95;">{verdict['details']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Time breakdown as Google Maps inspired timeline
                            st.markdown("#### ⏱️ Journey Timeline")

                            paris_now = datetime.now(ZoneInfo('Europe/Paris'))
                            arrive_start_time = paris_now + timedelta(minutes=route['walk_to_start_min'])
                            arrive_end_time = arrive_start_time + timedelta(minutes=route['bike_time_min'])
                            arrive_destination_time = arrive_end_time + timedelta(minutes=route['walk_from_end_min'])

                            start_station_name = html.escape(start_station['name'])
                            end_station_name = html.escape(end_station['name'])
                            safe_start_address = html.escape(start_address.strip()) if start_address else "Your location"
                            safe_dest_address = html.escape(dest_address.strip()) if dest_address else "Destination"

                            predicted_bikes = max(0, int(round(start_pred.get('bikes_predicted', 0))))
                            predicted_docks = max(0, int(round(end_pred.get('docks_predicted', 0))))

                            timeline_events = [
                                {
                                    "time": paris_now.strftime("%H:%M"),
                                    "icon": "🚶",
                                    "title": "Leave now",
                                    "subtitle": safe_start_address,
                                    "body": f"Walk towards <strong>{start_station_name}</strong> to pick up your bike.",
                                    "chips": [
                                        f"{format_minutes(route['walk_to_start_min'])}",
                                        f"{format_distance(route['walk_to_start_km'])}"
                                    ],
                                },
                                {
                                    "time": arrive_start_time.strftime("%H:%M"),
                                    "icon": "🚲",
                                    "title": start_station_name,
                                    "subtitle": "Pickup station",
                                    "body": f"Model expects roughly <strong>{predicted_bikes}</strong> bikes available when you arrive.",
                                    "prediction_label": "AI forecast",
                                    "prediction_icon": AI_PREDICTION_ICON,
                                    "prediction_html": f"<span class=\"timeline-prediction-number\">{predicted_bikes}</span> bikes ready for pickup",
                                    "chips": [
                                        f"Ride {format_minutes(route['bike_time_min'])}",
                                        f"{format_distance(route['bike_distance_km'])}"
                                    ],
                                },
                                {
                                    "time": arrive_end_time.strftime("%H:%M"),
                                    "icon": "🅿️",
                                    "title": end_station_name,
                                    "subtitle": "Return station",
                                    "body": f"Dock the bike here. Forecast shows <strong>{predicted_docks}</strong> free docks.",
                                    "prediction_label": "AI forecast",
                                    "prediction_icon": AI_PREDICTION_ICON,
                                    "prediction_html": f"<span class=\"timeline-prediction-number\">{predicted_docks}</span> docks open on arrival",
                                    "chips": [
                                        f"Walk {format_minutes(route['walk_from_end_min'])}",
                                        f"{format_distance(route['walk_from_end_km'])}"
                                    ],
                                },
                                {
                                    "time": arrive_destination_time.strftime("%H:%M"),
                                    "icon": "🏁",
                                    "title": safe_dest_address,
                                    "subtitle": "Destination",
                                    "body": f"Total journey about <strong>{format_minutes(route['total_time_min'])}</strong>. Enjoy the ride!",
                                    "chips": [],
                                },
                            ]

                            timeline_style, timeline_markup = build_timeline_html(timeline_events)
                            timeline_html = textwrap.dedent(
                                f"""
                                <!DOCTYPE html>
                                <html>
                                <head>
                                    <meta charset=\"utf-8\">
                                    {timeline_style}
                                </head>
                                <body>
                                    {timeline_markup}
                                </body>
                                </html>
                                """
                            ).strip()
                            timeline_height = max(520, 220 + len(timeline_events) * 160)
                            components.html(timeline_html, height=timeline_height, scrolling=False)
                            
                            # Google Maps integration button
                            st.markdown("---")
                            
                            # Create Google Maps URL with waypoints (API format)
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
                            <div style="text-align: center;">
                                <a href="{google_maps_url}" target="_blank" style="text-decoration: none; display: inline-block;">
                                    <div style="display: inline-flex; align-items: center; gap: 0.85rem; background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
                                                color: white; padding: 0.85rem 1.8rem; border-radius: 10px;
                                                font-weight: 600; font-size: 1.05rem; box-shadow: 0 6px 12px rgba(0,0,0,0.12);
                                                transition: transform 0.2s ease, box-shadow 0.2s ease;">
                                        <img src="https://upload.wikimedia.org/wikipedia/commons/a/aa/Google_Maps_icon_%282020%29.svg" alt="Google Maps" width="28" height="28" style="background: white; border-radius: 50%; padding: 2px; box-shadow: 0 0 0 1px rgba(0,0,0,0.08);">
                                        <span>Open Route in Google Maps</span>
                                    </div>
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                                        
            except Exception as e:
                import traceback
                st.error(f"❌ Error planning journey: {str(e)}")
                st.code(traceback.format_exc())
                st.info("💡 Make sure FastAPI is running and try again")
    
    # ============================================================
    # UNIFIED MAP: One map for everything (route + search)
    # ============================================================
    st.markdown("---")
    if "mode" not in st.session_state:
        st.session_state["mode"] = "bike"
    mode = st.session_state["mode"]
    st.markdown('<h3 style="text-align: center; color: var(--text-primary); font-weight: 600; margin: 2rem 0 1rem 0;">Paris Vélib\' Network</h3>', unsafe_allow_html=True)
    
    # Data preparation for map
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

    # Select metric based on mode (bike or dock)
    metric_col = "pct_bikes" if mode == "bike" else "pct_docks"
    df["metric"] = df[metric_col]

    # Apply count-based color mapping for map markers
    df["color"] = df.apply(lambda row: count_to_marker_color(row, mode), axis=1)
    df = add_color_columns(df)

    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    map_df = df.dropna(subset=["lat", "lon"]).copy()
    
    # ============================================================
    # STATION SEARCH BAR (above the map)
    # ============================================================
    
    # Create searchable options
    station_options = {}
    station_lookup = {}
    for _, row in map_df.iterrows():
        station_name = row.get('name', 'Unknown')
        station_code = str(row.get('stationcode', ''))
        display_text = f"{station_name} ({station_code})"
        station_options[display_text] = station_code
        station_lookup[display_text] = row
    
    sorted_options = [""] + sorted(station_options.keys())
    
    # Centered search bar with professional icon overlay
    col_left, col_search, col_right = st.columns([1, 3, 1])
    with col_search:
        st.markdown("""
        <style>
        /* Ensure container allows absolute positioned icon */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {
            position: relative;
            overflow: visible;
        }
        /* Inject SVG magnifier as background icon */
        div[data-testid="stSelectbox"] div[data-baseweb="select"]::before {
            content: "";
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            width: 18px;
            height: 18px;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ccircle cx='11' cy='11' r='7' stroke='%23262730' stroke-width='2' fill='none'/%3E%3Cline x1='16.5' y1='16.5' x2='21' y2='21' stroke='%23262730' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-size: 18px 18px;
            opacity: 0.7;
            pointer-events: none;
            z-index: 2;
        }
        /* Add padding so text doesn't overlap icon */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            padding-left: 2.8rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        selected_station = st.selectbox(
            "Search by station name",
            options=sorted_options,
            help="Start typing a station name (e.g., 'République', 'Bastille', 'Louvre')",
            placeholder="Type to search...",
            label_visibility="collapsed",
            key="station_search_select"
        )
    
    station_code_input = station_options.get(selected_station, "") if selected_station else ""
    
    # Prepare ALL stations as base layer
    all_stations_map = map_df[['lat', 'lon', 'name', 'stationcode', 'numbikesavailable', 'numdocksavailable', 'color']].copy()
    all_stations_map = add_color_columns(all_stations_map)
    
    # Check states: route planned? station searched?
    route_info = st.session_state.get('route_data', None)
    
    # Determine map state: route, search, or default
    if route_info is not None:
        # STATE 1: Route planned - show route overlay
        st.markdown('<p style="text-align: center; color: var(--text-secondary); margin-bottom: 1.5rem;">All Vélib\' stations shown in gray. Your route is highlighted in color.</p>', unsafe_allow_html=True)
        
        # Background stations (gray, small, semi-transparent)
        all_stations_map['type'] = 'background'
        all_stations_map['color'] = [[180, 180, 180, 80]] * len(all_stations_map)
        all_stations_map['radius'] = 60
        
        # Extract route info
        start_lat = route_info['start_lat']
        start_lon = route_info['start_lon']
        dest_lat = route_info['dest_lat']
        dest_lon = route_info['dest_lon']
        start_station = route_info['start_station']
        end_station = route_info['end_station']
        
        # Create route points
        route_points = pd.DataFrame({
            'lat': [start_lat, start_station['lat'], end_station['lat'], dest_lat],
            'lon': [start_lon, start_station['lon'], end_station['lon'], dest_lon],
            'type': ['start', 'start_station', 'end_station', 'destination'],
            'name': ['Your location', start_station['name'], end_station['name'], 'Destination'],
            'stationcode': ['', start_station.get('stationcode', ''), end_station.get('stationcode', ''), ''],
            'numbikesavailable': [0, start_station.get('numbikesavailable', 0), end_station.get('numbikesavailable', 0), 0],
            'numdocksavailable': [0, start_station.get('numdocksavailable', 0), end_station.get('numdocksavailable', 0), 0]
        })
        route_points = add_color_columns(route_points)
        
        color_map = {
            'start': [255, 0, 0, 235],
            'start_station': [0, 200, 0, 255],
            'end_station': [0, 100, 255, 255],
            'destination': [255, 0, 0, 235]
        }
        route_points['color'] = route_points['type'].map(color_map)
        route_points['radius'] = 120
        
        # Combine: background + route
        map_data = pd.concat([all_stations_map, route_points], ignore_index=True)
        
        # Zoom to route
        view_state = pdk.ViewState(
            latitude=(start_lat + dest_lat) / 2,
            longitude=(start_lon + dest_lon) / 2,
            zoom=13,
            pitch=0
        )
        
    elif station_code_input and selected_station in station_lookup:
        # STATE 2: Station searched (no route) - zoom to selected station
        selected_row = station_lookup[selected_station]
        center_lat = float(selected_row["lat"])
        center_lon = float(selected_row["lon"])
        
        # Color all stations by availability
        all_stations_map['radius'] = 95
        all_stations_map['type'] = 'station'
        map_data = all_stations_map
        
        # Zoom to selected station
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=15,
            pitch=0
        )
        
    else:
        # STATE 3: Default view - show all Paris
        
        # Color all stations by availability
        all_stations_map['radius'] = 95
        all_stations_map['type'] = 'station'
        map_data = all_stations_map
        
        # Default view: Paris center (zoomed out to show more area)
        view_state = pdk.ViewState(
            latitude=48.8566,
            longitude=2.3522,
            zoom=11,  # Reduced from 12 to show wider area
            pitch=0
        )
    
    # Tooltip (same for all states)
    tooltip = {
        "html": """
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                <div style="font-size: 15px; font-weight: 600; margin-bottom: 8px; color: #5DBB63;">{name}</div>
                <div style="font-size: 13px; line-height: 1.6;">
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span style="color: #9CA3AF;">🚲 Bikes:</span>
                        <span style="font-weight: 600; color: {bikes_color};">{numbikesavailable}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 4px 0;">
                        <span style="color: #9CA3AF;">🅿️ Docks:</span>
                        <span style="font-weight: 600; color: {docks_color};">{numdocksavailable}</span>
                    </div>
                </div>
            </div>
        """,
        "style": {
            "backgroundColor": "rgba(255, 255, 255, 0.98)",
            "color": "#262730",
            "fontSize": "14px",
            "borderRadius": "8px",
            "padding": "12px 16px",
            "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.15)",
            "border": "1px solid #E5E7EB"
        },
    }
        
    
    # Create and display the map
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=map_data,
        get_position='[lon, lat]',
        get_color='color',
        get_radius='radius',  # Radius in meters (from dataframe), auto-scales with zoom
        pickable=True,
        auto_highlight=True,
        radius_min_pixels=4,  # Same as Explore Map - minimum visibility when zoomed out
        radius_max_pixels=20,  # Same as Explore Map - avoid huge circles when zoomed in
        get_line_color=[255, 255, 255, 255],  # Pure white outer ring
        line_width_min_pixels=1.8,  # Same as Explore Map - thin border
        stroked=True,
        filled=True,
        opacity=0.8  # Same as Explore Map - solid, professional look
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
        mode = st.radio(
            "Choose what availability to highlight on the map",
            ["bike", "dock"],
            format_func=lambda x: "🚲 Bikes available" if x == "bike" else "🅿️ Docks available",
            key="mode"
        )
        resource_label = "bikes" if mode == "bike" else "docks free"
        st.markdown(f"**Station availability:** 🟢 Good (5+ {resource_label}) · 🟠 Low (1-4 {resource_label}) · 🔴 Empty")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Your Route:**")
            st.markdown("🔴 You/Destination · 🟢 Start Station · 🔵 End Station")
        with col2:
            st.markdown("**Network:**")
            st.markdown(f"⚪ All {len(df):,} Vélib stations (gray)")
    
    # ============================================================
    # STATION DETAILS & PREDICTIONS (when a station is selected)
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
                                line=dict(color='#2C3E50', width=2.5, shape='spline'),
                                marker=dict(size=5, color='#2C3E50'),
                                hovertemplate='<b>%{x|%H:%M}</b><br>Bikes: %{y}<extra></extra>'
                            ))
                        elif hist_times:
                            st.warning("⚠️ Invalid historical time data types detected")
                        
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
                                        line=dict(color='#BDC3C7', width=2, dash='dot', shape='spline'),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ))
                                
                                # Add predictions trace with smooth curve
                                fig.add_trace(go.Scatter(
                                    x=pred_times,
                                    y=pred_bikes,
                                    mode='lines+markers',
                                    name='AI Predictions',
                                    line=dict(color='#3498DB', width=3, shape='spline'),
                                    marker=dict(size=10, symbol='diamond', color='#3498DB'),
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

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("💡 Make sure FastAPI is running on http://127.0.0.1:8000")
