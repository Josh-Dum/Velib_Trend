"""
Journey Planner Module

This module provides functionality to plan bike journeys between two addresses,
automatically finding the best Vélib stations and predicting availability.
"""

import requests
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from typing import Tuple, Dict, Optional
import time
from functools import lru_cache


# Constants
WALKING_SPEED_KMH = 5.0  # Average walking speed
BIKING_SPEED_KMH = 15.0  # Average biking speed in city
BIKE_THRESHOLD_SAFE = 4  # >= 4 bikes = safe
BIKE_THRESHOLD_MIN = 2   # >= 2 bikes = possible
DOCK_THRESHOLD_SAFE = 4  # >= 4 docks = safe
DOCK_THRESHOLD_MIN = 2   # >= 2 docks = possible

# Geocoding cache (LRU cache for 128 most recent addresses)
@lru_cache(maxsize=128)


def geocode_address(address: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert address to (latitude, longitude) using Nominatim.
    Cached with LRU cache to avoid redundant API calls.
    
    Args:
        address: Address string (e.g., "24 Rue de Rivoli, Paris")
    
    Returns:
        (lat, lon) tuple or (None, None) if not found
    
    Raises:
        requests.RequestException: If network error occurs
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
        "countrycodes": "fr",  # Limit to France for faster results
        "addressdetails": 1
    }
    headers = {
        "User-Agent": "VelibTrend/1.0 (Educational Project)"
    }
    
    try:
        # Add small delay to respect Nominatim rate limiting (1 req/sec)
        time.sleep(1)
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            return float(data[0]['lat']), float(data[0]['lon'])
        
        return None, None
    
    except requests.RequestException as e:
        print(f"Geocoding error for '{address}': {e}")
        return None, None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS coordinates using Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in kilometers
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance


def find_nearest_station(lat: float, lon: float, stations_df: pd.DataFrame) -> Dict:
    """
    Find the closest Vélib station to given coordinates.
    
    Args:
        lat, lon: Target coordinates
        stations_df: DataFrame with all stations (must have 'lat', 'lon' columns)
    
    Returns:
        Dictionary with station info + distance_km
    """
    # Calculate distance to all stations
    stations_df = stations_df.copy()
    stations_df['distance_km'] = stations_df.apply(
        lambda row: haversine_distance(lat, lon, row['lat'], row['lon']),
        axis=1
    )
    
    # Find nearest
    nearest_idx = stations_df['distance_km'].idxmin()
    nearest = stations_df.loc[nearest_idx].to_dict()
    
    return nearest


def plan_route(start_lat: float, start_lon: float, 
               dest_lat: float, dest_lon: float,
               stations_df: pd.DataFrame) -> Dict:
    """
    Plan complete journey from start to destination.
    
    Args:
        start_lat, start_lon: Starting coordinates
        dest_lat, dest_lon: Destination coordinates
        stations_df: DataFrame with all stations
    
    Returns:
        Dictionary with route information including:
        - start_station: Station info dict
        - end_station: Station info dict
        - walk_to_start_km/min: Distance and time to start station
        - bike_distance_km/time_min: Biking segment
        - walk_from_end_km/min: Distance and time from end station
        - total_time_min: Total journey time
        - arrival_at_start_min: When user arrives at start station (from now)
        - arrival_at_end_min: When user arrives at end station (from now)
    """
    # Find nearest stations
    start_station = find_nearest_station(start_lat, start_lon, stations_df)
    end_station = find_nearest_station(dest_lat, dest_lon, stations_df)
    
    # Calculate distances
    walk_to_start_km = start_station['distance_km']
    bike_distance_km = haversine_distance(
        start_station['lat'], start_station['lon'],
        end_station['lat'], end_station['lon']
    )
    walk_from_end_km = haversine_distance(
        end_station['lat'], end_station['lon'],
        dest_lat, dest_lon
    )
    
    # Calculate times
    walk_to_start_min = (walk_to_start_km / WALKING_SPEED_KMH) * 60
    bike_time_min = (bike_distance_km / BIKING_SPEED_KMH) * 60
    walk_from_end_min = (walk_from_end_km / WALKING_SPEED_KMH) * 60
    total_time_min = walk_to_start_min + bike_time_min + walk_from_end_min
    
    # Arrival times (from now)
    arrival_at_start_min = walk_to_start_min
    arrival_at_end_min = walk_to_start_min + bike_time_min
    
    return {
        'start_station': start_station,
        'end_station': end_station,
        'walk_to_start_km': walk_to_start_km,
        'walk_to_start_min': walk_to_start_min,
        'bike_distance_km': bike_distance_km,
        'bike_time_min': bike_time_min,
        'walk_from_end_km': walk_from_end_km,
        'walk_from_end_min': walk_from_end_min,
        'total_time_min': total_time_min,
        'arrival_at_start_min': arrival_at_start_min,
        'arrival_at_end_min': arrival_at_end_min
    }


def get_prediction_at_time(station_code: str, minutes_from_now: float, api_base_url: str) -> Dict:
    """
    Get prediction for a station at a specific time offset.
    Uses interpolation if time is between prediction points.
    
    Args:
        station_code: Station code
        minutes_from_now: Time offset in minutes from current time
        api_base_url: FastAPI base URL
    
    Returns:
        Dictionary with:
        - bikes_predicted: Predicted number of bikes
        - docks_predicted: Predicted number of docks
        - confidence: 'high', 'medium', or 'low'
    
    Raises:
        requests.RequestException: If API call fails
    """
    try:
        # Get predictions from API
        response = requests.get(f"{api_base_url}/predict/{station_code}", timeout=40)
        response.raise_for_status()
        data = response.json()
        
        # Extract current data from API response
        # API format: {"current": {"bikes_available": int, "docks_available": int, "capacity": int}}
        current = data['current']
        current_bikes = current['bikes_available']
        current_docks = current['docks_available']
        capacity = current['capacity']
        
        # Extract predictions from API response
        # API format: {"predictions": {"T+1h": {"bikes": int}, "T+2h": {"bikes": int}, "T+3h": {"bikes": int}}}
        predictions = data['predictions']
        
        # Build prediction arrays for interpolation
        # Time points: 0min (now), 60min (T+1h), 120min (T+2h), 180min (T+3h)
        pred_times = [0, 60, 120, 180]
        
        pred_bikes = [
            current_bikes,
            predictions['T+1h']['bikes'],
            predictions['T+2h']['bikes'],
            predictions['T+3h']['bikes']
        ]
        
        # The model only predicts bikes, so compute docks = capacity - bikes
        pred_docks = [capacity - bikes for bikes in pred_bikes]
        
        # Interpolate to get prediction at specific time
        bikes_predicted = np.interp(minutes_from_now, pred_times, pred_bikes)
        docks_predicted = np.interp(minutes_from_now, pred_times, pred_docks)
        
        # Determine confidence based on time horizon
        if minutes_from_now <= 30:
            confidence = 'high'
        elif minutes_from_now <= 90:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'bikes_predicted': round(bikes_predicted, 1),
            'docks_predicted': round(docks_predicted, 1),
            'confidence': confidence
        }
        
    except requests.RequestException as e:
        print(f"Prediction API error for station {station_code}: {e}")
        raise
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error parsing prediction data for station {station_code}: {e}")
        print(f"API response: {data}")
        raise


def get_journey_verdict(bikes_predicted: float, docks_predicted: float) -> Dict:
    """
    Analyze predictions and provide recommendation.
    
    Args:
        bikes_predicted: Predicted number of bikes at start station
        docks_predicted: Predicted number of docks at end station
    
    Returns:
        Dictionary with:
        - status: 'success', 'warning', or 'error'
        - verdict: User-friendly message
        - icon: Emoji icon
        - details: Additional explanation
    """
    if bikes_predicted >= BIKE_THRESHOLD_SAFE and docks_predicted >= DOCK_THRESHOLD_SAFE:
        return {
            'status': 'success',
            'verdict': 'Perfect! Good to go 🎉',
            'icon': '✅',
            'details': f'{bikes_predicted:.0f} bikes and {docks_predicted:.0f} docks should be available.'
        }
    
    elif bikes_predicted >= BIKE_THRESHOLD_MIN and docks_predicted >= DOCK_THRESHOLD_MIN:
        issues = []
        if bikes_predicted < BIKE_THRESHOLD_SAFE:
            issues.append(f'only {bikes_predicted:.0f} bike(s) at start')
        if docks_predicted < DOCK_THRESHOLD_SAFE:
            issues.append(f'only {docks_predicted:.0f} dock(s) at end')
        
        return {
            'status': 'warning',
            'verdict': 'Possible, but tight ⚠️',
            'icon': '⚠️',
            'details': f'Note: {" and ".join(issues)}. Consider having a backup plan.'
        }
    
    else:
        problems = []
        if bikes_predicted < BIKE_THRESHOLD_MIN:
            problems.append('very few bikes at start station')
        if docks_predicted < DOCK_THRESHOLD_MIN:
            problems.append('very few docks at end station')
        
        return {
            'status': 'error',
            'verdict': 'Risky route ❌',
            'icon': '❌',
            'details': f'Problem: {" and ".join(problems)}. Try alternative stations or timing.'
        }
