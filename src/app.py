from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
import json
import os
import time

from fastapi import FastAPI, HTTPException, Query
import boto3
from botocore.exceptions import ClientError

from src.fetch_live_velib import fetch_live, fetch_live_all, normalize
from src.data_utils import validate_and_sanitize
from src.fetch_historical import get_24h_history_with_fallback


app = FastAPI(title="Velib Trend API", version="0.1.0")

# SageMaker configuration
SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", "velib-lstm-v3-endpoint")
AWS_REGION = os.environ.get("AWS_REGION", "eu-west-3")

# ============== CACHING FOR PERFORMANCE ==============
# Cache S3 historical data (reduces from 4.5s to 0s on cache hit)
_s3_history_cache: Dict[str, Tuple[Tuple[List[int], bool], float]] = {}

# Cache live stations data (reduces from 7.6s to 0s on cache hit)
_live_stations_cache: Tuple[Optional[List[Dict]], float] = (None, 0)

CACHE_TTL_SECONDS = 1800  # 30 minutes (historical data changes slowly)
LIVE_CACHE_TTL_SECONDS = 300  # 5 minutes (balance between freshness and performance)


@app.on_event("startup")
async def startup_event():
    """
    Pre-warm cache on startup to improve first user experience.
    
    This eliminates the 17s wait for the first Journey Planner request
    by fetching live stations data when the server starts.
    """
    import asyncio
    import threading
    
    def prewarm_cache():
        print("🔥 Pre-warming cache...")
        try:
            t0 = time.time()
            get_cached_live_stations()
            t1 = time.time()
            print(f"✅ Cache pre-warmed in {t1-t0:.1f}s (live stations loaded)")
        except Exception as e:
            print(f"⚠️ Cache pre-warm failed: {e}")
    
    # Run in background thread to not block startup
    thread = threading.Thread(target=prewarm_cache)
    thread.start()


def get_cached_live_stations() -> List[Dict]:
    """
    Get live stations data with caching.
    
    Cache TTL is 5 minutes because:
    - Live data changes frequently, but 5min is fresh enough for predictions
    - Fetching ALL 1,498 stations takes 17-20 seconds
    - 5-minute cache eliminates wait for multiple journey plans in a session
    - Pre-warming at startup ensures first user gets cached data
    
    Returns:
        List of station dictionaries (normalized)
    """
    global _live_stations_cache
    now = time.time()
    
    cached_data, timestamp = _live_stations_cache
    age_seconds = now - timestamp
    
    # Check if cache is valid
    if cached_data is not None and age_seconds < LIVE_CACHE_TTL_SECONDS:
        # Cache hit!
        return cached_data
    
    # Cache miss or expired - fetch fresh data
    raw = fetch_live_all()
    data = normalize(raw)
    
    # Store in cache
    _live_stations_cache = (data, now)
    
    return data


def get_cached_24h_history(station_code: str, current_bikes: int, capacity: int) -> Tuple[List[int], bool]:
    """
    Get 24-hour historical data with caching.
    
    Cache TTL is 5 minutes because:
    - Hourly snapshots don't change frequently
    - Predictions are T+1h, T+2h, T+3h - still valid after 5 min
    - Reduces S3 calls from 24 files per request to once per 5 min
    
    Args:
        station_code: Station code
        current_bikes: Current bikes available (for simulation fallback)
        capacity: Station capacity (for simulation fallback)
    
    Returns:
        Tuple of (historical_bikes list, is_simulated bool)
    """
    cache_key = station_code
    now = time.time()
    
    # Check cache
    if cache_key in _s3_history_cache:
        cached_data, timestamp = _s3_history_cache[cache_key]
        age_seconds = now - timestamp
        
        if age_seconds < CACHE_TTL_SECONDS:
            # Cache hit! Return cached data
            return cached_data
    
    # Cache miss or expired - fetch fresh data from S3
    historical_bikes, is_simulated = get_24h_history_with_fallback(
        station_code=station_code,
        current_bikes=current_bikes,
        capacity=capacity
    )
    
    # Store in cache
    _s3_history_cache[cache_key] = ((historical_bikes, is_simulated), now)
    
    return historical_bikes, is_simulated


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/stations")
def get_stations(
    limit: int = Query(10, ge=1, le=1000, description="Number of records to fetch when all=false"),
    all: bool = Query(False, description="If true, return all stations using pagination"),
    validate: bool = Query(False, description="Coerce numeric/float fields and report issues"),
) -> Dict[str, Any]:
    try:
        # Use cached data instead of fetching fresh every time!
        data: List[Dict[str, Any]] = get_cached_live_stations()
        
        # If not fetching all, limit the results
        if not all:
            data = data[:limit]
        
        issues: List[str] = []
        if validate:
            data, issues = validate_and_sanitize(data)
        return {
            "count": len(data),
            "all": all,
            "validate": validate,
            "issues": issues[:50],  # truncate to keep response small
            "records": data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/{station_code}")
def predict_station(station_code: str) -> Dict[str, Any]:
    """
    Get predictions for a specific station.
    
    This endpoint:
    1. Fetches live data for the station
    2. Fetches 24h historical data from S3 snapshots (with fallback to simulation)
    3. Calls SageMaker endpoint for predictions
    4. Returns predictions + metadata
    """
    try:
        # Step 1: Fetch live data for all stations (WITH CACHING!)
        # Cache reduces latency from 17s → ~0s on cache hit (5-minute TTL)
        data = get_cached_live_stations()
        
        # Find the requested station
        station_data = None
        for station in data:
            if str(station.get("stationcode")) == str(station_code):
                station_data = station
                break
        
        if not station_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Station {station_code} not found"
            )
        
        # Step 2: Extract current station info
        current_bikes = int(station_data.get("numbikesavailable", 0))
        capacity = int(station_data.get("capacity", 35))
        lat = float(station_data.get("lat", 48.8566))
        lon = float(station_data.get("lon", 2.3522))
        name = station_data.get("name", f"Station {station_code}")
        
        # Step 3: Fetch 24h historical data from S3 (WITH CACHING!)
        # This is the key optimization: cache reduces latency from 18s → ~0s on cache hit
        historical_bikes, is_simulated = get_cached_24h_history(
            station_code=station_code,
            current_bikes=current_bikes,
            capacity=capacity
        )
        
        # Step 4: Get current time features
        now = datetime.now(timezone.utc)
        hour = now.hour
        day_of_week = now.weekday()  # Monday=0, Sunday=6
        is_weekend = day_of_week >= 5  # Saturday=5, Sunday=6
        
        # Step 5: Prepare SageMaker payload
        payload = {
            "station_code": station_code,
            "historical_bikes": historical_bikes,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "capacity": capacity,
            "latitude": lat,
            "longitude": lon
        }
        
        # Step 6: Call SageMaker endpoint
        try:
            sagemaker_client = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
            
            response = sagemaker_client.invoke_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            # Parse response
            response_body = json.loads(response['Body'].read().decode())
            predictions = response_body.get('predictions', {})
            
            # Step 7: Return comprehensive response
            return {
                "station_code": station_code,
                "station_name": name,
                "current": {
                    "bikes_available": current_bikes,
                    "docks_available": int(station_data.get("numdocksavailable", 0)),
                    "capacity": capacity,
                    "timestamp": now.isoformat()
                },
                "predictions": {
                    "T+1h": {
                        "bikes": predictions.get('T+1h', 0),
                        "change": predictions.get('T+1h', 0) - current_bikes,
                        "time": (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0).isoformat()
                    },
                    "T+2h": {
                        "bikes": predictions.get('T+2h', 0),
                        "change": predictions.get('T+2h', 0) - current_bikes,
                        "time": (now + timedelta(hours=2)).replace(minute=0, second=0, microsecond=0).isoformat()
                    },
                    "T+3h": {
                        "bikes": predictions.get('T+3h', 0),
                        "change": predictions.get('T+3h', 0) - current_bikes,
                        "time": (now + timedelta(hours=3)).replace(minute=0, second=0, microsecond=0).isoformat()
                    }
                },
                "historical_24h": [
                    {
                        "bikes": bikes,
                        "time": (now - timedelta(hours=23-i)).replace(minute=0, second=0, microsecond=0).isoformat()
                    }
                    for i, bikes in enumerate(historical_bikes)
                ],
                "model": {
                    "version": response_body.get('model_version', 'unknown'),
                    "inference_time_ms": response_body.get('inference_time_ms', 0)
                },
                "metadata": {
                    "simulated_history": is_simulated,  # True if fallback used, False if real S3 data
                    "lat": lat,
                    "lon": lon
                }
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            raise HTTPException(
                status_code=503,
                detail=f"SageMaker error ({error_code}): {error_msg}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
