from typing import Any, Dict, List
from datetime import datetime, timezone, timedelta
import json
import os

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
        raw = fetch_live_all() if all else fetch_live(limit=limit)
        data: List[Dict[str, Any]] = normalize(raw)
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
        # Step 1: Fetch live data for all stations
        raw = fetch_live_all()
        data = normalize(raw)
        
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
        
        # Step 3: Fetch 24h historical data from S3 (with fallback to simulation)
        historical_bikes, is_simulated = get_24h_history_with_fallback(
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
