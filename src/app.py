from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query

from src.fetch_live_velib import fetch_live, fetch_live_all, normalize
from src.data_utils import validate_and_sanitize


app = FastAPI(title="Velib Trend API", version="0.1.0")


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
