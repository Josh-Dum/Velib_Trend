import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import requests
from src.data_utils import validate_and_sanitize, save_csv

API_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records"


def fetch_live(limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch a small sample of live Velib station availability.

    Inputs:
      - limit: max number of station records to fetch
    Output:
      - list of records (dicts)
    Error modes:
      - raises for non-200 responses
      - returns empty list if payload missing
    """
    params = {
        "limit": limit,
        "timezone": "Europe/Paris",
    }
    r = requests.get(API_URL, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    records = payload.get("results", []) if isinstance(payload, dict) else []
    return records


def fetch_live_all(page_size: int = 100, max_pages: int = 50) -> List[Dict[str, Any]]:
        """Fetch all live Velib station availability by paginating.

        Inputs:
            - page_size: number of records per page requested from the API
            - max_pages: safety upper bound on pages to prevent infinite loops
        Output:
            - list of all records
        Notes:
            - Uses 'limit' and 'offset' pagination.
            - Stops when a page returns fewer than page_size or max_pages reached.
        """
        all_records: List[Dict[str, Any]] = []
        offset = 0
        pages = 0
        while True:
            params = {
                "limit": page_size,
                "offset": offset,
                "timezone": "Europe/Paris",
            }
            try:
                r = requests.get(API_URL, params=params, timeout=20)
                r.raise_for_status()
            except requests.HTTPError as e:
                # If the provider rejects the page (e.g., limit too high), try a smaller page.
                if page_size > 50:
                    page_size = 50
                    continue
                raise e
            payload = r.json()
            batch = payload.get("results", []) if isinstance(payload, dict) else []
            if not batch:
                break
            all_records.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
            pages += 1
            if pages >= max_pages:
                break
        return all_records


essential_fields = [
    "stationcode",
    "name",
    "capacity",
    "numdocksavailable",
    "numbikesavailable",
    "mechanical",
    "ebike",
    "coordonnees_geo",
    "duedate",
    "nom_arrondissement_communes",
    "code_insee_commune",
]


def normalize(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only essential fields and flatten coordinates."""
    out = []
    for rec in records:
        row = {k: rec.get(k) for k in essential_fields}
        coords = row.pop("coordonnees_geo", None)
        if isinstance(coords, dict):
            row["lat"] = coords.get("lat")
            row["lon"] = coords.get("lon")
        elif isinstance(coords, list) and len(coords) == 2:
            # Sometimes coords can be [lat, lon]
            row["lat"], row["lon"] = coords[0], coords[1]
        out.append(row)
    return out


def save_json(records: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stamp = datetime.utcnow().isoformat()
    payload = {
        "fetched_at_utc": stamp,
        "count": len(records),
        "records": records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch live Velib sample")
    parser.add_argument("--limit", type=int, default=5, help="number of records to fetch")
    parser.add_argument("--out", type=str, default="data/sample.json", help="output JSON path")
    parser.add_argument("--validate", action="store_true", help="validate and coerce numeric/float fields")
    parser.add_argument("--csv", type=str, default=None, help="optional CSV output path (e.g., data/sample.csv)")
    args = parser.parse_args()

    try:
        raw = fetch_live(limit=args.limit)
        norm = normalize(raw)
        if args.validate:
            norm, issues = validate_and_sanitize(norm)
            if issues:
                print(f"Validation warnings: {len(issues)} issue(s)")
                for msg in issues[:10]:
                    print(" -", msg)
                if len(issues) > 10:
                    print(f" ... {len(issues) - 10} more")
        save_json(norm, args.out)
        print(f"Saved {len(norm)} records to {args.out}")
        if args.csv:
            save_csv(norm, args.csv)
            print(f"Saved CSV to {args.csv}")
    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
