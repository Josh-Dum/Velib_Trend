"""Build a hot-tier aggregated object for the last N hours.

Reads per-snapshot metadata JSON files under <snapshots_dir>/index/*.json,
selects the most recent N snapshot captures, reads each snapshot (JSONL or .jsonl.gz),
and builds compact per-station rolling windows. Writes gzipped JSON to
index/last24h/all_stations.json.gz by default.

Usage:
  python src/hot_builder.py --snapshots-dir data/snapshots --hours 24

Output format (example):
{
  "stations": {
    "12345": {
      "stationcode": "12345",
      "name": "Place X",
      "lat": 48.8,
      "lon": 2.3,
      "points": [
         {"t":"2025-09-29T13:00:00Z","b":12,"d":3},
         ...
      ]
    },
    ...
  },
  "generated_at": "2025-09-29T14:01:02Z",
  "snapshots_used": ["data/snapshots/velib_...jsonl.gz", ...]
}
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_iso(ts: str) -> datetime:
    if not ts:
        raise ValueError('empty timestamp')
    # support trailing Z
    if ts.endswith('Z'):
        ts = ts.replace('Z', '+00:00')
    return datetime.fromisoformat(ts)


def read_metadata_files(index_dir: Path) -> List[Dict]:
    metas: List[Dict] = []
    if not index_dir.exists():
        return metas
    for p in sorted(index_dir.glob('*.json')):
        try:
            j = json.loads(p.read_text(encoding='utf-8'))
            metas.append(j)
        except Exception:
            # skip invalid metadata file
            continue
    return metas


def read_jsonl(path: Path) -> Iterable[Dict]:
    if path.suffix == '.gz' or str(path).endswith('.jsonl.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    else:
        with path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def get_station_id(rec: Dict) -> str:
    for k in ('stationcode', 'station_id', 'id', 'code'):
        v = rec.get(k)
        if v:
            return str(v)
    # fallback to name
    return str(rec.get('name', '<unknown>'))


def build_hot_object(snapshots_dir: Path, hours: int) -> Dict:
    index_dir = snapshots_dir / 'index'
    metas = read_metadata_files(index_dir)
    if not metas:
        raise RuntimeError(f'No metadata files found in {index_dir}')

    # parse capture_ts and sort descending
    enriched = []
    for m in metas:
        ts = m.get('capture_ts') or ''
        try:
            dt = parse_iso(ts) if ts else None
        except Exception:
            dt = None
        enriched.append((dt, m))
    # keep only those with a valid timestamp
    enriched = [t for t in enriched if t[0] is not None]
    if not enriched:
        raise RuntimeError('No metadata entries with valid capture_ts')
    enriched.sort(key=lambda x: x[0], reverse=True)

    # Determine the window: use the latest capture_dt as anchor
    latest_dt = enriched[0][0]
    window_start = latest_dt - timedelta_hours(hours)

    # select metadata entries within window (inclusive)
    selected = [m for dt, m in reversed(enriched) if dt >= window_start]
    # we reversed to process oldest->newest

    stations: Dict[str, Dict] = {}
    snapshots_used: List[str] = []

    for m in selected:
        snap_key = m.get('snapshot_key')
        if not snap_key:
            continue
        # resolve path: prefer local path if provided
        p = Path(snap_key)
        if not p.exists():
            # try relative to snapshots_dir
            p = snapshots_dir / os.path.basename(snap_key)
        if not p.exists():
            # cannot access snapshot (maybe stored in S3). skip
            continue
        snapshots_used.append(str(p))
        # use capture_ts from metadata (preferred); fallback to per-record capture_ts_utc
        ts = m.get('capture_ts')
        for rec in read_jsonl(p):
            sid = get_station_id(rec)
            if sid not in stations:
                # try top-level lat/lon, then nested coordonnees_geo (as in sample snapshots)
                coords = rec.get('coordonnees_geo') or {}
                lat = rec.get('lat') if rec.get('lat') is not None else coords.get('lat')
                lon = rec.get('lon') if rec.get('lon') is not None else coords.get('lon')
                stations[sid] = {
                    'stationcode': sid,
                    'name': rec.get('name'),
                    'lat': lat,
                    'lon': lon,
                    'points': deque(maxlen=hours + 2),
                }
            # compact point
            point_ts = ts or rec.get('capture_ts_utc') or rec.get('duedate_utc')
            point = {
                't': point_ts,
                'b': _safe_int(rec.get('numbikesavailable')),
                'd': _safe_int(rec.get('numdocksavailable')),
            }
            stations[sid]['points'].append(point)

    # materialize deques into lists and drop empty stations
    out_stations = {}
    for sid, info in stations.items():
        pts = list(info['points'])
        if not pts:
            continue
        out_stations[sid] = {
            'stationcode': info.get('stationcode'),
            'name': info.get('name'),
            'lat': info.get('lat'),
            'lon': info.get('lon'),
            'points': pts,
        }

    result = {
        'stations': out_stations,
        'generated_at': datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z'),
        'snapshots_used': snapshots_used,
    }
    return result


def _safe_int(v):
    try:
        if v is None or v == '':
            return None
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def timedelta_hours(h: int):
    from datetime import timedelta

    return timedelta(hours=h)


def write_gz_json(obj: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, 'wt', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshots-dir', type=str, default='data/snapshots')
    p.add_argument('--hours', type=int, default=24)
    p.add_argument('--out', type=str, default='index/last24h/all_stations.json.gz')
    return p.parse_args()


def main():
    args = parse_args()
    snapshots_dir = Path(args.snapshots_dir)
    out_path = Path(args.out)
    obj = build_hot_object(snapshots_dir, args.hours)
    write_gz_json(obj, out_path)
    print(f'Wrote hot-tier object to {out_path} (stations={len(obj["stations"])})')


if __name__ == '__main__':
    main()
