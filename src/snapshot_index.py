import gzip
import json
import os
from statistics import median
from typing import Optional
import io
import boto3
import botocore
from pathlib import Path


def _open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def compute_snapshot_stats(path: str):
    """Read a JSONL (or JSONL.GZ) snapshot and compute simple stats.

    Returns dict with keys: records, median_staleness_s, max_staleness_s, pct_stale, stale_count
    """
    staleness_vals = []
    stale_count = 0
    total = 0
    try:
        with _open_maybe_gzip(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                s = obj.get("staleness_sec")
                if s is not None:
                    try:
                        sv = float(s)
                        staleness_vals.append(sv)
                    except Exception:
                        pass
                if obj.get("stale_flag") is True:
                    stale_count += 1
    except FileNotFoundError:
        raise

    if staleness_vals:
        median_s = float(median(staleness_vals))
        max_s = float(max(staleness_vals))
    else:
        median_s = None
        max_s = None

    pct_stale = (stale_count / total * 100) if total > 0 else None

    return {
        "records": total,
        "median_staleness_s": median_s,
        "max_staleness_s": max_s,
        "stale_count": stale_count,
        "pct_stale": pct_stale,
    }


def append_index_for_file(snapshot_path: str, index_path: Optional[str] = None) -> str:
    """Write a per-snapshot metadata JSON next to the snapshot or to S3.

    The metadata file contains the minimal fields required by the project:
      - snapshot_key: S3 key or local path to the Bronze snapshot
      - capture_ts: ISO8601 capture timestamp (best-effort)
      - uploader_version: from env var UPLOADER_VERSION or 'unknown'
      - rows_bronze: integer record count

    If `index_path` is an S3 URI (s3://bucket/prefix/), the metadata will be
    uploaded to `s3://bucket/prefix/index/<snapshot_basename>.json`. If no
    index_path is provided, the metadata is written locally next to the
    snapshot under `<snapshot_dir>/index/<snapshot_basename>.json`.
    Returns the path (local or s3 uri) of the written metadata file.
    """
    stats = compute_snapshot_stats(snapshot_path)
    rows = stats.get("records", 0)
    basename = os.path.basename(snapshot_path)
    stem = basename
    # normalize stem to .json (remove .jsonl(.gz) suffixes)
    if stem.endswith('.jsonl.gz'):
        stem = stem[:-8]
    elif stem.endswith('.jsonl'):
        stem = stem[:-6]

    # best-effort capture_ts from filename
    capture_ts = None
    try:
        if basename.startswith("velib_"):
            ts_part = basename.split("velib_")[1].split(".")[0]
            if "_" in ts_part:
                d, t = ts_part.split("_")
                capture_ts = f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}Z"
    except Exception:
        capture_ts = None

    uploader_version = os.environ.get('UPLOADER_VERSION', 'unknown')

    metadata = {
        "snapshot_key": snapshot_path,
        "capture_ts": capture_ts or "",
        "uploader_version": uploader_version,
        "rows_bronze": rows,
    }

    if index_path and index_path.startswith('s3://'):
        # parse s3://bucket/prefix
        parts = index_path[5:].split('/', 1)
        s3_bucket = parts[0]
        s3_prefix = parts[1] if len(parts) > 1 else ''
        key_prefix = s3_prefix.rstrip('/') + '/index' if s3_prefix else 'index'
        key = f"{key_prefix}/{stem}.json"
        s3 = boto3.client('s3')
        s3.put_object(Bucket=s3_bucket, Key=key, Body=json.dumps(metadata).encode('utf-8'))
        return f's3://{s3_bucket}/{key}'

    # local write: snapshots dir / index / stem.json
    out_dir = Path(snapshot_path).parent / 'index'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(out_path)
