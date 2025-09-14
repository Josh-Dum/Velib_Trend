"""
Convert JSONL(.gz) Velib snapshots (Bronze) into partitioned Parquet (Silver).

This script intentionally does NOT add any new engineered features beyond
what exists in the raw JSON; it flattens the raw structure and prefixes
original fields with `raw_` to preserve provenance. It writes Parquet
partitioned by `date` and `hour` (based on `capture_ts_utc`).

Usage examples:
  python src/convert_snapshots_to_parquet.py --snapshots-dir data/snapshots --out-dir data/silver --use-index

CLI flags:
  --snapshots-dir : directory containing *.jsonl or *.jsonl.gz snapshots
  --out-dir       : output directory for parquet partitions
  --use-index     : use data/snapshots/index.csv to pick files (default: False)
  --date          : process only snapshots matching a specific date YYYY-MM-DD
  --overwrite     : overwrite existing parquet output for matching partitions

The script uses pandas + pyarrow. It performs atomic writes (tmp file + replace).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def list_snapshot_files(snapshots_dir: Path, use_index: bool) -> List[Path]:
    if use_index:
        index_path = snapshots_dir / "index.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found at {index_path}")
        df = pd.read_csv(index_path)
        files = [snapshots_dir / f for f in df['file_name'].tolist() if (snapshots_dir / f).exists()]
        return files

    patterns = ["*.jsonl", "*.jsonl.gz"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(snapshots_dir.glob(pat)))
    return files


def read_jsonl(file_path: Path) -> Iterable[dict]:
    if file_path.suffix == ".gz" or file_path.name.endswith('.jsonl.gz'):
        with gzip.open(file_path, 'rt', encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                yield json.loads(line)
    else:
        with file_path.open('r', encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                yield json.loads(line)


def flatten_record(rec: dict) -> dict:
    """Flatten top-level keys. Nested dicts are JSON-dumped to preserve raw content.

    We prefix every original top-level key with `raw_` to avoid collision with
    any future engineered columns.
    """
    out = {}
    for k, v in rec.items():
        if isinstance(v, (dict, list)):
            try:
                out[f"raw_{k}"] = json.dumps(v, ensure_ascii=False)
            except Exception:
                out[f"raw_{k}"] = str(v)
        else:
            out[f"raw_{k}"] = v
    return out


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet_atomic(df: pd.DataFrame, out_path: Path, overwrite: bool = False) -> None:
    if out_path.exists():
        if overwrite:
            out_path.unlink()
        else:
            raise FileExistsError(f"Output file exists: {out_path}")

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    ensure_dir(tmp)
    df.to_parquet(tmp, index=False, compression='snappy')
    # atomic replace
    shutil.move(str(tmp), str(out_path))


def process_files(files: List[Path], out_dir: Path, date_filter: str | None, overwrite: bool) -> None:
    for fp in files:
        print(f"Processing {fp}")
        rows = []
        for rec in read_jsonl(fp):
            flat = flatten_record(rec)
            # preserve capture_ts if present, else use file-modified time
            capture = flat.get('raw_capture_ts_utc') or flat.get('raw_capture_ts')
            if capture:
                try:
                    capture_dt = datetime.fromisoformat(capture.replace('Z', '+00:00'))
                except Exception:
                    # fallback parse
                    capture_dt = datetime.utcfromtimestamp(0)
            else:
                capture_dt = datetime.utcfromtimestamp(fp.stat().st_mtime)

            flat['capture_ts_utc'] = capture_dt.isoformat()
            flat['date'] = capture_dt.strftime('%Y-%m-%d')
            flat['hour'] = capture_dt.strftime('%H')
            rows.append(flat)

        if not rows:
            print(f"No records in {fp}, skipping")
            continue

        df = pd.DataFrame(rows)

        # apply date filter if requested
        if date_filter and not df['date'].eq(date_filter).any():
            print(f"Skipping {fp} due to date filter {date_filter}")
            continue

        # write partitioned by date/hour into out_dir
        for (date_val, hour_val), part in df.groupby(['date', 'hour']):
            part_dir = out_dir / f"date={date_val}" / f"hour={hour_val}"
            part_dir.mkdir(parents=True, exist_ok=True)
            out_file = part_dir / f"{fp.stem}.parquet"
            write_parquet_atomic(part, out_file, overwrite=overwrite)
            print(f"Wrote {out_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--snapshots-dir', type=str, default='data/snapshots', help='Directory with JSONL(.gz) snapshots')
    p.add_argument('--out-dir', type=str, default='data/silver', help='Output directory for Parquet')
    p.add_argument('--use-index', action='store_true', help='Use index.csv in snapshots dir to list files')
    p.add_argument('--date', type=str, default=None, help='Only process snapshots for this date YYYY-MM-DD')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing parquet files')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    snapshots_dir = Path(args.snapshots_dir)
    out_dir = Path(args.out_dir)

    files = list_snapshot_files(snapshots_dir, args.use_index)
    if not files:
        print(f"No snapshot files found in {snapshots_dir}")
        return

    process_files(files, out_dir, args.date, args.overwrite)


if __name__ == '__main__':
    main()
