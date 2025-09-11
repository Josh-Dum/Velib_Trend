import csv
import gzip
import json
import os
from statistics import median
from typing import Optional


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


def append_index_for_file(snapshot_path: str, index_path: Optional[str] = None):
    """Append a summary row for `snapshot_path` to `index.csv`.

    If `index_path` is None, use sibling file `index.csv` in the same directory.
    """
    if index_path is None:
        index_path = os.path.join(os.path.dirname(snapshot_path), "index.csv")

    stats = compute_snapshot_stats(snapshot_path)
    file_bytes = os.path.getsize(snapshot_path)
    gzip_flag = snapshot_path.endswith(".gz")
    capture_ts = None
    # try to parse capture_ts from filename pattern velib_YYYYMMDD_HHMMSS
    basename = os.path.basename(snapshot_path)
    try:
        if basename.startswith("velib_"):
            ts_part = basename.split("velib_")[1].split(".")[0]
            # format YYYYMMDD_HHMMSS
            if "_" in ts_part:
                d, t = ts_part.split("_")
                capture_ts = f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}Z"
    except Exception:
        capture_ts = None

    header = [
        "file_name",
        "capture_ts_utc",
        "records",
        "median_staleness_s",
        "max_staleness_s",
        "pct_stale",
        "stale_count",
        "gzip",
        "file_bytes",
    ]

    need_header = not os.path.exists(index_path)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        writer.writerow([
            basename,
            capture_ts or "",
            stats["records"],
            f"{stats['median_staleness_s']:.3f}" if stats["median_staleness_s"] is not None else "",
            f"{stats['max_staleness_s']:.3f}" if stats["max_staleness_s"] is not None else "",
            f"{stats['pct_stale']:.2f}" if stats["pct_stale"] is not None else "",
            stats["stale_count"],
            str(gzip_flag),
            file_bytes,
        ])

    return index_path
