import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Allow running both:
#   python -m src.snapshot_velib
#   python src/snapshot_velib.py
if __package__ is None or __package__ == "":
    # Add project root (parent of this file's directory) to sys.path so 'src.' imports work
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.fetch_live_velib import fetch_live_all  # type: ignore

# Snapshot script: fetch ALL station records, enrich with capture timestamp & staleness, write JSONL.
# Each line = one station record (raw fields) + capture_ts_utc + duedate_utc + staleness_sec.
# Filename pattern: velib_YYYYMMDD_HHMMSS.jsonl


def parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str or not isinstance(dt_str, str):
        return None
    try:
        # datetime.fromisoformat supports offsets like +02:00
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


def compute_staleness(capture_utc: datetime, duedate_raw: Optional[str]) -> (Optional[datetime], Optional[float]):
    d = parse_iso(duedate_raw)
    if not d:
        return None, None
    # Ensure timezone aware in UTC
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    d_utc = d.astimezone(timezone.utc)
    return d_utc, (capture_utc - d_utc).total_seconds()


def write_jsonl(records: List[Dict[str, Any]], out_dir: str, capture_ts: datetime) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts_label = capture_ts.strftime("%Y%m%d_%H%M%S")
    filename = f"velib_{ts_label}.jsonl"
    path = os.path.join(out_dir, filename)
    tmp_path = path + ".tmp"

    total = 0
    stale_values: List[float] = []
    max_staleness = -1.0
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in records:
                duedate_raw = rec.get("duedate")
                duedate_utc, staleness = compute_staleness(capture_ts, duedate_raw)
                line_obj = {
                    **rec,  # raw fields
                    "capture_ts_utc": capture_ts.isoformat().replace("+00:00", "Z"),
                    "duedate_utc": duedate_utc.isoformat().replace("+00:00", "Z") if duedate_utc else None,
                    "staleness_sec": round(staleness, 3) if staleness is not None else None,
                }
                if staleness is not None:
                    stale_values.append(staleness)
                    if staleness > max_staleness:
                        max_staleness = staleness
                f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
                total += 1
        os.replace(tmp_path, path)
    except Exception:
        # Cleanup tmp on failure
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    # Simple stats summary file (optional sidecar)
    if stale_values:
        stale_values_sorted = sorted(stale_values)
        mid = len(stale_values_sorted) // 2
        if len(stale_values_sorted) % 2:
            median = stale_values_sorted[mid]
        else:
            median = (stale_values_sorted[mid - 1] + stale_values_sorted[mid]) / 2
    else:
        median = None
    print(f"Snapshot written: {path}")
    print(f"Records: {total}")
    if median is not None:
        print(f"Median staleness (s): {median:.1f}")
    if max_staleness >= 0:
        print(f"Max staleness (s): {max_staleness:.1f}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Capture a full Velib availability snapshot (JSONL)")
    parser.add_argument("--out-dir", default="data/snapshots", help="Output directory for JSONL snapshots")
    parser.add_argument("--page-size", type=int, default=100, help="API page size (internal pagination)")
    parser.add_argument("--include-empty", action="store_true", help="Write file even if zero records returned")
    args = parser.parse_args()

    capture_ts = datetime.utcnow().replace(tzinfo=timezone.utc)
    try:
        records = fetch_live_all(page_size=args.page_size)
    except Exception as e:
        print(f"Fetch error: {e}")
        return 1

    if not records and not args.include_empty:
        print("No records fetched; skipping file (use --include-empty to force write).")
        return 0

    try:
        write_jsonl(records, args.out_dir, capture_ts)
    except Exception as e:
        print(f"Write error: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
