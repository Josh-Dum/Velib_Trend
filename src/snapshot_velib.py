import argparse
import json
import os
import sys
import gzip
import io
import time
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
import botocore

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


def write_jsonl(
    records: List[Dict[str, Any]],
    out_dir: str,
    capture_ts: datetime,
    stale_threshold_sec: Optional[int] = None,
    gzip_enabled: bool = True,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts_label = capture_ts.strftime("%Y%m%d_%H%M%S")
    filename = f"velib_{ts_label}.jsonl" + (".gz" if gzip_enabled else "")
    path = os.path.join(out_dir, filename)
    tmp_path = path + ".tmp"

    total = 0
    stale_values: List[float] = []
    max_staleness = -1.0
    stale_count = 0
    try:
        # Choose file opener (gzip text mode if enabled)
        opener = gzip.open if gzip_enabled else open
        mode = "wt" if gzip_enabled else "w"
        with opener(tmp_path, mode, encoding="utf-8") as f:  # type: ignore
            for rec in records:
                duedate_raw = rec.get("duedate")
                duedate_utc, staleness = compute_staleness(capture_ts, duedate_raw)
                stale_flag: Optional[bool] = None
                if staleness is not None and stale_threshold_sec is not None and stale_threshold_sec >= 0:
                    stale_flag = staleness >= stale_threshold_sec
                    if stale_flag:
                        stale_count += 1
                line_obj = {
                    **rec,  # raw fields
                    "capture_ts_utc": capture_ts.isoformat().replace("+00:00", "Z"),
                    "duedate_utc": duedate_utc.isoformat().replace("+00:00", "Z") if duedate_utc else None,
                    "staleness_sec": round(staleness, 3) if staleness is not None else None,
                    "stale_flag": stale_flag,
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
    print(f"Snapshot written: {path} (gzip={'on' if gzip_enabled else 'off'})")
    print(f"Records: {total}")
    if median is not None:
        print(f"Median staleness (s): {median:.1f}")
    if max_staleness >= 0:
        print(f"Max staleness (s): {max_staleness:.1f}")
    if stale_threshold_sec is not None and stale_threshold_sec >= 0 and total > 0:
        pct = (stale_count / total) * 100
        print(f"Stale (>= {stale_threshold_sec}s): {stale_count} ({pct:.1f}%)")
    return path


def _upload_to_s3(local_path: str, bucket: str, prefix: str = 'snapshots', region: Optional[str] = None, retries: int = 3, metadata: Optional[dict] = None) -> str:
    """Upload local file to S3 and return s3://... key. Retries on transient errors."""
    # Basic validation
    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)

    key = f"{prefix.rstrip('/')}/{os.path.basename(local_path)}"
    s3 = boto3.client('s3', region_name=region) if region else boto3.client('s3')

    # Ensure metadata values are strings and keys lowercase (S3 returns metadata as lower-case)
    extra_args = {}
    if metadata:
        extra_args['Metadata'] = {str(k).lower(): str(v) for k, v in metadata.items()}

    attempt = 0
    base = 1
    max_backoff = 30
    while True:
        attempt += 1
        try:
            # upload_file handles multipart and streaming
            s3.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
            return f"s3://{bucket}/{key}"
        except botocore.exceptions.NoCredentialsError:
            # Permanent configuration error — don't retry
            raise
        except botocore.exceptions.PartialCredentialsError:
            raise
        except botocore.exceptions.ClientError as e:
            # Inspect common permanent failure codes and fail-fast for them
            code = None
            try:
                code = e.response.get('Error', {}).get('Code')
            except Exception:
                code = None
            if code in ('AccessDenied', 'InvalidBucketName', 'InvalidAccessKeyId', 'PermanentRedirect'):
                raise
            last_exc = e
        except botocore.exceptions.BotoCoreError as e:
            # network/transport related errors — treat as retryable
            last_exc = e

        # If we've exhausted retries, re-raise last exception
        if attempt >= retries:
            raise last_exc

        # Exponential backoff with jitter
        sleep = min(max_backoff, base * (2 ** (attempt - 1))) + random.uniform(0, 1)
        print(f"[s3-upload] attempt {attempt}/{retries} failed: {last_exc} -> sleeping {sleep:.1f}s")
        time.sleep(sleep)



def main():
    parser = argparse.ArgumentParser(description="Capture a full Velib availability snapshot (JSONL)")
    parser.add_argument("--out-dir", default="data/snapshots", help="Output directory for JSONL snapshots")
    parser.add_argument("--page-size", type=int, default=100, help="API page size (internal pagination)")
    parser.add_argument("--include-empty", action="store_true", help="Write file even if zero records returned")
    parser.add_argument(
        "--stale-threshold-sec",
        type=int,
        default=900,
        help="Threshold in seconds to mark stale_flag (set negative to disable)",
    )
    parser.add_argument(
        "--no-gzip",
        action="store_true",
        help="Disable gzip compression (enabled by default)",
    )
    parser.add_argument('--to-s3', action='store_true', help='Upload snapshot to S3')
    parser.add_argument('--s3-bucket', type=str, default=None, help='S3 bucket name to upload snapshots')
    parser.add_argument('--s3-prefix', type=str, default='snapshots', help='S3 key prefix')
    parser.add_argument('--remove-local', action='store_true', help='Remove local snapshot file after successful S3 upload')
    parser.add_argument('--s3-upload-retries', type=int, default=3, help='S3 upload retry count')
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
        threshold = args.stale_threshold_sec if args.stale_threshold_sec >= 0 else None
        path = write_jsonl(
            records,
            args.out_dir,
            capture_ts,
            stale_threshold_sec=threshold,
            gzip_enabled=not args.no_gzip,
        )
        # Optionally upload to S3
        if args.to_s3:
            if not args.s3_bucket:
                print('Missing --s3-bucket for S3 upload')
                return 3
            meta = {
                'original_filename': os.path.basename(path),
                'gzip': str(not args.no_gzip),
                'capture_ts_utc': capture_ts.isoformat().replace('+00:00', 'Z'),
            }
            try:
                s3uri = _upload_to_s3(path, args.s3_bucket, prefix=args.s3_prefix, retries=args.s3_upload_retries, metadata=meta)
                print(f'Uploaded to {s3uri}')
                if args.remove_local:
                    try:
                        os.remove(path)
                        print('Removed local snapshot after upload')
                    except Exception as e:
                        print('Failed to remove local snapshot:', e)
            except Exception as e:
                print('S3 upload failed:', e)
                return 4
    except Exception as e:
        print(f"Write error: {e}")
        return 2
    return 0


def capture_snapshot(
    out_dir: str = "data/snapshots",
    page_size: int = 100,
    stale_threshold_sec: int = 900,
    gzip_enabled: bool = True,
    include_empty: bool = False,
) -> Dict[str, Any]:
    """Programmatic API to capture a snapshot (used by loop runner).

    Returns a dict with keys: path, records, capture_ts, stale_threshold_sec.
    Raises exceptions on fetch or write errors.
    """
    capture_ts = datetime.utcnow().replace(tzinfo=timezone.utc)
    records = fetch_live_all(page_size=page_size)
    if not records and not include_empty:
        return {
            "path": None,
            "records": 0,
            "capture_ts": capture_ts,
            "stale_threshold_sec": stale_threshold_sec,
            "skipped": True,
        }
    threshold = stale_threshold_sec if stale_threshold_sec >= 0 else None
    path = write_jsonl(
        records,
        out_dir,
        capture_ts,
        stale_threshold_sec=threshold,
        gzip_enabled=gzip_enabled,
    )
    return {
        "path": path,
        "records": len(records),
        "capture_ts": capture_ts,
        "stale_threshold_sec": stale_threshold_sec,
        "skipped": False,
    }
    return {
        "path": path,
        "records": len(records),
        "capture_ts": capture_ts,
        "stale_threshold_sec": stale_threshold_sec,
        "skipped": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
