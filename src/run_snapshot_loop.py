import argparse
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Optional

# Allow both direct and module execution (same pattern as snapshot script)
if __package__ is None or __package__ == "":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.snapshot_velib import capture_snapshot  # type: ignore
from src.snapshot_index import append_index_for_file  # type: ignore


def align_sleep(interval_sec: int) -> float:
    now = time.time()
    return interval_sec - (now % interval_sec)


def main():
    parser = argparse.ArgumentParser(
        description="Run periodic Velib snapshots in a loop"
    )
    parser.add_argument("--interval-sec", type=int, default=600, help="Interval between successful snapshots")
    parser.add_argument("--max-runs", type=int, default=0, help="Stop after N runs (0 = infinite)")
    parser.add_argument("--out-dir", default="data/snapshots", help="Snapshot output directory")
    parser.add_argument("--page-size", type=int, default=100, help="API page size for pagination")
    parser.add_argument("--stale-threshold-sec", type=int, default=900, help="Stale threshold seconds (negative disables)")
    parser.add_argument("--no-gzip", action="store_true", help="Disable gzip compression")
    parser.add_argument("--include-empty", action="store_true", help="Write file even if zero records")
    parser.add_argument('--to-s3', action='store_true', help='Upload snapshot to S3')
    parser.add_argument('--s3-bucket', type=str, default=None, help='S3 bucket name to upload snapshots')
    parser.add_argument('--s3-prefix', type=str, default='snapshots', help='S3 key prefix')
    parser.add_argument('--remove-local', action='store_true', help='Remove local snapshot file after successful S3 upload')
    parser.add_argument('--s3-upload-retries', type=int, default=3, help='S3 upload retry count')
    parser.add_argument("--no-index", action="store_true", help="Disable writing a snapshot index (enabled by default)")
    parser.add_argument("--jitter-sec", type=int, default=0, help="Add up to this many random seconds before each run")
    parser.add_argument("--align", action="store_true", help="Align first run to the next interval boundary")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-run output")
    args = parser.parse_args()

    runs = 0
    consecutive_failures = 0
    base_backoff = 10  # seconds
    max_backoff = min(args.interval_sec, 300)

    if args.align:
        sleep_first = align_sleep(args.interval_sec)
        if not args.quiet:
            print(f"[loop] Aligning first run in {sleep_first:.1f}s (interval={args.interval_sec}s)")
        time.sleep(sleep_first)

    try:
        while True:
            if args.jitter_sec > 0:
                jit = random.uniform(0, args.jitter_sec)
                if not args.quiet:
                    print(f"[loop] Jitter sleep {jit:.1f}s")
                time.sleep(jit)

            start = time.time()
            start_ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            if not args.quiet:
                print(f"[loop] Run {runs + 1} start {start_ts}")
            try:
                result = capture_snapshot(
                    out_dir=args.out_dir,
                    page_size=args.page_size,
                    stale_threshold_sec=args.stale_threshold_sec,
                    gzip_enabled=not args.no_gzip,
                    include_empty=args.include_empty,
                    # S3 options forwarded
                    # These are optional; capture_snapshot currently uploads and returns local path
                    # We will handle index writing to S3 below if --to-s3 is set
                    
                )
                consecutive_failures = 0
                if result.get("skipped"):
                    if not args.quiet:
                        print("[loop] Skipped (no records)")
                else:
                    if not args.quiet:
                        print(
                            f"[loop] Wrote {result['records']} records -> {result['path']}"
                        )
                    # Append index entry unless disabled
                    if result.get("path") and not args.no_index:
                        try:
                            # If uploading to S3 and s3_bucket set, write index to S3 path
                            if args.to_s3 and args.s3_bucket:
                                # create s3 index prefix s3://bucket/prefix/index/
                                prefix = args.s3_prefix.rstrip('/') if args.s3_prefix else ''
                                if prefix:
                                    index_s3_uri = f"s3://{args.s3_bucket}/{prefix}/"
                                else:
                                    index_s3_uri = f"s3://{args.s3_bucket}/"
                                append_index_for_file(result["path"], index_path=index_s3_uri)
                            else:
                                append_index_for_file(result["path"])
                        except Exception as e:
                            print(f"[loop] index append error: {e}")
            except Exception as e:
                consecutive_failures += 1
                backoff = min(base_backoff * (2 ** (consecutive_failures - 1)), max_backoff)
                print(f"[loop] ERROR ({consecutive_failures}) {e} -> sleeping {backoff}s")
                time.sleep(backoff)
                continue  # retry loop after backoff

            runs += 1
            if args.max_runs and runs >= args.max_runs:
                if not args.quiet:
                    print("[loop] Reached max_runs, exiting")
                break

            elapsed = time.time() - start
            sleep_time = max(0.0, args.interval_sec - elapsed)
            if not args.quiet:
                print(f"[loop] Sleeping {sleep_time:.1f}s (elapsed {elapsed:.1f}s)")
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        if not args.quiet:
            print("[loop] Interrupted by user")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())