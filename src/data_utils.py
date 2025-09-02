from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable, List, Tuple


NumericFields = {"numdocksavailable", "numbikesavailable", "mechanical", "ebike"}
FloatFields = {"lat", "lon"}


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        try:
            # sometimes numeric strings like "3.0" appear
            f = float(value)
            return int(f)
        except Exception:
            return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def sanitize_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Coerce types for known fields and report issues.

    - NumericFields -> int or None
    - FloatFields -> float or None
    Returns (new_row, issues)
    """
    issues: List[str] = []
    out = dict(row)

    for k in NumericFields:
        if k in out:
            v = out[k]
            coerced = _to_int(v)
            if coerced is None and v not in (None, ""):
                issues.append(f"{k}: could not parse '{v}' as int")
            out[k] = coerced

    for k in FloatFields:
        if k in out:
            v = out[k]
            coerced = _to_float(v)
            if coerced is None and v not in (None, ""):
                issues.append(f"{k}: could not parse '{v}' as float")
            out[k] = coerced

    return out, issues


def validate_and_sanitize(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    all_rows: List[Dict[str, Any]] = []
    all_issues: List[str] = []
    for i, r in enumerate(rows):
        cleaned, issues = sanitize_row(r)
        if issues:
            all_issues.extend([f"row {i}: {msg}" for msg in issues])
        all_rows.append(cleaned)
    return all_rows, all_issues


def save_csv(records: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Determine headers: union of keys in order of first record, then others
    headers: List[str] = []
    seen = set()
    for rec in records:
        for k in rec.keys():
            if k not in seen:
                seen.add(k)
                headers.append(k)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)
