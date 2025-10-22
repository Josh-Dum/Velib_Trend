"""
Test script for historical data fetching from S3.

This script tests the S3 historical data fetching functionality
to ensure we can retrieve 24-hour bike availability history.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fetch_historical import get_24h_history, get_24h_history_with_fallback, get_last_24_snapshots


def test_list_snapshots():
    """Test listing the last 24 snapshots from S3."""
    print("=" * 70)
    print("TEST 1: Listing Last 24 Snapshots")
    print("=" * 70)
    
    snapshot_keys = get_last_24_snapshots()
    
    print(f"\nFound {len(snapshot_keys)} snapshots")
    if snapshot_keys:
        print(f"\nFirst (oldest): {snapshot_keys[0]}")
        print(f"Last (newest):  {snapshot_keys[-1]}")
        print(f"\nAll snapshots:")
        for i, key in enumerate(snapshot_keys):
            print(f"  {i+1:2d}. {key}")
    else:
        print("⚠️  No snapshots found!")
    
    return len(snapshot_keys) >= 20


def test_fetch_history(station_code: str = "16107"):
    """Test fetching 24h history for a specific station."""
    print("\n" + "=" * 70)
    print(f"TEST 2: Fetching 24h History for Station {station_code}")
    print("=" * 70)
    
    history = get_24h_history(station_code)
    
    if history:
        print(f"\n✅ Successfully fetched history!")
        print(f"   Length: {len(history)} hours")
        print(f"   Values: {history}")
        print(f"   Min: {min(history)} bikes")
        print(f"   Max: {max(history)} bikes")
        print(f"   Avg: {sum(history) / len(history):.1f} bikes")
        return True
    else:
        print("\n❌ Could not fetch history from S3")
        return False


def test_with_fallback(station_code: str = "16107"):
    """Test fetching with fallback to simulated data."""
    print("\n" + "=" * 70)
    print(f"TEST 3: Fetching with Fallback for Station {station_code}")
    print("=" * 70)
    
    # Use realistic fallback values
    current_bikes = 10
    capacity = 35
    
    history, is_simulated = get_24h_history_with_fallback(
        station_code=station_code,
        current_bikes=current_bikes,
        capacity=capacity
    )
    
    print(f"\n✅ Got history (simulated: {is_simulated})")
    print(f"   Length: {len(history)} hours")
    print(f"   Values: {history}")
    print(f"   Min: {min(history)} bikes")
    print(f"   Max: {max(history)} bikes")
    print(f"   Avg: {sum(history) / len(history):.1f} bikes")
    
    if is_simulated:
        print("\n⚠️  Using SIMULATED data (S3 fetch failed or insufficient data)")
    else:
        print("\n✅ Using REAL data from S3!")
    
    return True


if __name__ == "__main__":
    print("\n🧪 TESTING S3 HISTORICAL DATA FETCHING")
    print("=" * 70)
    
    # Test 1: List snapshots
    has_snapshots = test_list_snapshots()
    
    if not has_snapshots:
        print("\n⚠️  Not enough snapshots in S3. Tests may use simulated data.")
    
    # Test 2: Fetch real history
    test_fetch_history("16107")
    
    # Test 3: Test with fallback
    test_with_fallback("16107")
    
    # Test 4: Test with non-existent station (should fallback)
    test_with_fallback("99999")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 70)
