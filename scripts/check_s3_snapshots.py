#!/usr/bin/env python3
"""
Quick script to check recent snapshots in S3.
Useful for verifying the hourly Lambda collection is working.
"""

import boto3
from datetime import datetime, timezone

S3_BUCKET = "velib-trend-josh-dum-2025"
SNAPSHOT_PREFIX = "velib/snapshots/"
INDEX_PREFIX = "velib/index/"

def list_recent_snapshots(max_items: int = 24):
    """List the most recent snapshots from S3."""
    s3 = boto3.client('s3', region_name='eu-west-3')
    
    print(f"\n📊 Recent snapshots in s3://{S3_BUCKET}/{SNAPSHOT_PREFIX}")
    print("=" * 80)
    
    # List snapshot files
    response = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=SNAPSHOT_PREFIX,
        MaxKeys=max_items
    )
    
    if 'Contents' not in response:
        print("❌ No snapshots found!")
        return
    
    # Sort by last modified (newest first)
    snapshots = sorted(
        response['Contents'], 
        key=lambda x: x['LastModified'], 
        reverse=True
    )
    
    print(f"\nFound {len(snapshots)} snapshot(s):\n")
    
    for i, obj in enumerate(snapshots[:max_items], 1):
        key = obj['Key']
        size_kb = obj['Size'] / 1024
        modified = obj['LastModified'].astimezone(timezone.utc)
        age_hours = (datetime.now(timezone.utc) - modified).total_seconds() / 3600
        
        print(f"{i:2d}. {key.split('/')[-1]}")
        print(f"    Size: {size_kb:.1f} KB | Modified: {modified.strftime('%Y-%m-%d %H:%M:%S UTC')} ({age_hours:.1f}h ago)")
    
    # Check for gaps (expected: 1 snapshot per hour)
    if len(snapshots) >= 2:
        latest = snapshots[0]['LastModified']
        previous = snapshots[1]['LastModified']
        gap_hours = (latest - previous).total_seconds() / 3600
        
        print(f"\n⏱️  Gap between last two snapshots: {gap_hours:.2f} hours")
        if 0.9 <= gap_hours <= 1.1:
            print("   ✅ Hourly schedule is working correctly!")
        else:
            print(f"   ⚠️  Expected ~1 hour gap, got {gap_hours:.2f}h")

if __name__ == "__main__":
    list_recent_snapshots()
