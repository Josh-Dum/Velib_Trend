#!/usr/bin/env python3
"""
Bronze → Silver Data Converter
==============================

Purpose: Transform raw snapshots (Bronze) into clean ML-ready data (Silver).

What this script does:
1. Downloads all snapshots from S3 to local temp directory
2. Loads and parses all JSONL files
3. Cleans data (removes unnecessary fields)
4. Adds temporal features for ML training
5. Saves as a single Parquet file for fast loading

Author: Josh
Date: October 9, 2025
"""

import boto3
import pandas as pd
import json
import gzip
from pathlib import Path
from datetime import datetime
from io import BytesIO
import os
import sys

# Configuration
S3_BUCKET = "velib-trend-josh-dum-2025"
S3_PREFIX = "velib/snapshots/"
REGION = "eu-west-3"
OUTPUT_DIR = Path("data/silver")
OUTPUT_FILE = OUTPUT_DIR / "velib_training_data.parquet"


def check_aws_credentials():
    """
    Check if AWS credentials are configured.
    
    Returns:
        bool: True if credentials are found, False otherwise
    """
    # Check environment variables
    if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
        return True
    
    # Check AWS credentials file
    aws_creds_file = Path.home() / '.aws' / 'credentials'
    if aws_creds_file.exists():
        return True
    
    return False


def download_snapshots_from_s3():
    """
    Download all snapshots from S3 and return list of records.
    
    Returns:
        list: All records from all snapshots
    """
    print("🔄 Step 1: Connecting to S3...")
    s3 = boto3.client('s3', region_name=REGION)
    
    # List all snapshots
    print(f"📥 Step 2: Listing snapshots from s3://{S3_BUCKET}/{S3_PREFIX}")
    response = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=S3_PREFIX
    )
    
    if 'Contents' not in response:
        raise ValueError("No snapshots found in S3!")
    
    snapshots = response['Contents']
    print(f"✅ Found {len(snapshots)} snapshots")
    
    # Download and parse each snapshot
    print(f"\n🔄 Step 3: Downloading and parsing snapshots...")
    all_records = []
    
    for i, obj in enumerate(snapshots, 1):
        key = obj['Key']
        filename = key.split('/')[-1]
        
        try:
            # Download from S3
            response = s3.get_object(Bucket=S3_BUCKET, Key=key)
            gzip_content = response['Body'].read()
            
            # Decompress and parse JSONL
            with gzip.open(BytesIO(gzip_content), 'rt', encoding='utf-8') as f:
                records = [json.loads(line) for line in f]
            
            all_records.extend(records)
            
            if i % 10 == 0:  # Progress every 10 files
                print(f"  ✓ Processed {i}/{len(snapshots)} snapshots ({len(all_records):,} records so far)")
        
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
            continue
    
    print(f"✅ Downloaded {len(all_records):,} total records from {len(snapshots)} snapshots\n")
    return all_records


def create_dataframe(records):
    """
    Convert records to DataFrame and add initial timestamp parsing.
    
    Args:
        records: List of dictionaries from snapshots
        
    Returns:
        pd.DataFrame: Initial dataframe with parsed timestamps
    """
    print("🔄 Step 4: Creating DataFrame...")
    df = pd.DataFrame(records)
    
    # Parse timestamp
    df['capture_time'] = pd.to_datetime(df['capture_ts_utc'])
    
    print(f"✅ DataFrame created: {len(df):,} rows × {len(df.columns)} columns")
    print(f"   Time range: {df['capture_time'].min()} → {df['capture_time'].max()}")
    print(f"   Unique stations: {df['stationcode'].nunique()}\n")
    
    return df


def clean_data(df):
    """
    Clean the data by removing unnecessary fields.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    print("🔄 Step 5: Cleaning data...")
    
    # Fields to keep
    keep_fields = [
        'stationcode',           # Station identifier
        'numbikesavailable',     # TARGET variable
        'numdocksavailable',     # Useful context
        'capacity',              # Station capacity
        'capture_time',          # Timestamp
        'capture_ts_utc',        # Original timestamp string
    ]
    
    # Extract lat/lon from coordonnees_geo if exists
    if 'coordonnees_geo' in df.columns:
        print("  → Extracting coordinates...")
        df['latitude'] = df['coordonnees_geo'].apply(
            lambda x: x.get('lat') if isinstance(x, dict) else None
        )
        df['longitude'] = df['coordonnees_geo'].apply(
            lambda x: x.get('lon') if isinstance(x, dict) else None
        )
        keep_fields.extend(['latitude', 'longitude'])
    
    # Keep only necessary fields
    df_clean = df[keep_fields].copy()
    
    # Remove any rows with missing critical data
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['stationcode', 'numbikesavailable', 'capture_time'])
    removed_rows = initial_rows - len(df_clean)
    
    if removed_rows > 0:
        print(f"  ⚠️  Removed {removed_rows} rows with missing data")
    
    print(f"✅ Data cleaned: {len(df_clean):,} rows × {len(df_clean.columns)} columns\n")
    
    return df_clean


def add_temporal_features(df):
    """
    Add temporal features for ML training.
    
    Features added:
    - hour: Hour of day (0-23)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - is_weekend: Boolean (Saturday or Sunday)
    - is_rush_hour: Boolean (morning/evening commute hours)
    - part_of_day: Integer (0=night, 1=morning, 2=afternoon, 3=evening)
    - is_lunch_time: Boolean (lunch hours 12-13)
    - month: Month of year (1-12)
    - season: Season (1=winter, 2=spring, 3=summer, 4=autumn)
    
    Args:
        df: Input DataFrame with 'capture_time' column
        
    Returns:
        pd.DataFrame: DataFrame with added features
    """
    print("🔄 Step 6: Adding temporal features...")
    
    # Extract basic temporal features
    df['hour'] = df['capture_time'].dt.hour
    df['day_of_week'] = df['capture_time'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday=5, Sunday=6
    
    # Rush hour patterns (strong commuting signal)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19])
    
    # Part of day (behavioral patterns differ)
    # Using integer encoding: 0=night, 1=morning, 2=afternoon, 3=evening
    df['part_of_day'] = pd.cut(
        df['hour'], 
        bins=[0, 6, 12, 18, 24], 
        labels=[0, 1, 2, 3],  # Integer labels for neural network compatibility
        include_lowest=True
    ).astype(int)
    
    # Lunch time (midday bike usage spike)
    df['is_lunch_time'] = df['hour'].isin([12, 13])
    
    # Month and season (summer vs winter patterns)
    df['month'] = df['capture_time'].dt.month
    df['season'] = (df['month'] % 12 // 3 + 1)  # 1=winter, 2=spring, 3=summer, 4=autumn
    
    # Print summary statistics
    print(f"  ✅ Basic features: hour, day_of_week, is_weekend")
    print(f"     Hour range: {df['hour'].min()}-{df['hour'].max()}")
    print(f"     Weekend records: {df['is_weekend'].sum():,} ({df['is_weekend'].mean():.1%})")
    print(f"  ✅ Enhanced features: is_rush_hour, part_of_day, is_lunch_time, month, season")
    print(f"     Rush hour records: {df['is_rush_hour'].sum():,} ({df['is_rush_hour'].mean():.1%})")
    print(f"     Lunch time records: {df['is_lunch_time'].sum():,} ({df['is_lunch_time'].mean():.1%})")
    print(f"     Part of day distribution:")
    print(f"       {df['part_of_day'].value_counts().to_dict()}")
    print(f"     Season distribution:")
    print(f"       {df['season'].value_counts().sort_index().to_dict()}")
    print(f"✅ Feature engineering complete\n")
    
    return df


def save_to_parquet(df, output_path):
    """
    Save DataFrame to Parquet format.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the Parquet file
    """
    print(f"🔄 Step 7: Saving to Parquet...")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Parquet
    df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"✅ Saved to: {output_path}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Column list: {list(df.columns)}\n")


def print_data_summary(df):
    """
    Print summary statistics of the final dataset.
    
    Args:
        df: Final DataFrame
    """
    print("=" * 80)
    print("📊 FINAL DATASET SUMMARY")
    print("=" * 80)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"  Total records: {len(df):,}")
    print(f"  Unique stations: {df['stationcode'].nunique()}")
    print(f"  Time range: {(df['capture_time'].max() - df['capture_time'].min()).total_seconds() / 3600:.1f} hours")
    print(f"  Date range: {df['capture_time'].min().date()} → {df['capture_time'].max().date()}")
    
    print(f"\n🚲 Bike Availability:")
    print(f"  Mean bikes available: {df['numbikesavailable'].mean():.1f}")
    print(f"  Median bikes available: {df['numbikesavailable'].median():.1f}")
    print(f"  Max bikes available: {df['numbikesavailable'].max()}")
    
    print(f"\n🏙️ Station Info:")
    print(f"  Mean station capacity: {df['capacity'].mean():.1f}")
    print(f"  Total system capacity: {df['capacity'].sum() / df['stationcode'].nunique():.0f}")
    
    print(f"\n⏰ Temporal Distribution:")
    print(f"  Unique hours: {df['hour'].nunique()}")
    print(f"  Unique days: {df['day_of_week'].nunique()}")
    print(f"  Weekend proportion: {df['is_weekend'].mean():.1%}")
    
    print(f"\n✅ Data Quality:")
    print(f"  Missing values per column:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print(f"    → No missing values! ✅")
    else:
        for col, count in missing[missing > 0].items():
            print(f"    → {col}: {count} ({count/len(df):.1%})")
    
    print("\n" + "=" * 80)


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("🚀 BRONZE → SILVER DATA CONVERTER")
    print("=" * 80)
    print(f"Source: s3://{S3_BUCKET}/{S3_PREFIX}")
    print(f"Target: {OUTPUT_FILE}")
    print("=" * 80 + "\n")
    
    # Check AWS credentials
    if not check_aws_credentials():
        print("❌ ERROR: AWS credentials not found!")
        print("\nPlease configure AWS credentials using one of these methods:")
        print("\n1. Environment variables (recommended for this session):")
        print("   $env:AWS_ACCESS_KEY_ID='your-access-key'")
        print("   $env:AWS_SECRET_ACCESS_KEY='your-secret-key'")
        print("   $env:AWS_DEFAULT_REGION='eu-west-3'")
        print("\n2. AWS credentials file (~/.aws/credentials):")
        print("   Run: aws configure")
        print("\nThen run this script again.")
        sys.exit(1)
    
    try:
        # Step 1-3: Download from S3
        records = download_snapshots_from_s3()
        
        # Step 4: Create DataFrame
        df = create_dataframe(records)
        
        # Step 5: Clean data
        df_clean = clean_data(df)
        
        # Step 6: Add features
        df_features = add_temporal_features(df_clean)
        
        # Step 7: Save to Parquet
        save_to_parquet(df_features, OUTPUT_FILE)
        
        # Print summary
        print_data_summary(df_features)
        
        print("\n🎉 SUCCESS! Data is ready for ML training.")
        print(f"📂 Load with: pd.read_parquet('{OUTPUT_FILE}')")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
