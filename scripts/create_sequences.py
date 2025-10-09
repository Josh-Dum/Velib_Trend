"""
Create LSTM-compatible sequences from Silver Parquet data.

This script transforms tabular time series data into sequences for training
a Multi-Input LSTM model that predicts bike availability 1h, 2h, and 3h ahead.

Input: data/silver/velib_training_data.parquet (281,274 records)
Output: 
    - data/silver/sequences_train.npz (sequence features + targets)
    - data/silver/sequences_val.npz
    - data/silver/sequences_test.npz
    - data/silver/scaler.pkl (for inverse transform during inference)
    - data/silver/station_mappings.json (stationcode → numeric ID)

Architecture requirements:
    - Sequence length: 24 timesteps (24 hours of history)
    - Prediction horizon: 3 timesteps (T+1, T+2, T+3 hours ahead)
    - Static features: hour, day_of_week, is_weekend, capacity, station_id, lat, lon
    - Target: numbikesavailable (bikes available at T+1, T+2, T+3)

Split strategy (time-based to avoid data leakage):
    - Train: 70% (first 131 hours / 5.4 days)
    - Val: 15% (next 28 hours / 1.2 days)
    - Test: 15% (last 28 hours / 1.2 days)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import json
from datetime import datetime
from typing import Tuple, Dict

# Constants
SEQUENCE_LENGTH = 24  # 24 hours of history
PREDICTION_HORIZONS = [1, 2, 3]  # Predict 1h, 2h, 3h ahead
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
SILVER_DIR = DATA_DIR / "silver"
INPUT_FILE = SILVER_DIR / "velib_training_data.parquet"
OUTPUT_TRAIN = SILVER_DIR / "sequences_train.npz"
OUTPUT_VAL = SILVER_DIR / "sequences_val.npz"
OUTPUT_TEST = SILVER_DIR / "sequences_test.npz"
SCALER_FILE = SILVER_DIR / "scaler.pkl"
MAPPINGS_FILE = SILVER_DIR / "station_mappings.json"


def load_data() -> pd.DataFrame:
    """Load the Silver Parquet data."""
    print(f"📂 Loading data from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
    print(f"   Unique stations: {df['stationcode'].nunique()}")
    print(f"   Time range: {df['capture_time'].min()} to {df['capture_time'].max()}")
    return df


def create_station_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """
    Create a mapping from stationcode (string) to numeric ID.
    This allows us to use station_id as an embedding in the LSTM.
    """
    unique_stations = sorted(df['stationcode'].unique())
    mapping = {station: idx for idx, station in enumerate(unique_stations)}
    print(f"✅ Created station mapping for {len(mapping)} stations")
    return mapping


def prepare_data(df: pd.DataFrame, station_mapping: Dict[str, int]) -> pd.DataFrame:
    """
    Prepare data for sequence creation:
    1. Sort by station and time
    2. Add numeric station_id
    3. Handle missing values
    4. Convert datetime to proper format
    """
    print("🔧 Preparing data for sequence creation...")
    
    # Sort by station and time (crucial for sequences)
    df = df.sort_values(['stationcode', 'capture_time']).reset_index(drop=True)
    
    # Add numeric station_id
    df['station_id'] = df['stationcode'].map(station_mapping)
    
    # Convert capture_time to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['capture_time']):
        df['capture_time'] = pd.to_datetime(df['capture_time'])
    
    # Handle missing values (forward fill within each station)
    numeric_cols = ['numbikesavailable', 'numdocksavailable', 'capacity', 'latitude', 'longitude']
    for col in numeric_cols:
        df[col] = df.groupby('stationcode')[col].fillna(method='ffill')
        # If still NaN (first value), use backward fill
        df[col] = df.groupby('stationcode')[col].fillna(method='bfill')
    
    # Drop any remaining rows with NaN (should be very few)
    initial_len = len(df)
    df = df.dropna(subset=numeric_cols)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"⚠️  Dropped {dropped} rows with missing values ({dropped/initial_len*100:.2f}%)")
    
    print(f"✅ Data prepared: {len(df):,} rows")
    return df


def split_data_by_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by time (not by stations) to avoid data leakage.
    Train: first 70% of time
    Val: next 15% of time
    Test: last 15% of time
    """
    print("✂️  Splitting data by time...")
    
    # Get unique timestamps (sorted)
    unique_times = df['capture_time'].sort_values().unique()
    n_times = len(unique_times)
    
    # Calculate split points
    train_end_idx = int(n_times * TRAIN_RATIO)
    val_end_idx = int(n_times * (TRAIN_RATIO + VAL_RATIO))
    
    train_end_time = unique_times[train_end_idx]
    val_end_time = unique_times[val_end_idx]
    
    # Split dataframes
    df_train = df[df['capture_time'] < train_end_time].copy()
    df_val = df[(df['capture_time'] >= train_end_time) & (df['capture_time'] < val_end_time)].copy()
    df_test = df[df['capture_time'] >= val_end_time].copy()
    
    print(f"✅ Train: {len(df_train):,} rows ({df_train['capture_time'].min()} to {df_train['capture_time'].max()})")
    print(f"✅ Val:   {len(df_val):,} rows ({df_val['capture_time'].min()} to {df_val['capture_time'].max()})")
    print(f"✅ Test:  {len(df_test):,} rows ({df_test['capture_time'].min()} to {df_test['capture_time'].max()})")
    
    return df_train, df_val, df_test


def fit_scaler(df_train: pd.DataFrame) -> StandardScaler:
    """
    Fit StandardScaler on training data for the target variable (numbikesavailable).
    We normalize the target to help LSTM training converge faster.
    Static features will be normalized separately in the LSTM model.
    """
    print("📊 Fitting scaler on training data...")
    scaler = StandardScaler()
    scaler.fit(df_train[['numbikesavailable']])
    print(f"✅ Scaler fitted (mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f})")
    return scaler


def create_sequences_for_station(
    station_df: pd.DataFrame,
    scaler: StandardScaler,
    sequence_length: int = SEQUENCE_LENGTH,
    horizons: list = PREDICTION_HORIZONS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for a single station.
    
    Returns:
        X_seq: (n_samples, sequence_length) - normalized bikes_available sequence
        X_static: (n_samples, n_static_features) - static features at prediction time
        y: (n_samples, len(horizons)) - target bikes_available at T+1, T+2, T+3
    """
    # Sort by time (should already be sorted, but double-check)
    station_df = station_df.sort_values('capture_time')
    
    # Extract target variable (bikes available)
    bikes_available = station_df['numbikesavailable'].values
    
    # Normalize target
    bikes_normalized = scaler.transform(bikes_available.reshape(-1, 1)).flatten()
    
    # Extract static features (at each timestep)
    static_features = station_df[['hour', 'day_of_week', 'is_weekend', 'capacity', 
                                   'station_id', 'latitude', 'longitude']].values
    
    # Create sequences
    X_seq_list = []
    X_static_list = []
    y_list = []
    
    max_horizon = max(horizons)
    n_samples = len(bikes_normalized)
    
    for i in range(sequence_length, n_samples - max_horizon):
        # Input sequence: past 24 hours of bikes_available (normalized)
        seq = bikes_normalized[i - sequence_length:i]
        
        # Static features at prediction time (timestep i)
        static = static_features[i]
        
        # Targets: bikes_available at T+1, T+2, T+3 (original scale, not normalized)
        targets = [bikes_available[i + h] for h in horizons]
        
        X_seq_list.append(seq)
        X_static_list.append(static)
        y_list.append(targets)
    
    if len(X_seq_list) == 0:
        return None, None, None
    
    return (
        np.array(X_seq_list),      # (n_samples, sequence_length)
        np.array(X_static_list),   # (n_samples, n_static_features)
        np.array(y_list)           # (n_samples, len(horizons))
    )


def create_sequences_for_split(
    df: pd.DataFrame,
    scaler: StandardScaler,
    split_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for all stations in a data split (train/val/test).
    """
    print(f"🔄 Creating sequences for {split_name} split...")
    
    X_seq_all = []
    X_static_all = []
    y_all = []
    
    stations = df['stationcode'].unique()
    n_stations = len(stations)
    skipped = 0
    
    for idx, station in enumerate(stations):
        if (idx + 1) % 100 == 0:
            print(f"   Processing station {idx + 1}/{n_stations}...")
        
        station_df = df[df['stationcode'] == station]
        
        # Skip stations with insufficient data
        if len(station_df) < SEQUENCE_LENGTH + max(PREDICTION_HORIZONS):
            skipped += 1
            continue
        
        X_seq, X_static, y = create_sequences_for_station(station_df, scaler)
        
        if X_seq is not None:
            X_seq_all.append(X_seq)
            X_static_all.append(X_static)
            y_all.append(y)
    
    if skipped > 0:
        print(f"⚠️  Skipped {skipped}/{n_stations} stations due to insufficient data")
    
    # Concatenate all sequences
    X_seq_final = np.concatenate(X_seq_all, axis=0)
    X_static_final = np.concatenate(X_static_all, axis=0)
    y_final = np.concatenate(y_all, axis=0)
    
    print(f"✅ {split_name} sequences created:")
    print(f"   X_seq shape: {X_seq_final.shape} (samples, sequence_length)")
    print(f"   X_static shape: {X_static_final.shape} (samples, static_features)")
    print(f"   y shape: {y_final.shape} (samples, horizons)")
    
    return X_seq_final, X_static_final, y_final


def save_sequences(
    X_seq: np.ndarray,
    X_static: np.ndarray,
    y: np.ndarray,
    output_path: Path
):
    """Save sequences to compressed numpy file."""
    np.savez_compressed(
        output_path,
        X_seq=X_seq,
        X_static=X_static,
        y=y
    )
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved to {output_path} ({file_size_mb:.2f} MB)")


def save_scaler(scaler: StandardScaler):
    """Save scaler for later use during inference."""
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Saved scaler to {SCALER_FILE}")


def save_station_mappings(mapping: Dict[str, int]):
    """Save station mappings for later use."""
    with open(MAPPINGS_FILE, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ Saved station mappings to {MAPPINGS_FILE}")


def print_summary(
    X_seq_train: np.ndarray,
    X_seq_val: np.ndarray,
    X_seq_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray
):
    """Print final summary statistics."""
    print("\n" + "="*80)
    print("📊 SEQUENCE CREATION SUMMARY")
    print("="*80)
    print(f"Total sequences created: {len(X_seq_train) + len(X_seq_val) + len(X_seq_test):,}")
    print(f"  - Train: {len(X_seq_train):,} sequences ({len(X_seq_train)/(len(X_seq_train)+len(X_seq_val)+len(X_seq_test))*100:.1f}%)")
    print(f"  - Val:   {len(X_seq_val):,} sequences ({len(X_seq_val)/(len(X_seq_train)+len(X_seq_val)+len(X_seq_test))*100:.1f}%)")
    print(f"  - Test:  {len(X_seq_test):,} sequences ({len(X_seq_test)/(len(X_seq_train)+len(X_seq_val)+len(X_seq_test))*100:.1f}%)")
    print()
    print(f"Target statistics (bikes_available):")
    print(f"  - Train mean: {y_train.mean():.2f} ± {y_train.std():.2f}")
    print(f"  - Train range: [{y_train.min():.0f}, {y_train.max():.0f}]")
    print(f"  - Val mean: {y_val.mean():.2f} ± {y_val.std():.2f}")
    print(f"  - Test mean: {y_test.mean():.2f} ± {y_test.std():.2f}")
    print()
    print("📁 Output files:")
    print(f"  - {OUTPUT_TRAIN}")
    print(f"  - {OUTPUT_VAL}")
    print(f"  - {OUTPUT_TEST}")
    print(f"  - {SCALER_FILE}")
    print(f"  - {MAPPINGS_FILE}")
    print()
    print("🎉 Sequences are ready for LSTM training!")
    print("="*80)


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("🚀 LSTM SEQUENCE CREATION PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create output directory
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Create station mapping
    station_mapping = create_station_mapping(df)
    save_station_mappings(station_mapping)
    
    # Step 3: Prepare data
    df = prepare_data(df, station_mapping)
    
    # Step 4: Split data by time
    df_train, df_val, df_test = split_data_by_time(df)
    
    # Step 5: Fit scaler on training data
    scaler = fit_scaler(df_train)
    save_scaler(scaler)
    
    # Step 6: Create sequences for each split
    X_seq_train, X_static_train, y_train = create_sequences_for_split(df_train, scaler, "TRAIN")
    X_seq_val, X_static_val, y_val = create_sequences_for_split(df_val, scaler, "VAL")
    X_seq_test, X_static_test, y_test = create_sequences_for_split(df_test, scaler, "TEST")
    
    # Step 7: Save sequences
    save_sequences(X_seq_train, X_static_train, y_train, OUTPUT_TRAIN)
    save_sequences(X_seq_val, X_static_val, y_val, OUTPUT_VAL)
    save_sequences(X_seq_test, X_static_test, y_test, OUTPUT_TEST)
    
    # Step 8: Print summary
    print_summary(X_seq_train, X_seq_val, X_seq_test, y_train, y_val, y_test)
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
