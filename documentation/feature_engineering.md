# Feature Engineering for LSTM Model

## Overview
This document describes the feature engineering strategy implemented for the Vélib' availability prediction model, including normalization approaches for different feature types.

## Feature Categories

### 1. Continuous Features (9 total) - NORMALIZED
These features are normalized using StandardScaler (mean=0, std=1):

| Feature | Range | Description | Why Normalize |
|---------|-------|-------------|---------------|
| `hour` | 0-23 | Hour of day | Different scale from other features |
| `day_of_week` | 0-6 | Day of week (0=Monday) | Sequential encoding needs scaling |
| `capacity` | 20-70 | Station capacity | Large variance across stations |
| `station_id` | 1-1498 | Numeric station identifier | Very large range |
| `latitude` | 48.8-48.9 | GPS latitude | Small range but consistent scaling needed |
| `longitude` | 2.2-2.4 | GPS longitude | Small range but consistent scaling needed |
| `part_of_day` | 0-3 | Time period (night/morning/afternoon/evening) | Ordinal encoding |
| `month` | 1-12 | Month of year | Seasonal patterns |
| `season` | 0-3 | Season (0=winter, 3=autumn) | Quarterly patterns |

**Why normalize?**
- Neural networks train faster with normalized inputs (mean≈0, std≈1)
- Prevents features with large values (like `station_id`) from dominating learning
- Ensures all continuous features contribute equally to gradient updates

### 2. Binary Features (3 total) - NOT NORMALIZED
These features are kept as-is (values: 0 or 1):

| Feature | Values | Description | Why NOT Normalize |
|---------|--------|-------------|-------------------|
| `is_weekend` | 0/1 | Weekend flag | Already binary, normalization would break semantic meaning |
| `is_rush_hour` | 0/1 | Rush hour flag (7-9h, 17-19h) | Binary flags work well as-is for neural networks |
| `is_lunch_time` | 0/1 | Lunch time flag (12-14h) | 0/1 values are already on consistent scale |

**Why NOT normalize?**
- Binary values (0/1) already have consistent scale
- Normalization would transform them to meaningless values (e.g., -1.2 or 0.7)
- Neural networks interpret 0/1 as "switches" or "indicators" - this is intuitive and effective
- StandardScaler expects continuous distributions, not binary data

## Temporal Feature Engineering Details

### Enhanced Temporal Features (Added in Phase 1)

#### 1. `is_rush_hour` (Binary)
```python
is_rush_hour = ((hour >= 7) & (hour < 9)) | ((hour >= 17) & (hour < 19))
```
- Captures morning (7-9h) and evening (17-19h) rush hours
- Critical for bike-sharing: demand spikes during commute times

#### 2. `part_of_day` (Ordinal: 0-3)
```python
0 = Night   (0-6h)
1 = Morning (6-12h)
2 = Afternoon (12-18h)
3 = Evening (18-24h)
```
- Represents broader time periods than raw hour
- Helps model learn different demand patterns throughout the day

#### 3. `is_lunch_time` (Binary)
```python
is_lunch_time = (hour >= 12) & (hour < 14)
```
- Captures lunch period when bike usage changes
- Paris lunch culture: many people bike to restaurants

#### 4. `month` (Numeric: 1-12)
- Seasonal variations in bike usage
- Weather impact: summer vs. winter usage patterns
- Currently constant (October only) but will be useful with more data

#### 5. `season` (Ordinal: 0-3)
```python
0 = Winter  (Dec-Feb)
1 = Spring  (Mar-May)
2 = Summer  (Jun-Aug)
3 = Autumn  (Sep-Nov)
```
- Quarterly patterns: tourism, weather, daylight hours
- Currently constant (autumn only) but will be useful with more data

## Normalization Implementation

### Training Phase
1. **Fit scalers on training data only** (to prevent data leakage):
   - `scaler.pkl`: For target variable (`numbikesavailable`)
   - `static_scaler.pkl`: For 9 continuous features only

2. **Apply scalers to all splits** (train/val/test):
   - Continuous features → `static_scaler.transform()`
   - Binary features → kept as-is
   - Target → `scaler.transform()`

3. **Feature concatenation**:
   ```python
   # Final X_static shape: (n_samples, 12)
   X_static = [continuous_normalized (9 features), binary_raw (3 features)]
   ```

### Inference Phase
1. Load both scalers from disk
2. Normalize continuous features with `static_scaler`
3. Keep binary features as 0/1
4. Concatenate in same order as training

## Validation Results

From `scripts/verify_sequences.py` (Oct 29, 2025):

### Continuous Features (Positions 0-8)
✅ All have mean ≈ 0, std ≈ 1:
- hour: mean=0.009, std=0.996
- capacity: mean=0.001, std=0.999
- station_id: mean=-0.000, std=1.000

### Binary Features (Positions 9-11)
✅ All contain only 0 or 1:
- is_weekend: [0, 1], mean=0.327 (32.7% weekend samples)
- is_rush_hour: [0, 1], mean=0.252 (25.2% rush hour samples)
- is_lunch_time: [0, 1], mean=0.081 (8.1% lunch time samples)

## Files Modified

1. **`scripts/bronze_to_silver.py`**:
   - `add_temporal_features()`: Generates all 5 new temporal features
   - Integer encoding for `part_of_day` (0-3)

2. **`scripts/create_sequences.py`**:
   - `fit_static_scaler()`: Fits StandardScaler on 9 continuous features only
   - `create_sequences_for_station()`: Separates continuous/binary features
   - Saves `static_scaler.pkl` for inference

3. **`scripts/verify_sequences.py`**:
   - Validation script to verify normalization correctness
   - Checks for binary-only values in positions 9-11

## Performance Impact

**Expected improvements from enhanced features:**
- 5-15% MAE reduction based on Kaggle research
- Better rush hour predictions (critical business hours)
- Improved weekend vs. weekday patterns
- More accurate lunch period forecasts

## Next Steps

- [ ] Train Model v5 with normalized features
- [ ] Compare performance vs. Model v4
- [ ] Deploy best model to SageMaker
- [ ] Add weather features when API available
- [ ] Add holiday/special event flags
