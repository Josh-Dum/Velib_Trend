# Model Improvement Strategies - Research Report

**Date**: October 29, 2025  
**Current Model Performance**: MAE 2.79 bikes (T+1h), R² 0.859  
**Target**: Reduce MAE below 2.0 bikes (~30% improvement)

---

## 📊 Current Model Analysis

### Performance Metrics (Model v4)
- **T+1h**: MAE = 2.79 bikes, R² = 0.859 (86% variance explained)
- **T+2h**: MAE = 3.09 bikes, R² = 0.828 (83% variance explained)
- **T+3h**: MAE = 3.51 bikes, R² = 0.788 (79% variance explained)

### Current Features
Our model currently uses:
- ✅ **Temporal**: `hour`, `day_of_week`, `is_weekend`
- ✅ **Station-specific**: `capacity`
- ✅ **Historical**: 24-hour sequence of `num_bikes_available`
- ❌ **Missing**: Weather, holidays, events, advanced temporal patterns

### Error Analysis
- **Average station availability**: ~12 bikes
- **Average error**: ~2.8 bikes
- **Error rate**: ~23-30% (decent but improvable)
- **Bias**: Very low (-0.199 to +0.262 bikes) → model is well-calibrated

---

## 🔬 Kaggle Competition Insights

Based on analysis of winning solutions from major bike-sharing competitions:
- [Bike Sharing Demand Competition](https://www.kaggle.com/competitions/bike-sharing-demand)
- [UCI Bike Rental Data Set](https://www.kaggle.com/code/melikedilekci/uci-bike-rental-data-set)
- [Bike Rental Count Prediction](https://www.kaggle.com/code/lakshmi25npathi/bike-rental-count-prediction-using-python)

### Top-Ranked Features (by impact)

| Feature Category | Specific Features | Impact | Currently Used |
|------------------|-------------------|--------|----------------|
| **Date & Time** | Hour of day, day of week, month, season | 🔥🔥🔥 High | ✅ Partial |
| **Weather** | Temperature (real, feels-like), rain, humidity, wind | 🔥🔥🔥 High | ❌ No |
| **Calendar** | Public holidays, school vacations, working day | 🔥🔥 Medium-High | ❌ No |
| **Location** | Station ID, neighborhood type, proximity to transit | 🔥🔥 Medium-High | ✅ Partial |
| **Lag Features** | Previous hour/day availability, recent trends | 🔥🔥 Medium | ✅ Via sequences |
| **Special Events** | Festivals, protests, sports events | 🔥 Medium | ❌ No |

---

## 🎯 Proposed Improvements (Prioritized)

### Priority 1: Quick Wins (No New Data Required)

#### 1.1 Enhanced Temporal Features
**Effort**: Low (30 minutes)  

Add these derived features to `bronze_to_silver.py`:
```python
# Rush hour patterns (strong commuting signal)
is_rush_hour = hour.isin([7, 8, 9, 17, 18, 19])

# Part of day (behavioral patterns differ)
part_of_day = pd.cut(hour, bins=[0, 6, 12, 18, 24], 
                     labels=['night', 'morning', 'afternoon', 'evening'])

# Lunch time (midday bike usage spike)
is_lunch_time = hour.isin([12, 13])

# Month and season (summer vs winter patterns)
month = timestamp.dt.month
season = (month % 12 // 3 + 1)  # 1=winter, 2=spring, 3=summer, 4=autumn
```

**Why this works**: Kaggle winners consistently show that bike usage has distinct patterns for:
- Morning commute (7-9am): Residential → Business districts
- Evening commute (5-7pm): Business → Residential  
- Lunch hour (12-2pm): Short trips for meals
- Summer vs Winter: 30-50% usage difference

---

#### 1.2 Switch to Huber Loss
**Effort**: Very Low (5 minutes)  
**Expected Impact**: 3-8% MAE reduction

Replace MSE with Huber Loss in `train_lstm.py`:
```python
import torch.nn as nn

# Current: criterion = nn.MSELoss()
criterion = nn.HuberLoss(delta=1.0)  # More robust to outliers
```

**Why this works**: Huber loss is less sensitive to outliers (e.g., empty or full stations), which can distort MSE-based training.

---

#### 1.3 Bidirectional LSTM
**Effort**: Low (15 minutes)  
**Expected Impact**: 5-15% MAE reduction

Modify LSTM layer in `train_lstm.py` to look at patterns bidirectionally:
```python
# Change from:
self.lstm1 = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)

# To:
self.lstm1 = nn.LSTM(input_size=1, hidden_size=128, 
                     batch_first=True, bidirectional=True)

# Adjust next layer input size (128 → 256 because bidirectional doubles output)
self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, batch_first=True)
```

**Why this works**: Bidirectional LSTMs can learn patterns from both past and future context in the 24-hour sequence, improving pattern recognition.

---

### Priority 2: Weather Integration (High Impact)

#### 2.1 Historical Weather Data
**Effort**: Medium (3-4 hours)  
**Expected Impact**: 10-20% MAE reduction  
**Data Source**: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (FREE)

**Key weather features from Kaggle winners**:
- ✅ **Temperature** (°C): Strong positive correlation (15-25°C = peak usage)
- ✅ **Precipitation** (mm): Strong negative effect (rain → -30-50% usage)
- ✅ **Wind Speed** (km/h): Negative effect (>20km/h → -15% usage)
- ✅ **Humidity** (%): Moderate negative effect (>80% = discomfort)
- ✅ **Weather Code**: Clear/Cloudy/Rain/Snow (categorical impact)

**Implementation plan**:
1. Create `scripts/fetch_weather_data.py` to backfill Oct 1-28
2. Merge weather data with snapshots in `bronze_to_silver.py`
3. Add weather features to LSTM static inputs

**API Call Example**:
```python
import requests

# Paris coordinates
latitude, longitude = 48.8566, 2.3522

# Fetch historical weather (Oct 1-28, 2025)
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2025-10-01",
    "end_date": "2025-10-28",
    "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"],
    "timezone": "Europe/Paris"
}
response = requests.get(url, params=params)
weather_data = response.json()
```

**Expected behavior changes**:
- **Rain**: -30 to -50% bike availability (people avoid cycling)
- **Temperature**: Optimal 15-25°C, drops outside this range
- **Wind >20km/h**: -10 to -20% usage (headwinds, safety concerns)

---

#### 2.2 Live Weather Integration
**Effort**: Low (30 minutes)  
**Required**: Update Lambda function to include weather

Add current weather to snapshot collection for real-time predictions.

---

### Priority 3: Calendar Features (Medium Impact)

#### 3.1 French Public Holidays
**Effort**: Low (1 hour)  
**Expected Impact**: 3-5% MAE reduction

Add holiday detection in `bronze_to_silver.py`:
```python
import holidays

# French public holidays
fr_holidays = holidays.France(years=[2025])

df['is_public_holiday'] = df['timestamp'].dt.date.isin(fr_holidays)
df['is_school_vacation'] = df['timestamp'].apply(is_school_vacation)  # Custom function
```

**Why this works**: Kaggle winners show:
- **Public holidays**: -15 to -25% commuting usage
- **School vacations**: Different patterns (more leisure trips, fewer commutes)

---

#### 3.2 Working Day Classification
**Effort**: Very Low (15 minutes)  
**Expected Impact**: 2-3% MAE reduction

```python
# Is it a working day? (not weekend AND not public holiday)
df['is_working_day'] = ~df['is_weekend'] & ~df['is_public_holiday']
```

---

### Priority 4: Advanced Techniques (High Effort, High Reward)

#### 4.1 Station Clustering
**Effort**: High (6-8 hours)  
**Expected Impact**: 15-25% MAE reduction

**Concept**: Not all stations behave the same. Train specialized models per cluster.

**Cluster types** (identified in Kaggle competitions):
1. **Residential Clusters**: Morning departures (7-9am), evening arrivals (6-8pm)
2. **Business Districts**: Opposite pattern (morning arrivals, evening departures)
3. **Tourist Spots**: Weekend-heavy, seasonal peaks
4. **Transit Hubs**: High volume, balanced in/out flow

**Implementation**:
```python
from sklearn.cluster import KMeans

# Feature engineering for clustering
station_features = df.groupby('station_code').agg({
    'num_bikes_available': ['mean', 'std'],
    'hour_7_9_ratio': 'mean',   # Morning rush activity
    'hour_17_19_ratio': 'mean', # Evening rush activity
    'weekend_ratio': 'mean',    # Weekend vs weekday usage
    'latitude': 'first',
    'longitude': 'first'
})

# Cluster stations
kmeans = KMeans(n_clusters=4, random_state=42)
station_clusters = kmeans.fit_predict(station_features)

# Train one model per cluster
for cluster_id in range(4):
    cluster_stations = stations[station_clusters == cluster_id]
    # Train specialized model...
```

---

#### 4.2 Attention Mechanism
**Effort**: High (8-10 hours)  
**Expected Impact**: 8-15% MAE reduction

Add attention layer to focus on important time steps in 24-hour sequence.

```python
class AttentionLSTM(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.lstm = nn.LSTM(...)
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)
        return context
```

**Why this works**: Some hours are more predictive than others (e.g., T-1h is more important than T-23h).

---

#### 4.3 Ensemble Models
**Effort**: Medium (4-6 hours)  
**Expected Impact**: 5-10% MAE reduction

Train multiple models and average predictions:
```python
# Train 5 models with different random seeds
models = []
for seed in [42, 123, 456, 789, 1011]:
    torch.manual_seed(seed)
    model = train_model(seed=seed)
    models.append(model)

# Ensemble prediction
predictions = [model.predict(x) for model in models]
final_prediction = np.mean(predictions, axis=0)
```

**Why this works**: Reduces variance, more robust predictions.

---

## 📈 Implementation Roadmap

### Phase 1: Quick Wins (Today - 2 hours)
- ✅ Add `is_rush_hour`, `part_of_day`, `is_lunch_time`, `month`, `season`
- ✅ Switch to Huber Loss
- ✅ Implement Bidirectional LSTM
- **Expected**: MAE ~2.4-2.5 bikes (13-17% improvement)

### Phase 2: Weather Integration (This week - 4 hours)
- ✅ Fetch historical weather data (Open-Meteo API)
- ✅ Merge with training data
- ✅ Add weather features to model inputs
- **Expected**: MAE ~2.0-2.2 bikes (25-30% improvement)

### Phase 3: Calendar Features (This week - 2 hours)
- ✅ Add French public holidays
- ✅ Add school vacation detection
- ✅ Add `is_working_day` feature
- **Expected**: MAE ~1.9-2.1 bikes (28-35% improvement)

### Phase 4: Advanced Techniques (Next 2 weeks)
- ⏸️ Station clustering analysis
- ⏸️ Attention mechanism implementation
- ⏸️ Ensemble model setup
- **Expected**: MAE ~1.7-1.9 bikes (35-45% improvement)

---

## 🎯 Success Criteria

| Milestone | Target MAE | Expected Timeline | Status |
|-----------|------------|-------------------|--------|
| Current Model v4 | 2.79 bikes | - | ✅ Complete |
| Quick Wins | 2.4-2.5 bikes | 1 day | ⏸️ Planned |
| + Weather | 2.0-2.2 bikes | 1 week | ⏸️ Planned |
| + Calendar | 1.9-2.1 bikes | 1 week | ⏸️ Planned |
| + Advanced | 1.7-1.9 bikes | 2 weeks | ⏸️ Future |

---

## 📚 Key Learnings from Kaggle

### Most Impactful Features (Ranked)
1. **Hour of day** + **Day of week** → Captures commuting patterns
2. **Temperature** + **Weather conditions** → Comfort factor
3. **Rain/Precipitation** → Strong deterrent (-30-50% usage)
4. **Public holidays** + **Working day** → Changes behavior patterns
5. **Station location/type** → Different neighborhoods = different patterns
6. **Historical lag features** → Recent trends are predictive

### Common Mistakes to Avoid
- ❌ Treating all stations identically (clustering helps!)
- ❌ Ignoring weather (single biggest external factor)
- ❌ Using only MSE loss (Huber/MAE losses often better)
- ❌ Not engineering temporal features (hour alone isn't enough)
- ❌ Overlooking holidays (behavior changes dramatically)

### Winning Strategies
- ✅ Deep feature engineering on datetime
- ✅ Weather integration (especially rain + temperature)
- ✅ Station clustering or embeddings
- ✅ Ensemble multiple models
- ✅ Careful train/val/test splitting (respect temporal order)

---

## 🔗 References

1. [Bike Sharing Demand - Kaggle Competition](https://www.kaggle.com/competitions/bike-sharing-demand)
2. [Bike Rentals by Time and Temperature](https://www.kaggle.com/code/benhamner/bike-rentals-by-time-and-temperature)
3. [Bike Rental Count Prediction Using Python](https://www.kaggle.com/code/lakshmi25npathi/bike-rental-count-prediction-using-python)
4. [Bike Sharing Multiple Linear Regression](https://www.kaggle.com/code/gauravduttakiit/bike-sharing-multiple-linear-regression)
5. [Bike Sharing Demand - RMSLE 0.3194](https://www.kaggle.com/code/rajmehra03/bike-sharing-demand-rmsle-0-3194)
6. [Bike Rental Predictions using LR, RF, GBR](https://www.kaggle.com/code/yaroshevskiy/bike-rental-predictions-using-lr-rf-gbr)
7. [UCI Bike Rental Data Set](https://www.kaggle.com/code/melikedilekci/uci-bike-rental-data-set)
8. [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)

---

**Last Updated**: October 29, 2025  
**Next Review**: After Phase 1 implementation
