# Journey Planner Feature - Implementation Plan

## 📋 Overview

**Feature Name:** Journey Planner  
**Purpose:** Help users plan bike trips by automatically finding the best stations and predicting availability  
**User Story:** *"I want to go from address A to address B. Will I find a bike at the start and a dock at the end?"*

---

## 🎯 Core Functionality

### User Inputs
1. **Start Location** - Address or location (e.g., "24 Rue de Rivoli, Paris")
2. **Destination** - Target address (e.g., "Gare du Nord, Paris")

### Automated Process
1. **Geocode** addresses to lat/lon coordinates
2. **Find nearest station** to start location
3. **Find nearest station** to destination
4. **Calculate route**:
   - Walking distance to start station
   - Biking distance between stations
   - Walking distance from end station to destination
5. **Estimate times** for each segment
6. **Get ML predictions** for both stations at calculated arrival times
7. **Provide verdict** - Is this route feasible?

### Output
- **Route visualization** on map
- **Time breakdown** (walk + bike + walk)
- **Availability prediction** (bikes at start, docks at end)
- **Clear recommendation** (Go / Caution / Risk)

---

## 🏗️ Architecture

### Frontend (Streamlit)
```
Sidebar:
├─ 🗺️ Explore Map (existing)
└─ 🚴 Plan Journey (new tab)

Journey Planner Tab:
├─ Input Form
│  ├─ Start address (text input)
│  └─ Destination address (text input)
├─ Calculate Button
└─ Results Section
   ├─ Route Map (with line overlay)
   ├─ Time Breakdown (metrics)
   ├─ Station Info Cards
   ├─ Availability Predictions
   └─ Verdict (success/warning/error)
```

### Backend Components
1. **Geocoding Service** - Convert address → coordinates
2. **Station Finder** - Find nearest station to coordinates
3. **Distance Calculator** - Haversine distance between points
4. **Route Planner** - Calculate complete route with times
5. **Prediction Service** - Get ML predictions for both stations
6. **Verdict Engine** - Analyze predictions and provide recommendation

---

## 🔧 Technical Implementation

### 1. Geocoding Function

**Service:** Nominatim (OpenStreetMap) - Free, no API key required  
**Rate Limit:** 1 request/second (add delay if needed)

```python
def geocode_address(address: str) -> tuple[float, float]:
    """
    Convert address to (latitude, longitude) using Nominatim.
    
    Args:
        address: Address string (e.g., "24 Rue de Rivoli, Paris")
    
    Returns:
        (lat, lon) tuple or (None, None) if not found
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
        "countrycodes": "fr",  # Limit to France for faster results
        "addressdetails": 1
    }
    headers = {
        "User-Agent": "VelibTrend/1.0 (Educational Project)"
    }
    
    response = requests.get(url, params=params, headers=headers, timeout=10)
    data = response.json()
    
    if data and len(data) > 0:
        return float(data[0]['lat']), float(data[0]['lon'])
    
    return None, None
```

**Error Handling:**
- Invalid address → Show error message
- Network timeout → Fallback message
- Rate limit → Add 1-second delay between requests

---

### 2. Haversine Distance Calculation

**Purpose:** Calculate distance between two GPS coordinates  
**Formula:** Great-circle distance (spherical Earth approximation)

```python
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS coordinates using Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in kilometers
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance
```

---

### 3. Find Nearest Station

```python
def find_nearest_station(lat: float, lon: float, stations_df: pd.DataFrame) -> dict:
    """
    Find the closest Vélib station to given coordinates.
    
    Args:
        lat, lon: Target coordinates
        stations_df: DataFrame with all stations (must have 'lat', 'lon' columns)
    
    Returns:
        Dictionary with station info + distance
    """
    # Calculate distance to all stations
    stations_df = stations_df.copy()
    stations_df['distance_km'] = stations_df.apply(
        lambda row: haversine_distance(lat, lon, row['lat'], row['lon']),
        axis=1
    )
    
    # Find nearest
    nearest_idx = stations_df['distance_km'].idxmin()
    nearest = stations_df.loc[nearest_idx].to_dict()
    
    return nearest
```

---

### 4. Route Planning

**Constants:**
```python
WALKING_SPEED_KMH = 5.0  # Average walking speed
BIKING_SPEED_KMH = 15.0  # Average biking speed in city
```

**Route Calculation:**
```python
def plan_route(start_lat: float, start_lon: float, 
               dest_lat: float, dest_lon: float,
               stations_df: pd.DataFrame) -> dict:
    """
    Plan complete journey from start to destination.
    
    Returns:
        {
            'start_station': {...},
            'end_station': {...},
            'walk_to_start_km': float,
            'walk_to_start_min': float,
            'bike_distance_km': float,
            'bike_time_min': float,
            'walk_from_end_km': float,
            'walk_from_end_min': float,
            'total_time_min': float,
            'arrival_at_start_min': float,  # When user arrives at start station
            'arrival_at_end_min': float     # When user arrives at end station
        }
    """
    # Find nearest stations
    start_station = find_nearest_station(start_lat, start_lon, stations_df)
    end_station = find_nearest_station(dest_lat, dest_lon, stations_df)
    
    # Calculate distances
    walk_to_start_km = start_station['distance_km']
    bike_distance_km = haversine_distance(
        start_station['lat'], start_station['lon'],
        end_station['lat'], end_station['lon']
    )
    walk_from_end_km = haversine_distance(
        end_station['lat'], end_station['lon'],
        dest_lat, dest_lon
    )
    
    # Calculate times
    walk_to_start_min = (walk_to_start_km / WALKING_SPEED_KMH) * 60
    bike_time_min = (bike_distance_km / BIKING_SPEED_KMH) * 60
    walk_from_end_min = (walk_from_end_km / WALKING_SPEED_KMH) * 60
    total_time_min = walk_to_start_min + bike_time_min + walk_from_end_min
    
    # Arrival times (from now)
    arrival_at_start_min = walk_to_start_min
    arrival_at_end_min = walk_to_start_min + bike_time_min
    
    return {
        'start_station': start_station,
        'end_station': end_station,
        'walk_to_start_km': walk_to_start_km,
        'walk_to_start_min': walk_to_start_min,
        'bike_distance_km': bike_distance_km,
        'bike_time_min': bike_time_min,
        'walk_from_end_km': walk_from_end_km,
        'walk_from_end_min': walk_from_end_min,
        'total_time_min': total_time_min,
        'arrival_at_start_min': arrival_at_start_min,
        'arrival_at_end_min': arrival_at_end_min
    }
```

---

### 5. Get Predictions at Specific Time

**Current API:** Returns predictions at T+1h, T+2h, T+3h  
**Need:** Prediction at arbitrary time (e.g., T+17min)

**Solution:** Interpolation between prediction points

```python
def get_prediction_at_time(station_code: str, minutes_from_now: float, api_base_url: str) -> dict:
    """
    Get prediction for a station at a specific time offset.
    Uses interpolation if time is between prediction points.
    
    Args:
        station_code: Station code
        minutes_from_now: Time offset in minutes from current time
        api_base_url: FastAPI base URL
    
    Returns:
        {
            'bikes_predicted': float,
            'docks_predicted': float,
            'confidence': str  # 'high', 'medium', 'low'
        }
    """
    import numpy as np
    
    # Get predictions from API
    response = requests.get(f"{api_base_url}/predict/{station_code}")
    data = response.json()
    
    # Current availability
    current_bikes = data['current']['bikes']
    current_docks = data['current']['docks']
    
    # Prediction points (in minutes)
    pred_times = [0, 60, 120, 180]  # T+0, T+1h, T+2h, T+3h
    pred_bikes = [
        current_bikes,
        data['predictions']['t_plus_1h']['bikes'],
        data['predictions']['t_plus_2h']['bikes'],
        data['predictions']['t_plus_3h']['bikes']
    ]
    pred_docks = [
        current_docks,
        data['predictions']['t_plus_1h']['docks'],
        data['predictions']['t_plus_2h']['docks'],
        data['predictions']['t_plus_3h']['docks']
    ]
    
    # Interpolate
    bikes_predicted = np.interp(minutes_from_now, pred_times, pred_bikes)
    docks_predicted = np.interp(minutes_from_now, pred_times, pred_docks)
    
    # Determine confidence (closer to current = higher confidence)
    if minutes_from_now <= 30:
        confidence = 'high'
    elif minutes_from_now <= 90:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return {
        'bikes_predicted': round(bikes_predicted, 1),
        'docks_predicted': round(docks_predicted, 1),
        'confidence': confidence
    }
```

---

### 6. Verdict Logic

**Thresholds:**
```python
BIKE_THRESHOLD_SAFE = 4    # >= 4 bikes = safe
BIKE_THRESHOLD_MIN = 2     # >= 2 bikes = possible
DOCK_THRESHOLD_SAFE = 4    # >= 4 docks = safe
DOCK_THRESHOLD_MIN = 2     # >= 2 docks = possible
```

**Decision Tree:**
```python
def get_journey_verdict(bikes_predicted: float, docks_predicted: float) -> dict:
    """
    Analyze predictions and provide recommendation.
    
    Returns:
        {
            'status': str,  # 'success', 'warning', 'error'
            'verdict': str,  # User-friendly message
            'icon': str,     # Emoji icon
            'details': str   # Additional explanation
        }
    """
    if bikes_predicted >= BIKE_THRESHOLD_SAFE and docks_predicted >= DOCK_THRESHOLD_SAFE:
        return {
            'status': 'success',
            'verdict': 'Perfect! Good to go 🎉',
            'icon': '✅',
            'details': f'{bikes_predicted:.0f} bikes and {docks_predicted:.0f} docks should be available.'
        }
    
    elif bikes_predicted >= BIKE_THRESHOLD_MIN and docks_predicted >= DOCK_THRESHOLD_MIN:
        issues = []
        if bikes_predicted < BIKE_THRESHOLD_SAFE:
            issues.append(f'only {bikes_predicted:.0f} bike(s) at start')
        if docks_predicted < DOCK_THRESHOLD_SAFE:
            issues.append(f'only {docks_predicted:.0f} dock(s) at end')
        
        return {
            'status': 'warning',
            'verdict': 'Possible, but tight ⚠️',
            'icon': '⚠️',
            'details': f'Note: {" and ".join(issues)}. Consider having a backup plan.'
        }
    
    else:
        problems = []
        if bikes_predicted < BIKE_THRESHOLD_MIN:
            problems.append('very few bikes at start station')
        if docks_predicted < DOCK_THRESHOLD_MIN:
            problems.append('very few docks at end station')
        
        return {
            'status': 'error',
            'verdict': 'Risky route ❌',
            'icon': '❌',
            'details': f'Problem: {" and ".join(problems)}. Try alternative stations or timing.'
        }
```

---

## 🎨 UI Design

### Layout Structure

```
[🚴 Plan Journey Tab Selected]

┌─────────────────────────────────────────────────┐
│  🚴 Plan Your Journey                           │
│  Enter your start and end points                │
├─────────────────────────────────────────────────┤
│  📍 From                    🎯 To               │
│  [24 Rue de Rivoli, ...]   [Gare du Nord, ...] │
│                                                  │
│              [🔍 Plan My Route]                 │
└─────────────────────────────────────────────────┘

[After clicking "Plan My Route"...]

┌─────────────────────────────────────────────────┐
│  🗺️ Your Route                                  │
│                                                  │
│  [Interactive Map with Route Line]              │
│  • User location (red pin)                      │
│  • Start station (green marker)                 │
│  • End station (blue marker)                    │
│  • Destination (red pin)                        │
│  • Walking path (dotted line)                   │
│  • Biking path (solid line)                     │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  📊 Journey Breakdown                           │
├─────────────────┬─────────────────┬─────────────┤
│  🚶 Walk Start  │  🚴 Bike Ride   │  🚶 Walk End│
│    5 min        │    15 min       │    3 min    │
│    0.4 km       │    3.8 km       │    0.2 km   │
└─────────────────┴─────────────────┴─────────────┘

┌─────────────────────────────────────────────────┐
│  🚲 Start Station: République - Voltaire        │
│  📍 200m from your location                     │
│  🔮 In 5 min: ~6 bikes available                │
│  ✅ Good availability                           │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  🅿️ End Station: Gare du Nord - Est             │
│  📍 150m from destination                       │
│  🔮 In 20 min: ~4 docks available               │
│  ✅ Good availability                           │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  ✅ Perfect! Good to go 🎉                      │
│  6 bikes and 4 docks should be available.      │
│                                                  │
│  Total journey time: ~23 minutes                │
└─────────────────────────────────────────────────┘
```

---

## 📝 Implementation Steps

### Phase 1: Core Functionality (MVP)
- [ ] **Step 1.1:** Add geocoding function with Nominatim
- [ ] **Step 1.2:** Implement Haversine distance calculator
- [ ] **Step 1.3:** Create nearest station finder
- [ ] **Step 1.4:** Build route planning logic
- [ ] **Step 1.5:** Add prediction interpolation function
- [ ] **Step 1.6:** Implement verdict logic
- [ ] **Step 1.7:** Create Journey Planner UI tab
- [ ] **Step 1.8:** Add input form (start + destination)
- [ ] **Step 1.9:** Display basic results (stations + verdict)
- [ ] **Step 1.10:** Test with real addresses

### Phase 2: Visual Enhancements
- [ ] **Step 2.1:** Draw route line on map
- [ ] **Step 2.2:** Add custom markers (start/end stations)
- [ ] **Step 2.3:** Create time breakdown metrics
- [ ] **Step 2.4:** Design station info cards
- [ ] **Step 2.5:** Add loading spinner during calculation

### Phase 3: Advanced Features (Optional)
- [ ] **Step 3.1:** Browser geolocation for "Use my location"
- [ ] **Step 3.2:** Address autocomplete suggestions
- [ ] **Step 3.3:** Alternative station suggestions if risky
- [ ] **Step 3.4:** Save favorite routes (localStorage)
- [ ] **Step 3.5:** Share route link functionality

---

## 🧪 Testing Plan

### Test Cases

**1. Valid Route**
- Input: "24 Rue de Rivoli, Paris" → "Gare du Nord, Paris"
- Expected: Route found, predictions shown, verdict displayed

**2. Very Close Addresses**
- Input: Same start and end address
- Expected: Error message "Start and destination are too close"

**3. Invalid Address**
- Input: "xyzabc123"
- Expected: Error message "Address not found"

**4. Stations Very Far**
- Input: "Paris" → "Lyon"
- Expected: Warning about very long route or error

**5. Network Error**
- Simulate: Nominatim timeout
- Expected: Graceful error message

---

## 🚀 Performance Considerations

### Optimization Strategies
1. **Cache geocoding results** - Same address → don't re-geocode
2. **Batch predictions** - Consider adding batch endpoint to FastAPI
3. **Rate limiting** - Add 1s delay for Nominatim calls
4. **Error boundaries** - Wrap each step in try-except
5. **Loading states** - Show progress during calculation

### Expected Performance
- Geocoding: ~1-2 seconds per address
- Route calculation: <100ms
- ML predictions: ~5-8 seconds (existing API)
- **Total time: ~10-15 seconds** for complete journey plan

---

## 📚 Dependencies

### New Libraries Needed
```python
# Already in requirements.txt:
# - requests (for Nominatim API)
# - numpy (for interpolation)
# - pandas (existing)

# No new dependencies required!
```

---

## 🔒 Security & Privacy

### Considerations
1. **No user location storage** - Addresses processed in memory only
2. **HTTPS for APIs** - Use secure connections
3. **Rate limiting respect** - Follow Nominatim terms (1 req/sec)
4. **User-Agent header** - Identify our app properly
5. **No personal data** - Don't log addresses

---

## 📊 Success Metrics

### MVP Success Criteria
- ✅ User can enter two addresses
- ✅ System finds appropriate stations automatically
- ✅ Predictions are accurate within model's MAE (±3 bikes)
- ✅ Verdict is clear and actionable
- ✅ Results appear in <15 seconds
- ✅ Error messages are helpful

### UX Goals
- **Intuitive** - No training needed
- **Fast** - Results in ~10 seconds
- **Reliable** - Handles edge cases gracefully
- **Actionable** - Clear yes/no/maybe answer

---

## 🎯 Future Enhancements

### Post-MVP Ideas
1. **Real-time alternatives** - Show 2-3 alternative routes
2. **Time scheduling** - "I want to leave at 8:30 AM"
3. **Return trip** - Plan round-trip automatically
4. **Weather integration** - Factor in rain/snow
5. **Route optimization** - Avoid hills, prefer bike lanes
6. **Multi-modal** - Combine with metro/bus
7. **Favorites** - Save frequent routes
8. **Notifications** - "Your station is ready!"

---

## 📝 Notes & Decisions

### Key Design Decisions
1. **Why Nominatim?** Free, no API key, good for Europe, OSM data
2. **Why haversine?** Good enough approximation for short distances
3. **Why interpolation?** Gives more accurate predictions for specific times
4. **Why dedicated tab?** Clear separation of concerns, won't clutter map view

### Known Limitations
1. **Straight-line distance** - Not actual bike route distance
2. **Average speeds** - Doesn't account for individual pace or terrain
3. **No real-time traffic** - Can't predict delays
4. **Station capacity** - Doesn't check if station is full/empty NOW

### Mitigation Strategies
- Show estimates as ranges ("~15 min" instead of "15.234 min")
- Add confidence indicators
- Encourage checking current status before departure
- Clear disclaimer: "Predictions are estimates"

---

## ✅ Definition of Done

### Checklist
- [ ] All Phase 1 steps completed
- [ ] Tested with 10+ different address pairs
- [ ] Error handling for all edge cases
- [ ] UI is clean and intuitive
- [ ] Code is documented with docstrings
- [ ] Commit message follows convention
- [ ] AGENTS.md updated with feature status
- [ ] Ready for demo/LinkedIn post

---

**Document Version:** 1.0  
**Created:** October 24, 2025  
**Author:** AI Assistant + User Collaboration  
**Status:** Ready for Implementation 🚀
