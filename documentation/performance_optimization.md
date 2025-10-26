# Performance Optimization Analysis

**Created:** October 24, 2025  
**Goal:** Identify and optimize all performance bottlenecks while respecting API rate limits and AWS costs

---

## 🔍 Current Performance Bottlenecks

### Journey Planner Flow Analysis

**User Action:** Enter start address + destination → Click "Plan My Route"

**Current Sequential Flow:**
```
1. Geocode start address          → 1-2s (Nominatim API + 1s rate limit sleep)
2. Geocode destination address    → 1-2s (Nominatim API + 1s rate limit sleep)
3. Find nearest start station     → <50ms (local calculation)
4. Find nearest end station       → <50ms (local calculation)
5. Calculate route                → <10ms (local calculation)
6. Get prediction for START       → 8-20s (FastAPI → SageMaker)
7. Get prediction for END         → 8-20s (FastAPI → SageMaker)
8. Calculate verdict              → <10ms (local calculation)

TOTAL: 18-44 seconds (mostly waiting on APIs!)
```

### Specific Bottlenecks Identified

#### 1. **Nominatim Geocoding: 2-4 seconds total**
- **Location:** `src/journey_planner.py` line 52: `time.sleep(1)`
- **Issue:** Mandatory 1-second sleep per address (2 addresses = 2 seconds)
- **Rate Limit:** 1 request/second (Nominatim terms of service)
- **Cost:** FREE (OpenStreetMap public service)
- **Impact:** Unavoidable but acceptable (UX: show "Geocoding addresses..." spinner)

#### 2. **SageMaker Predictions: 16-40 seconds total**
- **Location:** `src/app.py` line 58-130 (FastAPI `/predict/{station_code}`)
- **Issue:** 
  - Cold start: 4-8 seconds first time (container initialization)
  - Warm: 550-600ms per request
  - Journey Planner makes 2 sequential calls (start + end stations)
- **Rate Limit:** None (your own endpoint)
- **Cost:** $0.0002 per request (~$0.0004 per journey = 0.04 cents)
- **Impact:** **CRITICAL - Main bottleneck!**

#### 3. **Sequential API Calls**
- **Location:** `src/streamlit_app.py` lines 615-630 (Journey Planner)
- **Issue:** Predictions called one after another instead of parallel
- **Impact:** 2x latency (if both warm: 1.2s instead of 0.6s)

#### 4. **No Caching**
- **Issue:** Same station predictions requested multiple times in short period
- **Example:** User tries multiple routes from same start location
- **Impact:** Redundant SageMaker calls ($$$)

---

## 📊 API Rate Limits & Costs Summary

| Service | Rate Limit | Cost | Current Usage |
|---------|------------|------|---------------|
| **Nominatim** (geocoding) | 1 req/second | FREE | 2 per journey |
| **Vélib API** (live data) | No published limit | FREE | 1 per prediction |
| **SageMaker** (predictions) | No limit | $0.0002/req | 2 per journey |
| **S3** (historical data) | High | ~$0.0001/req | 1 per prediction |

**Total cost per journey:** ~$0.0008 (0.08 cents) + AWS data transfer

**Monthly estimate (100 journeys/day):**
- 100 journeys/day × 30 days = 3,000 journeys
- 3,000 × $0.0008 = **$2.40/month** (predictions only)
- SageMaker idle cost: **$1-3/month** (serverless)
- **Total: ~$3-5/month** ✅ Very affordable!

---

## 🚀 Optimization Strategy

### Priority 1: **Parallel Prediction Calls** (Easy Win)
**Impact:** Reduce journey time from 18-44s to 10-24s (up to 50% faster!)

**Current:**
```python
# Sequential - SLOW
start_pred = get_prediction_at_time(start_station['stationcode'], ...)  # 8-20s
end_pred = get_prediction_at_time(end_station['stationcode'], ...)      # 8-20s
# Total: 16-40s
```

**Optimized:**
```python
import concurrent.futures

# Parallel - FAST
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    future_start = executor.submit(get_prediction_at_time, start_station['stationcode'], ...)
    future_end = executor.submit(get_prediction_at_time, end_station['stationcode'], ...)
    
    start_pred = future_start.result()  # Wait for both
    end_pred = future_end.result()
# Total: 8-20s (max of the two, not sum!)
```

**Benefits:**
- ✅ 50% faster predictions
- ✅ No additional cost
- ✅ No breaking changes
- ✅ Simple implementation

---

### Priority 2: **Cache Predictions** (Medium Effort)
**Impact:** Reduce redundant SageMaker calls, lower costs, faster UX

**Strategy:** Cache predictions for 5-10 minutes per station
- Same station requested twice in 5 min → instant response
- Predictions are T+1h, T+2h, T+3h → still valid after 5 min

**Implementation Options:**

#### Option A: Streamlit Session State (Simple)
```python
# In streamlit_app.py
if 'prediction_cache' not in st.session_state:
    st.session_state.prediction_cache = {}

def get_cached_prediction(station_code, minutes_from_now):
    cache_key = f"{station_code}_{int(minutes_from_now/5)*5}"  # Round to 5-min buckets
    
    if cache_key in st.session_state.prediction_cache:
        cached_data, timestamp = st.session_state.prediction_cache[cache_key]
        if time.time() - timestamp < 300:  # 5 minutes
            return cached_data
    
    # Fetch fresh prediction
    pred = get_prediction_at_time(station_code, minutes_from_now, API_BASE_URL)
    st.session_state.prediction_cache[cache_key] = (pred, time.time())
    return pred
```

**Benefits:**
- ✅ Instant response for repeated queries
- ✅ Lower AWS costs
- ✅ Per-user cache (session state)
- ⚠️ Cons: Cache lost on browser refresh

#### Option B: FastAPI Server-Side Cache (Better for production)
```python
# In src/app.py
from functools import lru_cache
from datetime import datetime, timedelta

prediction_cache = {}

@app.get("/predict/{station_code}")
def predict_station(station_code: str):
    now = datetime.now()
    cache_key = f"{station_code}_{now.hour}"  # Cache per hour
    
    # Check cache
    if cache_key in prediction_cache:
        cached_data, timestamp = prediction_cache[cache_key]
        if now - timestamp < timedelta(minutes=10):
            cached_data['cached'] = True
            return cached_data
    
    # Fetch fresh (existing code...)
    result = {...}  # Your existing prediction logic
    
    # Store in cache
    prediction_cache[cache_key] = (result, now)
    result['cached'] = False
    return result
```

**Benefits:**
- ✅ Shared across all users
- ✅ Survives browser refresh
- ✅ Significant cost reduction (many users hit same stations)
- ✅ Cache invalidation per hour (predictions naturally expire)

---

### Priority 3: **Batch Predictions Endpoint** (Advanced)
**Impact:** Reduce SageMaker overhead for Journey Planner

**Current:** 2 separate calls to `/predict/{station_code}`
- Each call: fetch live data, call SageMaker, format response
- Overhead: 2× live data fetch, 2× SageMaker overhead

**Optimized:** Single call to `/predict/batch`
```python
POST /predict/batch
{
    "stations": [
        {"code": "22603", "minutes_from_now": 15},
        {"code": "23201", "minutes_from_now": 25}
    ]
}
```

**Backend:** Call SageMaker once with both stations (if model supports batch)
- Single live data fetch for all stations
- Single SageMaker invocation (if batch supported)

**Benefits:**
- ✅ Lower latency (less HTTP overhead)
- ✅ Lower cost (fewer API calls)
- ⚠️ Requires model refactoring (currently processes 1 station at a time)

---

### Priority 4: **Optimistic UI Loading** (UX Enhancement)
**Impact:** Make slow operations feel faster

**Strategy:** Show intermediate results as they arrive
```python
# Journey Planner UI
with st.spinner("🌍 Geocoding addresses..."):
    start_coords = geocode_address(start_address)
    dest_coords = geocode_address(dest_address)

st.success(f"✅ Found: {start_address} → {dest_address}")

with st.spinner("🔍 Finding nearest stations..."):
    start_station = find_nearest_station(...)
    end_station = find_nearest_station(...)

st.info(f"📍 Route: {start_station['name']} → {end_station['name']}")

with st.spinner("🔮 Getting availability predictions..."):
    # Parallel predictions here
    start_pred, end_pred = get_predictions_parallel(...)

# Show results
```

**Benefits:**
- ✅ User sees progress (not just loading)
- ✅ Perceived performance improvement
- ✅ Clear feedback at each step
- ✅ No code changes to core logic

---

## 🎯 Recommended Implementation Order

### Phase 1: Quick Wins (30 minutes)
1. ✅ **Parallel predictions** in Journey Planner
2. ✅ **Optimistic UI loading** with progress spinners
3. ✅ **Timeout adjustments** (increase to 45s for Journey Planner)

**Expected impact:** 50% faster, better UX

### Phase 2: Caching (1-2 hours)
1. ✅ **Streamlit session cache** for predictions
2. ✅ **FastAPI server cache** (10-minute TTL)
3. ✅ **Geocoding cache** (store common addresses)

**Expected impact:** 80% faster for repeat queries, lower costs

### Phase 3: Advanced (Optional, 4+ hours)
1. ⏸️ **Batch prediction endpoint** (requires model refactor)
2. ⏸️ **Redis cache** for production (replace in-memory)
3. ⏸️ **CDN for static data** (station locations)

---

## 🔒 Safety Considerations

### Rate Limit Compliance
1. **Nominatim:** Keep 1s sleep (required by TOS)
2. **SageMaker:** No limit, but monitor costs
3. **Vélib API:** No published limit, but add backoff if errors

### Cost Control
1. **Alert:** Set AWS CloudWatch alarm at $10/month
2. **Cache:** Reduce redundant SageMaker calls
3. **Monitor:** Track requests/day in logs

### User Experience
1. **Timeout:** Set reasonable limits (45s max for Journey Planner)
2. **Feedback:** Show progress, don't just spin
3. **Graceful degradation:** Show partial results if one prediction fails

---

## 📈 Expected Results

### Before Optimization
- Journey Planner: 18-44 seconds
- User drops off waiting → Bad UX
- Cost: $0.0008/journey (acceptable)

### After Phase 1 (Parallel + UI)
- Journey Planner: 10-24 seconds (50% faster)
- Progress feedback → Better UX
- Cost: Same ($0.0008/journey)

### After Phase 2 (+ Caching)
- Journey Planner: 2-24 seconds (instant if cached)
- Repeat queries: <2 seconds ⚡
- Cost: ~$0.0004/journey (50% reduction for repeat users)

---

## 🛠️ Implementation Checklist

### Phase 1: Quick Wins
- [ ] Add `concurrent.futures` to Journey Planner
- [ ] Implement parallel prediction calls
- [ ] Add progress spinners to UI
- [ ] Test with real addresses
- [ ] Measure performance improvement
- [ ] Update documentation

### Phase 2: Caching
- [ ] Add prediction cache to Streamlit
- [ ] Add geocoding cache (common addresses)
- [ ] Add server-side cache to FastAPI
- [ ] Test cache hit rates
- [ ] Monitor cost reduction
- [ ] Add cache metrics to UI (show "⚡ Cached" indicator)

### Phase 3: Monitoring
- [ ] Add CloudWatch cost alerts
- [ ] Log prediction latencies
- [ ] Track cache hit rates
- [ ] Monitor API error rates
- [ ] Set up performance dashboard

---

**Status:** Analysis complete, ready for Phase 1 implementation  
**Next Step:** Implement parallel predictions + progress UI
