# Features & Performance Optimization

## 🗺️ Journey Planner Feature

**"Will I find a bike at the start and a dock at the end?"**

The Journey Planner solves the "last mile" uncertainty in bike sharing.

### How It Works
1.  **Geocoding**: Converts user addresses (Start/End) to coordinates using **Nominatim (OSM)**.
2.  **Station Selection**: Finds the nearest *active* stations to both points.
3.  **Route Calculation**:
    - Walk to start station
    - Bike to end station (using **OSRM** for cycling routes)
    - Walk to destination
4.  **Prediction**:
    - Predicts bike availability at the **Start Station** at arrival time.
    - Predicts dock availability at the **End Station** at arrival time.
5.  **Verdict**: Displays a "Go", "Risk", or "Caution" status based on predictions.

---

## 🚀 Performance Optimization

### 1. Multi-Layer Caching
To ensure a snappy UI despite heavy ML computations, the system uses aggressive caching:

| Cache Layer | Target | TTL | Impact |
|-------------|--------|-----|--------|
| **Live Data** | `/stations` | 5 min | 17s → **0s** |
| **Historical** | S3 Snapshots | 30 min | 4.5s → **0s** |
| **Geocoding** | Address lookup | LRU (128) | 1s → **0s** |

### 2. Latency Analysis
- **Cold Start**: ~4-8s (Lambda/SageMaker container init).
- **Warm Request**: ~1.2s (End-to-end journey plan).
- **Bottleneck**: SageMaker inference (550ms) and Nominatim rate limiting (1s sleep).

### 3. Frontend Optimization
- **Pre-warming**: The backend pre-fetches live data on startup.
- **Asynchronous UI**: Streamlit displays the map immediately while the journey planner calculates in the background.
- **Vector Graphics**: Custom SVG markers for crisp rendering at any zoom level.
