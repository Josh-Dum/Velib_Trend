# AWS SageMaker Serverless Deployment Guide

**Project**: Paris Pulse - Vélib' Availability Prediction  
**Model**: Multi-Input LSTM v3 (PyTorch)  
**Deployment Strategy**: SageMaker Serverless Inference  
**Author**: Josh  
**Date**: October 21, 2025  

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Why Serverless Inference?](#why-serverless-inference)
3. [Prerequisites](#prerequisites)
4. [Phase 1: Model Packaging](#phase-1-model-packaging)
5. [Phase 2: AWS SageMaker Setup](#phase-2-aws-sagemaker-setup)
6. [Phase 3: Testing & Validation](#phase-3-testing--validation)
7. [Phase 4: API Integration](#phase-4-api-integration)
8. [Phase 5: Monitoring & Cost Control](#phase-5-monitoring--cost-control)
9. [Troubleshooting](#troubleshooting)
10. [Cost Breakdown](#cost-breakdown)

---

## Overview

### What We're Building

Deploy your production-ready LSTM model to AWS SageMaker using **Serverless Inference**. This will enable real-time predictions for Vélib' station availability (T+1h, T+2h, T+3h) through a REST API endpoint.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SERVERLESS INFERENCE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User/Frontend                                               │
│       │                                                      │
│       ├──> FastAPI Backend (src/app.py)                     │
│       │         │                                            │
│       │         ├──> SageMaker Serverless Endpoint          │
│       │         │    (Auto-scales 0→N instances)            │
│       │         │         │                                  │
│       │         │         ├──> Load model.tar.gz from S3    │
│       │         │         ├──> Run inference.py             │
│       │         │         └──> Return predictions           │
│       │         │                                            │
│       │         └──> Return JSON: {T+1h, T+2h, T+3h}        │
│       │                                                      │
│  Model Artifacts (S3):                                       │
│  └── s3://velib-trend-josh-dum-2025/models/lstm-v3/         │
│       └── model.tar.gz                                       │
│            ├── best_model.pth                                │
│            ├── scaler.pkl                                    │
│            ├── station_mappings.json                         │
│            ├── config.json                                   │
│            └── code/                                         │
│                 ├── inference.py                             │
│                 ├── requirements.txt                         │
│                 └── model_utils.py                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- ✅ **Pay-per-request**: Only charged when making predictions
- ✅ **Auto-scaling**: Scales from 0 to N instances automatically
- ✅ **Low cost**: ~$1-3/month for learning/demo
- ✅ **No idle charges**: No cost when endpoint is not being used
- ✅ **Production-ready**: Same infrastructure as always-on endpoints

---

## Why Serverless Inference?

### Comparison with Other Options

| Feature | Serverless | Always-On | Batch Transform |
|---------|-----------|-----------|-----------------|
| Cost (idle) | $0 | $47/month | $0 |
| Cost (active) | $0.20/1000 requests | Included | $0.10/job |
| Latency (cold) | 5-10s | <200ms | N/A (offline) |
| Latency (warm) | <200ms | <200ms | N/A |
| Best for | Learning, demo, low traffic | Production, high traffic | Batch processing |

### Why We Choose Serverless

1. **Cost-effective learning**: Perfect for exploring SageMaker without breaking the bank
2. **Demo-friendly**: Recruiters won't mind a 5s initial load
3. **Same concepts**: Learn all SageMaker features (deployment, monitoring, IAM)
4. **Scalable**: Can handle traffic spikes automatically
5. **Future-proof**: Easy to upgrade to always-on later if needed

---

## Prerequisites

### ✅ What You Already Have

- [x] Production-ready LSTM model (R²=0.815)
- [x] Model artifacts in `data/models/lstm/`:
  - `best_model.pth` (PyTorch weights)
  - `config.json` (architecture config)
- [x] Supporting files in `data/silver/`:
  - `scaler.pkl` (StandardScaler)
  - `station_mappings.json` (station ID mappings)
- [x] AWS account with credentials configured
- [x] S3 bucket: `velib-trend-josh-dum-2025` (eu-west-3)

### 📦 Required Python Packages

All packages should already be in your `requirements.txt`:
- `torch` (PyTorch for model inference)
- `boto3` (AWS SDK for Python)
- `numpy` (numerical operations)
- `pandas` (data manipulation)

### 🔑 AWS Permissions Needed

Your AWS account needs:
- SageMaker full access (create models, endpoints)
- S3 read/write access (upload model artifacts)
- IAM role creation (for SageMaker execution)
- CloudWatch Logs access (monitoring)

---

## Phase 1: Model Packaging

### 1.1 Create Directory Structure

**Goal**: Organize model artifacts for SageMaker deployment

**Create the following structure**:
```
inference/
├── model_artifacts/          # Model files
│   ├── best_model.pth       # PyTorch weights (copy from data/models/lstm/)
│   ├── config.json          # Model config (copy from data/models/lstm/)
│   ├── scaler.pkl           # StandardScaler (copy from data/silver/)
│   └── station_mappings.json # Station mappings (copy from data/silver/)
├── code/                    # Inference code
│   ├── inference.py         # SageMaker entry point (we'll create)
│   ├── requirements.txt     # Dependencies (we'll create)
│   └── model_utils.py       # Helper functions (we'll create)
└── test_local.py            # Local testing script (we'll create)
```

**What to do**:
```bash
# Create directories
mkdir inference
mkdir inference\model_artifacts
mkdir inference\code

# Copy model artifacts
copy data\models\lstm\best_model.pth inference\model_artifacts\
copy data\models\lstm\config.json inference\model_artifacts\
copy data\silver\scaler.pkl inference\model_artifacts\
copy data\silver\station_mappings.json inference\model_artifacts\
```

**Why these files?**
- `best_model.pth`: Your trained model weights
- `config.json`: Architecture details (LSTM layers, sizes, etc.)
- `scaler.pkl`: Normalize/denormalize predictions
- `station_mappings.json`: Map station codes to embedding IDs

---

### 1.2 Create Inference Script

**Goal**: Write the code that SageMaker will execute for predictions

**File**: `inference/code/inference.py`

**SageMaker Contract**: SageMaker expects 4 functions:

1. **`model_fn(model_dir)`**: Load model from S3
   - Called once when container starts
   - Returns the loaded model object
   
2. **`input_fn(request_body, content_type)`**: Parse incoming request
   - Converts JSON to Python objects
   - Validates input format
   
3. **`predict_fn(input_data, model)`**: Run inference
   - Takes parsed input and loaded model
   - Returns raw predictions
   
4. **`output_fn(prediction, accept)`**: Format response
   - Converts predictions to JSON
   - Returns HTTP response

**Input Format** (JSON):
```json
{
  "station_code": "16107",
  "historical_bikes": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  "hour": 14,
  "day_of_week": 1,
  "is_weekend": false,
  "capacity": 20,
  "latitude": 48.8566,
  "longitude": 2.3522
}
```

**Output Format** (JSON):
```json
{
  "station_code": "16107",
  "predictions": {
    "T+1h": 10.5,
    "T+2h": 9.2,
    "T+3h": 8.7
  },
  "inference_time_ms": 45.3
}
```

**Key Concepts**:
- **Cold start**: First request loads model (~5-10s)
- **Warm requests**: Subsequent requests are fast (<200ms)
- **Auto-scaling**: SageMaker adds instances if traffic increases
- **Scale-to-zero**: After 15 min idle, instances shut down (no cost)

---

### 1.3 Create Requirements File

**Goal**: Specify dependencies for SageMaker container

**File**: `inference/code/requirements.txt`

**Contents**:
```
torch==2.0.1
numpy==1.24.3
scikit-learn==1.3.0
```

**Why these versions?**
- Match your local training environment
- `torch`: PyTorch for model inference
- `numpy`: Array operations
- `scikit-learn`: For loading `scaler.pkl`

**Important**: Don't include `boto3` (already in SageMaker container)

---

### 1.4 Create Helper Functions

**Goal**: Modularize inference logic for cleaner code

**File**: `inference/code/model_utils.py`

**Functions to implement**:
- `load_model_artifacts(model_dir)`: Load all files (model, scaler, mappings)
- `preprocess_input(data)`: Normalize and prepare input tensors
- `postprocess_output(predictions, scaler)`: Denormalize predictions
- `validate_input(data)`: Check input format and ranges

**Benefits**:
- Cleaner `inference.py` code
- Easier testing and debugging
- Reusable components

---

### 1.5 Create Local Testing Script

**Goal**: Test inference logic before deploying to AWS

**File**: `inference/test_local.py`

**What it does**:
1. Loads model artifacts locally
2. Simulates SageMaker request format
3. Calls inference functions
4. Validates predictions match evaluation results
5. Measures inference time

**Why test locally?**
- Free (no AWS costs)
- Fast iteration (no deployment wait)
- Easier debugging (full Python environment)
- Catch errors before deployment

**Test cases**:
- Valid station with full 24h history
- Edge case: Station with missing hours
- Edge case: Unknown station code
- Edge case: Invalid input format

---

### 1.6 Package Model for S3

**Goal**: Create `model.tar.gz` file for SageMaker

**Structure inside tar.gz**:
```
model.tar.gz
├── best_model.pth
├── config.json
├── scaler.pkl
├── station_mappings.json
└── code/
    ├── inference.py
    ├── requirements.txt
    └── model_utils.py
```

**Command**:
```bash
cd inference
tar -czf model.tar.gz model_artifacts/* code/
```

**Important**: 
- Directory structure matters! SageMaker expects this exact format
- `code/` directory must be at the root of the archive
- Model artifacts must be at the root level

**Verify packaging**:
```bash
tar -tzf model.tar.gz  # List contents
```

---

## Phase 2: AWS SageMaker Setup

### 2.1 Upload Model to S3

**Goal**: Store model artifacts in S3 for SageMaker to access

**S3 Path**: `s3://velib-trend-josh-dum-2025/models/lstm-v3/model.tar.gz`

**Command**:
```bash
aws s3 cp inference/model.tar.gz s3://velib-trend-josh-dum-2025/models/lstm-v3/model.tar.gz --region eu-west-3
```

**Verify upload**:
```bash
aws s3 ls s3://velib-trend-josh-dum-2025/models/lstm-v3/
```

**Expected output**:
```
2025-10-21 15:30:45    1234567 model.tar.gz
```

**What happens next?**
- SageMaker will download this file when creating the endpoint
- File is cached for faster cold starts
- You can version models by changing the S3 path

---

### 2.2 Create IAM Execution Role

**Goal**: Give SageMaker permissions to access S3 and CloudWatch

**Why needed?**
- SageMaker needs to download your model from S3
- SageMaker needs to write logs to CloudWatch
- Security best practice: least privilege access

**Option A: AWS Console (Recommended for learning)**

1. Go to IAM Console → Roles → Create Role
2. Select "SageMaker" as trusted entity
3. Attach policies:
   - `AmazonSageMakerFullAccess`
   - Custom policy for S3 bucket access (see below)
4. Name: `SageMaker-VelibLSTM-ExecutionRole`
5. Note the Role ARN (e.g., `arn:aws:iam::123456789:role/SageMaker-VelibLSTM`)

**Custom S3 Policy**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::velib-trend-josh-dum-2025",
        "arn:aws:s3:::velib-trend-josh-dum-2025/*"
      ]
    }
  ]
}
```

**Option B: AWS CLI**

```bash
aws iam create-role --role-name SageMaker-VelibLSTM-ExecutionRole --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name SageMaker-VelibLSTM-ExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

**Save the Role ARN** - you'll need it for deployment!

---

### 2.3 Choose PyTorch Container Image

**Goal**: Select pre-built Docker image with PyTorch runtime

**What is a container image?**
- SageMaker uses Docker containers to run your code
- AWS provides pre-built images with PyTorch, numpy, etc.
- You don't need to build Docker images yourself!

**Image Selection**:
- **Framework**: PyTorch
- **Version**: 2.0.1 (match your local training)
- **Python**: 3.10
- **Region**: eu-west-3 (Paris)
- **Type**: Inference (optimized for serving, not training)

**Image URI Format**:
```
763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:2.0.1-cpu-py310
```

**Where to find available images**:
- [AWS Deep Learning Container Images](https://github.com/aws/deep-learning-containers/blob/master/available_images.md)
- Or use this AWS CLI command:
  ```bash
  aws ecr describe-images --region eu-west-3 --registry-id 763104351884 --repository-name pytorch-inference
  ```

**CPU vs GPU**:
- Use **CPU** for Serverless (cheaper, sufficient for LSTM)
- GPU only needed for large models or high throughput

---

### 2.4 Create SageMaker Model

**Goal**: Register your model with SageMaker

**What this does**:
- Links your S3 model artifacts with a container image
- Creates a SageMaker Model resource
- Does NOT deploy anything yet (no charges)

**AWS Console Method**:

1. SageMaker Console → Models → Create Model
2. Model name: `velib-lstm-v3-serverless`
3. IAM role: Select the role you created in 2.2
4. Container definition:
   - Container image: PyTorch image URI from 2.3
   - Model artifact: `s3://velib-trend-josh-dum-2025/models/lstm-v3/model.tar.gz`
   - Environment variables (optional):
     - `MODEL_NAME=velib-lstm-v3`
     - `SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code`

**AWS CLI Method**:

```bash
aws sagemaker create-model \
  --model-name velib-lstm-v3-serverless \
  --primary-container \
    Image=763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:2.0.1-cpu-py310,\
    ModelDataUrl=s3://velib-trend-josh-dum-2025/models/lstm-v3/model.tar.gz \
  --execution-role-arn arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMaker-VelibLSTM-ExecutionRole \
  --region eu-west-3
```

**Verify creation**:
```bash
aws sagemaker describe-model --model-name velib-lstm-v3-serverless --region eu-west-3
```

---

### 2.5 Create Serverless Endpoint Configuration

**Goal**: Configure serverless inference settings

**What is an endpoint configuration?**
- Defines how your model will be deployed
- Specifies memory size and concurrency
- For serverless: no instance type needed!

**Serverless Settings**:

| Setting | Value | Explanation |
|---------|-------|-------------|
| **Memory Size** | 4096 MB (4 GB) | RAM allocated to each instance |
| **Max Concurrency** | 5 | Max parallel requests per instance |

**Memory calculation**:
- Model size: ~0.5 MB (tiny!)
- PyTorch runtime: ~500 MB
- Working memory: ~1 GB
- **Total**: 2 GB (we choose 4 GB for safety)

**Concurrency**:
- Start with 5 (good for learning)
- Can increase to 20 if needed
- SageMaker auto-scales instances to handle load

**AWS Console Method**:

1. SageMaker Console → Endpoint Configurations → Create
2. Configuration name: `velib-lstm-v3-serverless-config`
3. Model: Select `velib-lstm-v3-serverless`
4. Production variant:
   - Variant name: `AllTraffic`
   - **Serverless Inference**:
     - Memory size: 4096 MB
     - Max concurrency: 5

**AWS CLI Method**:

```bash
aws sagemaker create-endpoint-config \
  --endpoint-config-name velib-lstm-v3-serverless-config \
  --production-variants \
    VariantName=AllTraffic,\
    ModelName=velib-lstm-v3-serverless,\
    ServerlessConfig={MemorySizeInMB=4096,MaxConcurrency=5} \
  --region eu-west-3
```

**Verify creation**:
```bash
aws sagemaker describe-endpoint-config --endpoint-config-name velib-lstm-v3-serverless-config --region eu-west-3
```

---

### 2.6 Deploy Serverless Endpoint

**Goal**: Create the live inference endpoint

**⚠️ COST STARTS HERE**: After this step, you'll be charged per request

**What happens during deployment?**
1. SageMaker creates serverless infrastructure (2-5 min)
2. Downloads model.tar.gz from S3
3. Initializes container image
4. Runs `model_fn()` to load your model
5. Endpoint status changes: Creating → InService

**AWS Console Method**:

1. SageMaker Console → Endpoints → Create Endpoint
2. Endpoint name: `velib-lstm-v3-serverless-endpoint`
3. Endpoint configuration: Select `velib-lstm-v3-serverless-config`
4. Click "Create endpoint"
5. Wait ~5 minutes for "InService" status

**AWS CLI Method**:

```bash
aws sagemaker create-endpoint \
  --endpoint-name velib-lstm-v3-serverless-endpoint \
  --endpoint-config-name velib-lstm-v3-serverless-config \
  --region eu-west-3
```

**Monitor deployment**:
```bash
# Check status
aws sagemaker describe-endpoint --endpoint-name velib-lstm-v3-serverless-endpoint --region eu-west-3

# Watch for "EndpointStatus": "InService"
```

**Deployment timeline**:
- 0-2 min: Creating infrastructure
- 2-4 min: Downloading model from S3
- 4-5 min: Initializing container
- 5 min: ✅ InService (ready for requests!)

**What if deployment fails?**
- Check CloudWatch Logs for errors
- Common issues:
  - IAM role permissions (can't access S3)
  - Wrong S3 path (model.tar.gz not found)
  - Invalid inference.py (syntax errors)
  - Missing dependencies in requirements.txt

---

## Phase 3: Testing & Validation

### 3.1 Test with AWS CLI

**Goal**: Verify endpoint responds correctly

**Create test payload** (`test_payload.json`):
```json
{
  "station_code": "16107",
  "historical_bikes": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  "hour": 14,
  "day_of_week": 1,
  "is_weekend": false,
  "capacity": 20,
  "latitude": 48.8566,
  "longitude": 2.3522
}
```

**Invoke endpoint**:
```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name velib-lstm-v3-serverless-endpoint \
  --body file://test_payload.json \
  --content-type application/json \
  --region eu-west-3 \
  output.json
```

**Expected output** (`output.json`):
```json
{
  "station_code": "16107",
  "predictions": {
    "T+1h": 10.5,
    "T+2h": 9.2,
    "T+3h": 8.7
  },
  "inference_time_ms": 45.3
}
```

**First request (cold start)**:
- Takes 5-10 seconds (model loading)
- This is normal for serverless!

**Subsequent requests (warm)**:
- Takes <200ms (model already loaded)
- Fast enough for real-time applications

**Measure latency**:
```bash
# Cold start test
time aws sagemaker-runtime invoke-endpoint ...

# Warm request test (run immediately after)
time aws sagemaker-runtime invoke-endpoint ...
```

---

### 3.2 Create Python Client

**Goal**: Programmatic access to endpoint

**File**: `scripts/predict_sagemaker.py`

**Key functions**:
```python
import boto3
import json
import time

def predict_station(station_code, historical_bikes, hour, day_of_week, is_weekend, capacity, lat, lon):
    """Make prediction for a single station."""
    client = boto3.client('sagemaker-runtime', region_name='eu-west-3')
    
    payload = {
        "station_code": station_code,
        "historical_bikes": historical_bikes,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "capacity": capacity,
        "latitude": lat,
        "longitude": lon
    }
    
    response = client.invoke_endpoint(
        EndpointName='velib-lstm-v3-serverless-endpoint',
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    result = json.loads(response['Body'].read().decode())
    return result
```

**Error handling**:
```python
try:
    result = predict_station(...)
except client.exceptions.ModelError as e:
    # Model inference error (bad input, model bug)
    print(f"Model error: {e}")
except client.exceptions.ServiceUnavailable as e:
    # Endpoint temporarily unavailable (cold start timeout)
    print(f"Service unavailable: {e}")
    # Retry after 5 seconds
except Exception as e:
    # Other errors (network, permissions, etc.)
    print(f"Unexpected error: {e}")
```

**Test script**:
```python
if __name__ == "__main__":
    # Test with known station
    result = predict_station(
        station_code="16107",
        historical_bikes=[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 
                          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        hour=14,
        day_of_week=1,
        is_weekend=False,
        capacity=20,
        lat=48.8566,
        lon=2.3522
    )
    print(json.dumps(result, indent=2))
```

---

### 3.3 Validation Tests

**Goal**: Verify predictions are accurate and consistent

**Test cases**:

1. **Consistency test**: Same input → same output
   ```python
   pred1 = predict_station(...)
   pred2 = predict_station(...)
   assert pred1 == pred2
   ```

2. **Range test**: Predictions within valid range
   ```python
   assert 0 <= pred['T+1h'] <= capacity
   ```

3. **Trend test**: T+1h, T+2h, T+3h follow logical pattern
   ```python
   # For decreasing pattern
   assert pred['T+1h'] >= pred['T+2h'] >= pred['T+3h']
   ```

4. **Multiple stations test**: Different stations work
   ```python
   for station in ["16107", "16109", "16114"]:
       result = predict_station(station, ...)
       assert result['station_code'] == station
   ```

5. **Performance test**: Latency within acceptable range
   ```python
   start = time.time()
   predict_station(...)
   latency_ms = (time.time() - start) * 1000
   assert latency_ms < 1000  # < 1 second (warm requests)
   ```

---

## Phase 4: API Integration

### 4.1 Enhance FastAPI Backend

**Goal**: Add `/predict` endpoint to `src/app.py`

**New endpoint**:
```python
from pydantic import BaseModel
import boto3
import json

# Request model
class PredictionRequest(BaseModel):
    station_code: str
    historical_bikes: list[float]  # 24 hours
    hour: int
    day_of_week: int
    is_weekend: bool
    capacity: int
    latitude: float
    longitude: float

# Response model
class PredictionResponse(BaseModel):
    station_code: str
    predictions: dict  # {T+1h, T+2h, T+3h}
    inference_time_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_availability(request: PredictionRequest):
    """Get predictions for a station."""
    client = boto3.client('sagemaker-runtime', region_name='eu-west-3')
    
    # Call SageMaker endpoint
    response = client.invoke_endpoint(
        EndpointName='velib-lstm-v3-serverless-endpoint',
        ContentType='application/json',
        Body=json.dumps(request.dict())
    )
    
    result = json.loads(response['Body'].read().decode())
    return result
```

**Key features to add**:
1. **Input validation**: Check historical_bikes length == 24
2. **Error handling**: Return 503 if endpoint unavailable
3. **Caching**: Cache predictions for 5 minutes (reduce costs)
4. **Rate limiting**: Max 100 requests/hour per IP
5. **Logging**: Log all predictions for debugging

**Caching example** (using Redis or in-memory):
```python
from cachetools import TTLCache

# Cache predictions for 5 minutes
cache = TTLCache(maxsize=1000, ttl=300)

@app.post("/predict")
async def predict_availability(request: PredictionRequest):
    cache_key = f"{request.station_code}_{request.hour}"
    
    # Check cache first
    if cache_key in cache:
        return cache[cache_key]
    
    # Call SageMaker
    result = invoke_sagemaker(request)
    
    # Cache result
    cache[cache_key] = result
    return result
```

---

### 4.2 Update Streamlit Frontend

**Goal**: Add prediction view to `src/streamlit_app.py`

**UI enhancements**:

1. **Mode toggle**: Live / Predicted
   ```python
   mode = st.radio("View mode", ["Live availability", "Predictions"])
   ```

2. **Station click handler**: Show predictions when clicking station
   ```python
   if mode == "Predictions":
       selected_station = st.text_input("Station code")
       if selected_station:
           predictions = fetch_predictions(selected_station)
           display_predictions(predictions)
   ```

3. **Prediction display**:
   ```python
   col1, col2, col3 = st.columns(3)
   with col1:
       st.metric("T+1h", f"{pred['T+1h']:.1f} bikes", delta=f"{delta_1h:+.1f}")
   with col2:
       st.metric("T+2h", f"{pred['T+2h']:.1f} bikes", delta=f"{delta_2h:+.1f}")
   with col3:
       st.metric("T+3h", f"{pred['T+3h']:.1f} bikes", delta=f"{delta_3h:+.1f}")
   ```

4. **Confidence indicators**:
   - Green: High availability (>60% capacity)
   - Orange: Medium (30-60%)
   - Red: Low (<30%)

5. **Loading states**:
   ```python
   with st.spinner("Loading predictions..."):
       predictions = fetch_predictions(selected_station)
   ```

---

## Phase 5: Monitoring & Cost Control

### 5.1 Enable CloudWatch Monitoring

**Goal**: Track endpoint performance and errors

**Metrics to monitor**:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `Invocations` | Total requests | - |
| `ModelLatency` | Inference time | >1000ms (p99) |
| `Overhead Latency` | Cold start time | >10000ms |
| `Invocation4XXErrors` | Client errors | >1% |
| `Invocation5XXErrors` | Server errors | >0.1% |

**View metrics**:
1. CloudWatch Console → Metrics → SageMaker
2. Select endpoint: `velib-lstm-v3-serverless-endpoint`
3. Add to dashboard for monitoring

**Create dashboard**:
```bash
aws cloudwatch put-dashboard \
  --dashboard-name VelibLSTM-Monitoring \
  --dashboard-body file://dashboard.json
```

---

### 5.2 Set Up CloudWatch Alarms

**Goal**: Get notified of issues

**Critical alarms**:

1. **High error rate**:
   ```bash
   aws cloudwatch put-metric-alarm \
     --alarm-name velib-lstm-high-errors \
     --metric-name Invocation5XXErrors \
     --namespace AWS/SageMaker \
     --statistic Sum \
     --period 300 \
     --threshold 10 \
     --comparison-operator GreaterThanThreshold
   ```

2. **High latency**:
   ```bash
   aws cloudwatch put-metric-alarm \
     --alarm-name velib-lstm-high-latency \
     --metric-name ModelLatency \
     --namespace AWS/SageMaker \
     --statistic Average \
     --period 300 \
     --threshold 1000 \
     --comparison-operator GreaterThanThreshold
   ```

**Notification setup**:
1. Create SNS topic
2. Subscribe your email
3. Link alarms to SNS topic

---

### 5.3 Cost Management

**Goal**: Keep AWS costs under $5/month

**Strategies**:

1. **Set up billing alerts**:
   ```bash
   aws budgets create-budget \
     --account-id YOUR-ACCOUNT-ID \
     --budget file://budget.json \
     --notifications-with-subscribers file://notifications.json
   ```

2. **Monitor costs**:
   - AWS Console → Billing → Cost Explorer
   - Filter by service: SageMaker
   - View daily breakdown

3. **Delete endpoint when not testing**:
   ```bash
   # Stop endpoint (stops charges)
   aws sagemaker delete-endpoint \
     --endpoint-name velib-lstm-v3-serverless-endpoint \
     --region eu-west-3
   
   # Recreate later (model/config still exist)
   aws sagemaker create-endpoint \
     --endpoint-name velib-lstm-v3-serverless-endpoint \
     --endpoint-config-name velib-lstm-v3-serverless-config \
     --region eu-west-3
   ```

4. **Use caching** (reduce SageMaker calls):
   - Cache predictions for 5-10 minutes
   - 1 cache hit = $0 SageMaker cost
   - 90% cache hit rate = 90% cost savings

**Expected costs**:

| Component | Usage | Cost/Month |
|-----------|-------|------------|
| Serverless inference | 1000 requests | $0.20 |
| S3 storage (model) | 1 MB | $0.02 |
| CloudWatch Logs | 100 MB | $0.50 |
| Data transfer | Negligible | $0.10 |
| **Total** | | **$0.82** |

**Cost scaling**:
- 10,000 requests: $2/month
- 100,000 requests: $20/month
- 1,000,000 requests: $200/month

---

## Troubleshooting

### Common Issues

#### 1. Endpoint creation fails

**Symptoms**: Status stays "Creating" or changes to "Failed"

**Causes**:
- IAM role doesn't have S3 access
- Model artifact path incorrect
- Container image doesn't exist

**Debug**:
```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name velib-lstm-v3-serverless-endpoint --region eu-west-3

# Check CloudWatch Logs
aws logs tail /aws/sagemaker/Endpoints/velib-lstm-v3-serverless-endpoint --region eu-west-3
```

**Fix**:
- Verify IAM role permissions
- Verify S3 path: `aws s3 ls s3://velib-trend-josh-dum-2025/models/lstm-v3/`
- Check container image URI

---

#### 2. Invocation fails with 500 error

**Symptoms**: `ModelError` or `InternalServerError`

**Causes**:
- Bug in `inference.py` code
- Missing dependencies
- Model file corrupted
- Invalid input format

**Debug**:
```bash
# Check CloudWatch Logs for stack trace
aws logs tail /aws/sagemaker/Endpoints/velib-lstm-v3-serverless-endpoint --region eu-west-3 --follow
```

**Fix**:
- Test `inference.py` locally first
- Verify all files in `model.tar.gz`
- Check input JSON matches expected format

---

#### 3. High latency (>10s every request)

**Symptoms**: All requests take 5-10 seconds

**Causes**:
- Endpoint is constantly cold starting
- Memory size too small
- Model loading is slow

**Debug**:
```bash
# Check OverheadLatency metric in CloudWatch
# If high, it's cold start related
```

**Fix**:
- Increase memory size (4 GB → 6 GB)
- Optimize model loading in `model_fn()`
- Send warm-up requests every 10 minutes

---

#### 4. Endpoint not scaling

**Symptoms**: 503 errors during load

**Causes**:
- Max concurrency reached
- Too many concurrent cold starts

**Debug**:
```bash
# Check Invocations and ConcurrentExecutions metrics
```

**Fix**:
- Increase max concurrency (5 → 10)
- Add retry logic in client
- Consider always-on endpoint for high traffic

---

#### 5. High costs

**Symptoms**: Bill >$10/month

**Causes**:
- Too many requests
- No caching
- Debugging with many test requests

**Debug**:
```bash
# Check Cost Explorer for SageMaker usage
# Filter by endpoint name
```

**Fix**:
- Implement caching (5 min TTL)
- Delete endpoint when not testing
- Use batch predictions for testing

---

## Cost Breakdown

### Serverless Inference Pricing (eu-west-3)

**Compute pricing**:
- $0.20 per 1000 requests
- $0.000200 per request

**Memory pricing**:
- Based on GB-seconds: `(Memory in GB) × (Duration in seconds) × $0.00001667`

**Example calculation** (4 GB memory, 200ms duration):
- Compute: 1000 requests × $0.000200 = $0.20
- Memory: 1000 × 4 GB × 0.2s × $0.00001667 = $0.013
- **Total**: $0.21 per 1000 requests

**Monthly estimates**:

| Requests/Month | Compute | Memory | Total |
|----------------|---------|--------|-------|
| 1,000 | $0.20 | $0.01 | $0.21 |
| 10,000 | $2.00 | $0.13 | $2.13 |
| 100,000 | $20.00 | $1.33 | $21.33 |

**Other costs**:
- S3 storage: $0.02/month (1 MB model)
- CloudWatch Logs: $0.50/month (100 MB logs)
- Data transfer: ~$0.10/month (minimal)

**Total monthly cost** (1000 requests): **~$0.83**

---

## Next Steps After Deployment

### Phase 6: Production Enhancements (Optional)

1. **A/B Testing**:
   - Deploy multiple model versions
   - Route 90% traffic to v3, 10% to v4
   - Compare performance

2. **Model versioning**:
   - Keep multiple models in S3
   - Easy rollback if issues

3. **Monitoring dashboard**:
   - Real-time latency graphs
   - Error rate tracking
   - Cost trends

4. **Auto-retraining**:
   - Weekly Lambda job
   - Download new data from S3
   - Retrain model
   - Deploy if improved

5. **Advanced caching**:
   - Redis cluster
   - Pre-compute predictions for popular stations
   - Cache hit rate >95%

---

## Summary

### What You'll Learn

✅ **SageMaker Serverless Inference**: Pay-per-request deployment  
✅ **Model Packaging**: Creating SageMaker-compatible artifacts  
✅ **IAM Roles**: Security and permissions management  
✅ **Container Images**: Using pre-built PyTorch images  
✅ **API Integration**: Connecting FastAPI to SageMaker  
✅ **Cost Optimization**: Caching, monitoring, budget alerts  
✅ **Production Deployment**: Real-world ML inference pipeline  

### Success Criteria

- ✅ Endpoint deployed and "InService"
- ✅ Predictions match evaluation results (R²=0.815)
- ✅ Latency <10s cold start, <200ms warm
- ✅ Error rate <1%
- ✅ Cost <$3/month during testing
- ✅ FastAPI integration working
- ✅ Streamlit UI showing predictions

### Timeline

- **Phase 1** (Model Packaging): 4-6 hours
- **Phase 2** (AWS Setup): 2-3 hours  
- **Phase 3** (Testing): 1-2 hours
- **Phase 4** (API Integration): 3-4 hours
- **Phase 5** (Monitoring): 1-2 hours

**Total**: 11-17 hours over 3-5 days

---

## Resources

### AWS Documentation
- [SageMaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
- [PyTorch Inference Container](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-docker-containers-frameworks.html)
- [IAM Roles for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)

### Helpful Commands
```bash
# List all endpoints
aws sagemaker list-endpoints --region eu-west-3

# Delete endpoint (stop charges)
aws sagemaker delete-endpoint --endpoint-name velib-lstm-v3-serverless-endpoint --region eu-west-3

# View CloudWatch Logs
aws logs tail /aws/sagemaker/Endpoints/velib-lstm-v3-serverless-endpoint --region eu-west-3 --follow

# Check S3 model
aws s3 ls s3://velib-trend-josh-dum-2025/models/lstm-v3/ --region eu-west-3
```

---

**Ready to start?** Let's begin with Phase 1: Model Packaging! 🚀
