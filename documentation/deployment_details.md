# AWS Deployment Details

## 📦 Phase 1: Model Packaging (Oct 21, 2025)

### Phase 1.1 - Directory Structure
- Created `inference/model_artifacts/` and `inference/code/` directories
- Copied 4 model artifacts: best_model.pth (1.44 MB), config.json, scaler.pkl, station_mappings.json
- Total model size: ~1.5 MB (excellent for serverless deployment)

### Phase 1.2 - Inference Script
- Implemented complete SageMaker inference contract (model_fn, input_fn, predict_fn, output_fn)
- Fixed checkpoint loading (handles model_state_dict in checkpoint dict)
- Predictions rounded to integers (10 bikes, not 9.800000190734863)
- Local testing successful: 7ms inference time (after initial load)
- Model loaded from epoch 12 (best validation checkpoint)

### Phase 1.3 - Requirements
- Minimal dependencies: torch==2.0.1, numpy==1.24.3, scikit-learn==1.3.0
- Excluded boto3 (already in SageMaker containers)

### Phase 1.4 - Tarball Packaging
- Successfully created tarball: 1.29 MB compressed
- Structure: model artifacts at root + code/ subdirectory
- Ready for S3 upload and SageMaker deployment

## 🔐 Phase 2: AWS Setup (Oct 22, 2025)

### Phase 2.1 - S3 Upload
- Added IAM write permissions via custom inline policy `VelibS3WriteAccess`
- Successfully uploaded model.tar.gz to S3 (1.3 MiB)
- S3 path: `s3://velib-trend-josh-dum-2025/sagemaker/velib-lstm-v3/model.tar.gz`

### Phase 2.2 - IAM Execution Role
- Role name: `VelibSageMakerExecutionRole`
- Role ARN: `arn:aws:iam::864559191020:role/VelibSageMakerExecutionRole`
- Attached policy: `AmazonSageMakerFullAccess` (includes S3 read, CloudWatch logs, SageMaker APIs)
- Trust relationship: Allows sagemaker.amazonaws.com to assume role

### Phase 2.3 - SageMaker Model
- Model name: `velib-lstm-v3`
- Container: PyTorch 2.0.1 CPU inference (py310)
- Model artifacts: S3 path configured
- Status: Active and ready for endpoint deployment

### Phase 2.4 - Serverless Endpoint Deployment
- **Endpoint name**: `velib-lstm-v3-endpoint`
- **Status**: InService (working after 4 recreations)
- **Container**: PyTorch 2.0.1 **CPU-only** (no GPU - cost optimized!)
- **Memory**: 3072 MB (3 GB) - Minimum viable for PyTorch

#### Bugs Encountered and Fixed:
1. ~~Missing sklearn import~~ → FIXED (added to inference.py)
2. ~~Wrong tarball structure~~ → FIXED (artifacts at root, code/ subdirectory)
3. ~~NumPy version incompatibility~~ → FIXED (flexible version range `numpy>=1.23.0,<2.0.0`)
4. ~~Test script input format mismatch~~ → FIXED (field names matched to inference.py)

#### Performance:
- First inference: 4.3s (cold start)
- Subsequent: ~550-600ms (warm)
- Model inference: 6-8ms (actual prediction time - very fast!)
- Cost per request: ~$0.0002 (0.02 cents)

### Phase 2.5 - Test Endpoint
- Test script: `scripts/test_sagemaker_endpoint.py` working perfectly
- Cold start latency: ~4.3 seconds (first request after idle)
- Warm latency: ~550-600ms (subsequent requests)
- Model inference time: 6-8ms (very fast!)
- Predictions: Reasonable and properly formatted (T+1h, T+2h, T+3h)
- Response includes: predictions, model version, inference time, timestamp

## 📚 Deployment Lessons Learned

### SageMaker Tarball Structure is Strict
- Model artifacts MUST be at root of tarball (not in subdirectories)
- Code files go in `code/` subdirectory
- SageMaker extracts to `/opt/ml/model/` and looks for artifacts there
- Verification: Use `tar -tzf model.tar.gz` to check structure

### Python Pickle is Environment-Sensitive
- Exact version pinning (e.g., `numpy==1.24.3`) can cause compatibility issues
- pickle files reference internal module structure (e.g., `numpy._core`)
- Different builds of same version may have different internal structure
- Solution: Use flexible version ranges (e.g., `numpy>=1.23.0,<2.0.0`)
- This allows environment to install compatible version for pickle deserialization

### API Contracts Must Match Exactly
- Field names between client and server must be identical
- Test scripts must match inference function expectations
- Document your API contract clearly (required fields, types, ranges)

### CloudWatch Logs Are Essential
- Every error provided exact next step for debugging
- Navigate: AWS Console → CloudWatch → Log groups → /aws/sagemaker/Endpoints/{endpoint-name}
- Look for ERROR or CRITICAL level messages
- Container startup logs show import/loading errors immediately

### Systematic Debugging Works
- Evidence-based approach: Error → CloudWatch logs → Fix → Verify
- Don't guess - use logs to identify exact issue
- One fix at a time - repackage, upload, recreate endpoint, test
- Persistence pays off (4 endpoint recreations led to success!)

### Performance Metrics
- Cold start: 4-5 seconds (first request after idle period)
- Warm latency: 550-600ms (subsequent requests)
- Model inference: 6-8ms (actual prediction time - very fast!)
- Rest of latency: Container overhead (network, serialization, etc.)
- Cost: ~$0.0002 per inference (0.02 cents) - affordable for portfolio project

## 🌐 S3 Historical Data Performance

### Real-World Measurements
- Single snapshot fetch: 0.508s (S3 download: 0.477s [94%], gzip: 0.002s, parse: 0.006s)
- Sequential (theoretical): 12.2s (24 × 0.508s)
- **Parallel with ThreadPoolExecutor (10 workers): 3.9s** ✅ **3.1x speedup achieved!**
- Bottleneck: Network/S3 API latency (94% of time), not code efficiency

### File Characteristics
- Compressed: 87.5 KB per snapshot
- Decompressed: 834.6 KB (9.5x compression)
- Total 24 snapshots: ~2.05 MB transfer

### End-to-End Prediction Latency (User-Facing)
- Vélib API: 200-500ms
- S3 history fetch: 3.9s (parallel)
- SageMaker inference: 600ms (warm) / 4.3s (cold)
- FastAPI overhead: 10-50ms
- **Total: 4.8-8.7s** (acceptable for portfolio project)

### Cost Per Prediction Request
- S3 GET (24 requests): $0.00001
- S3 transfer (~2 MB): $0.00024
- SageMaker inference: $0.0002
- **Total: $0.00025** (0.025 cents) - very affordable!

### Monthly Cost Estimates
- 100 requests: $0.03
- 1,000 requests: $0.25
- 10,000 requests: $2.50

### Optimization Conclusion
Parallel downloads work correctly and provide 3.1x speedup. Network latency is the remaining bottleneck. Current 4.8s latency is acceptable for portfolio/demo purposes. Future optimizations (caching, DynamoDB, pre-aggregation) available if needed but add cost/complexity.
