# 📘 Project Documentation

Welcome to the documentation for **Paris Pulse (Velib Trend)**.
This folder contains all the technical details, architecture decisions, and guides for the project.

## 📚 Documentation Index

### 1. [Architecture & Deployment](01_architecture_and_deployment.md)
*   **Hybrid Cloud Architecture** (Streamlit + Lambda + SageMaker)
*   **Backend Deployment** (Docker, AWS Lambda)
*   **Inference Deployment** (SageMaker Serverless)
*   **Frontend Deployment** (Streamlit Community Cloud)

### 2. [Model Development](02_model_development.md)
*   **Performance History** (v3 vs v4)
*   **Feature Engineering** (Normalization, Temporal features)
*   **Optimization Strategy** (Optuna, GPU acceleration)
*   **Future Improvements**

### 3. [Features & Optimization](03_features_and_optimization.md)
*   **Journey Planner** logic and workflow
*   **Performance Optimization** (Caching strategies, Latency analysis)

---

## 🛠️ Quick Tech Stack Overview

| Component | Technology | Hosting |
|-----------|------------|---------|
| **Frontend** | Streamlit, Plotly, PyDeck | Streamlit Community Cloud |
| **Backend** | FastAPI, Mangum | AWS Lambda (Docker) |
| **ML Model** | PyTorch (LSTM) | Amazon SageMaker (Serverless) |
| **Data** | Pandas, NumPy | Amazon S3 |
| **Orchestration** | Docker Compose | Local / GitHub Actions |

---

## 🔄 Project Lifecycle

1.  **Data Collection**: 
    *   AWS Lambda (`velib-snapshot-collector`) runs hourly.
    *   Saves raw JSON snapshots to S3 (Bronze Layer).
2.  **Data Processing**:
    *   `scripts/bronze_to_silver.py`: Converts raw S3 data to Parquet (Silver Layer).
    *   `scripts/create_sequences.py`: Creates time-series sequences for LSTM training.
3.  **Training & Evaluation**: 
    *   `scripts/train_lstm.py`: Trains the model (locally/EC2) using PyTorch.
    *   `scripts/evaluate_lstm.py`: Generates performance metrics (MAE, R²) and plots.
4.  **Deployment**:
    *   **Model**: Packaged as `model.tar.gz` and deployed to SageMaker Serverless.
    *   **Backend**: Docker container deployed to AWS Lambda.
    *   **Frontend**: Connected to GitHub and deployed on Streamlit Cloud.
5.  **Inference**: 
    *   User Request → Streamlit → AWS Lambda (Backend) → SageMaker (Model) → Prediction.