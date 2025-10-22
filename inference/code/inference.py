"""
SageMaker Inference Script for Vélib LSTM Model

This script implements the SageMaker inference contract with 4 required functions:
- model_fn: Load model from S3
- input_fn: Parse incoming JSON requests
- predict_fn: Run inference
- output_fn: Format predictions as JSON

Author: Josh
Date: October 21, 2025
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler  # Required to unpickle scaler


# ============================================================================
# MODEL ARCHITECTURE (must match training)
# ============================================================================

class MultiInputLSTM(nn.Module):
    """
    Multi-Input LSTM model for time series prediction.
    
    This is the SAME architecture as train_lstm.py - must be identical!
    """
    
    def __init__(
        self,
        lstm_hidden_1: int = 128,
        lstm_hidden_2: int = 64,
        dense_hidden: int = 32,
        static_features_size: int = 7,
        output_size: int = 3,
        dropout_rate: float = 0.2
    ):
        """Initialize the Multi-Input LSTM model."""
        super(MultiInputLSTM, self).__init__()
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_1,
            num_layers=1,
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_1,
            hidden_size=lstm_hidden_2,
            num_layers=1,
            batch_first=True
        )
        
        # Dense layers
        self.fc1 = nn.Linear(lstm_hidden_2 + static_features_size, dense_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_hidden, output_size)
    
    def forward(self, X_seq: torch.Tensor, X_static: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Reshape sequence for LSTM: (batch, 24) → (batch, 24, 1)
        X_seq = X_seq.unsqueeze(-1)
        
        # Pass through LSTM layers
        lstm_out1, _ = self.lstm1(X_seq)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Take last timestep output
        lstm_last = lstm_out2[:, -1, :]
        
        # Concatenate with static features
        combined = torch.cat([lstm_last, X_static], dim=1)
        
        # Pass through dense layers
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        predictions = self.fc2(x)
        
        return predictions


# ============================================================================
# SAGEMAKER INFERENCE CONTRACT
# ============================================================================

def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load model and artifacts from SageMaker model directory.
    
    This function is called ONCE when the container starts (cold start).
    It loads:
    - PyTorch model weights
    - Scaler for normalization/denormalization
    - Station mappings for embedding lookup
    - Model configuration
    
    Args:
        model_dir: Path to model artifacts (e.g., /opt/ml/model/)
        
    Returns:
        dict: Contains model, scaler, station_mappings, config
    """
    try:
        print(f"[model_fn] Loading model from: {model_dir}")
        
        # Use CPU for inference (cheaper than GPU for serverless)
        device = torch.device("cpu")
        
        # Load configuration
        config_path = os.path.join(model_dir, "config.json")
        print(f"[model_fn] Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load station mappings
        mappings_path = os.path.join(model_dir, "station_mappings.json")
        print(f"[model_fn] Loading station mappings from: {mappings_path}")
        with open(mappings_path, 'r') as f:
            station_mappings = json.load(f)
        
        # Load scaler (for normalization/denormalization)
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        print(f"[model_fn] Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"[model_fn] Scaler loaded successfully!")
        
        # Initialize model with config
        print(f"[model_fn] Initializing model architecture...")
        model = MultiInputLSTM(
            lstm_hidden_1=config.get('lstm_hidden_1', 128),
            lstm_hidden_2=config.get('lstm_hidden_2', 64),
            dense_hidden=config.get('dense_hidden', 32),
            static_features_size=config.get('static_features_size', 7),
            output_size=config.get('output_size', 3),
            dropout_rate=config.get('dropout_rate', 0.2)
        )
        
        # Load trained weights
        model_path = os.path.join(model_dir, "best_model.pth")
        print(f"[model_fn] Loading model weights from: {model_path}")
        
        # Load checkpoint (contains model_state_dict, optimizer, epoch, etc.)
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract just the model weights from checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[model_fn] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # If it's just the state dict directly (not a checkpoint)
            model.load_state_dict(checkpoint)
        
        # Set to evaluation mode (disables dropout)
        model.eval()
        model.to(device)
        
        print(f"[model_fn] Model loaded successfully!")
        print(f"[model_fn] - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"[model_fn] - Stations: {len(station_mappings)}")
        
        # Return everything needed for inference
        return {
            'model': model,
            'scaler': scaler,
            'station_mappings': station_mappings,
            'config': config,
            'device': device
        }
    
    except Exception as e:
        print(f"[model_fn] ERROR: Failed to load model!")
        print(f"[model_fn] Exception type: {type(e).__name__}")
        print(f"[model_fn] Exception message: {str(e)}")
        import traceback
        print(f"[model_fn] Traceback:")
        traceback.print_exc()
        raise


def input_fn(request_body: str, content_type: str = 'application/json') -> Dict[str, Any]:
    """
    Parse and validate incoming request.
    
    This function is called for EACH request. It converts the JSON input
    into Python objects that predict_fn can use.
    
    Expected input format:
    {
        "station_code": "16107",
        "historical_bikes": [12, 11, 10, ..., 11],  # 24 hours
        "hour": 14,
        "day_of_week": 1,
        "is_weekend": false,
        "capacity": 20,
        "latitude": 48.8566,
        "longitude": 2.3522
    }
    
    Args:
        request_body: Raw request body (JSON string)
        content_type: Content type (should be 'application/json')
        
    Returns:
        dict: Parsed and validated input data
        
    Raises:
        ValueError: If input format is invalid
    """
    print(f"[input_fn] Parsing request (content_type: {content_type})")
    
    if content_type != 'application/json':
        raise ValueError(f"Unsupported content type: {content_type}. Expected 'application/json'")
    
    # Parse JSON
    try:
        data = json.loads(request_body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    
    # Validate required fields
    required_fields = [
        'station_code', 'historical_bikes', 'hour', 'day_of_week',
        'is_weekend', 'capacity', 'latitude', 'longitude'
    ]
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Validate historical_bikes length
    if len(data['historical_bikes']) != 24:
        raise ValueError(f"historical_bikes must have 24 values, got {len(data['historical_bikes'])}")
    
    # Validate ranges
    if not (0 <= data['hour'] <= 23):
        raise ValueError(f"hour must be 0-23, got {data['hour']}")
    
    if not (0 <= data['day_of_week'] <= 6):
        raise ValueError(f"day_of_week must be 0-6, got {data['day_of_week']}")
    
    if not isinstance(data['is_weekend'], bool):
        raise ValueError(f"is_weekend must be boolean, got {type(data['is_weekend'])}")
    
    print(f"[input_fn] Request validated successfully for station: {data['station_code']}")
    
    return data


def predict_fn(input_data: Dict[str, Any], model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference on the input data.
    
    This function is called for EACH request after input_fn. It:
    1. Extracts model, scaler, etc. from model_dict
    2. Prepares input tensors
    3. Runs model inference
    4. Denormalizes predictions
    5. Returns results with metadata
    
    Args:
        input_data: Parsed input from input_fn
        model_dict: Model artifacts from model_fn
        
    Returns:
        dict: Predictions and metadata
    """
    start_time = time.time()
    
    # Extract model artifacts
    model = model_dict['model']
    scaler = model_dict['scaler']
    station_mappings = model_dict['station_mappings']
    device = model_dict['device']
    
    station_code = input_data['station_code']
    print(f"[predict_fn] Running inference for station: {station_code}")
    
    # Check if station exists in mappings
    if station_code not in station_mappings:
        # Unknown station - use a default embedding (e.g., 0)
        print(f"[predict_fn] Warning: Unknown station {station_code}, using default embedding")
        station_id = 0
    else:
        station_id = station_mappings[station_code]
    
    # Prepare sequence input (24 hours of bike availability)
    historical_bikes = np.array(input_data['historical_bikes'], dtype=np.float32)
    
    # Normalize using the same scaler from training
    historical_bikes_normalized = scaler.transform(historical_bikes.reshape(-1, 1)).flatten()
    
    # Convert to PyTorch tensor: (1, 24) - batch size of 1
    X_seq = torch.tensor(historical_bikes_normalized, dtype=torch.float32).unsqueeze(0)
    
    # Prepare static features: [hour, day_of_week, is_weekend, capacity, station_id, lat, lon]
    X_static = torch.tensor([
        input_data['hour'],
        input_data['day_of_week'],
        1.0 if input_data['is_weekend'] else 0.0,
        input_data['capacity'],
        station_id,
        input_data['latitude'],
        input_data['longitude']
    ], dtype=torch.float32).unsqueeze(0)  # (1, 7)
    
    # Move tensors to device
    X_seq = X_seq.to(device)
    X_static = X_static.to(device)
    
    # Run inference (no gradient computation needed)
    with torch.no_grad():
        predictions_normalized = model(X_seq, X_static)  # (1, 3)
    
    # Convert to numpy and denormalize
    predictions_normalized = predictions_normalized.cpu().numpy().flatten()  # (3,)
    predictions = scaler.inverse_transform(predictions_normalized.reshape(-1, 1)).flatten()
    
    # Round to integers (you can't have 9.8 bikes!) and ensure non-negative
    predictions = np.maximum(predictions, 0)
    predictions = np.round(predictions).astype(int)
    
    # Calculate inference time
    inference_time_ms = (time.time() - start_time) * 1000
    
    print(f"[predict_fn] Predictions: T+1h={predictions[0]:.1f}, T+2h={predictions[1]:.1f}, T+3h={predictions[2]:.1f}")
    print(f"[predict_fn] Inference time: {inference_time_ms:.2f}ms")
    
    # Return predictions with metadata
    return {
        'station_code': station_code,
        'predictions': {
            'T+1h': int(predictions[0]),
            'T+2h': int(predictions[1]),
            'T+3h': int(predictions[2])
        },
        'inference_time_ms': round(inference_time_ms, 2),
        'model_version': 'v3_475snapshots',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    }


def output_fn(prediction: Dict[str, Any], accept: str = 'application/json') -> str:
    """
    Format predictions as JSON response.
    
    This function is called LAST for each request. It converts the prediction
    dictionary into a JSON string that will be returned to the client.
    
    Args:
        prediction: Prediction dict from predict_fn
        accept: Content type requested by client
        
    Returns:
        str: JSON-formatted prediction
        
    Raises:
        ValueError: If accept type is not supported
    """
    print(f"[output_fn] Formatting output (accept: {accept})")
    
    if accept != 'application/json':
        raise ValueError(f"Unsupported accept type: {accept}. Expected 'application/json'")
    
    # Convert to JSON string
    response = json.dumps(prediction, indent=2)
    
    print(f"[output_fn] Response ready ({len(response)} bytes)")
    
    return response


# ============================================================================
# LOCAL TESTING (optional, for debugging)
# ============================================================================

if __name__ == "__main__":
    """
    Local testing script - runs inference without SageMaker.
    
    Usage: python inference.py
    """
    print("=" * 80)
    print("LOCAL TESTING MODE")
    print("=" * 80)
    
    # Simulate model directory (parent of code/ directory)
    model_dir = Path(__file__).parent.parent / "model_artifacts"
    
    # Test model loading
    print("\n1. Testing model_fn()...")
    model_dict = model_fn(str(model_dir))
    print("   ✅ Model loaded successfully!")
    
    # Test input parsing
    print("\n2. Testing input_fn()...")
    test_input = json.dumps({
        "station_code": "16107",
        "historical_bikes": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "hour": 14,
        "day_of_week": 1,
        "is_weekend": False,
        "capacity": 20,
        "latitude": 48.8566,
        "longitude": 2.3522
    })
    parsed_input = input_fn(test_input, 'application/json')
    print("   ✅ Input parsed successfully!")
    
    # Test prediction
    print("\n3. Testing predict_fn()...")
    prediction = predict_fn(parsed_input, model_dict)
    print("   ✅ Prediction generated successfully!")
    
    # Test output formatting
    print("\n4. Testing output_fn()...")
    response = output_fn(prediction, 'application/json')
    print("   ✅ Output formatted successfully!")
    
    # Display final result
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(response)
    print("=" * 80)
    print("\n✅ All tests passed! Inference script is ready for SageMaker deployment.")
