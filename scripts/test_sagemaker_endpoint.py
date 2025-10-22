"""
Test SageMaker endpoint with real prediction request.
This script sends a sample request to the deployed endpoint and displays the predictions.
"""

import boto3
import json
import numpy as np
from datetime import datetime

# Configuration
ENDPOINT_NAME = 'velib-lstm-v3-endpoint'
REGION = 'eu-west-3'

def create_test_payload():
    """
    Create a realistic test payload for a Vélib station.
    Uses typical pattern: moderate usage during afternoon.
    """
    # Simulate 24 hours of bike availability (afternoon pattern)
    # Pattern: stable afternoon usage ~10-15 bikes available
    available_bikes_24h = [
        12, 11, 10, 10, 9, 8, 7, 6,      # Hours 0-7 (night → morning)
        5, 6, 8, 10, 12, 14, 15, 16,     # Hours 8-15 (morning → afternoon)
        15, 14, 13, 12, 11, 10, 10, 9    # Hours 16-23 (evening → night)
    ]
    
    payload = {
        "historical_bikes": available_bikes_24h,
        "station_code": "16107",  # Example station
        "hour": 14,             # 2 PM (current time)
        "day_of_week": 2,       # Tuesday (0=Monday, 6=Sunday)
        "is_weekend": False,
        "capacity": 35,
        "latitude": 48.8566,
        "longitude": 2.3522
    }
    
    return payload

def test_endpoint():
    """Send request to SageMaker endpoint and display results."""
    
    print("=" * 70)
    print("🚀 TESTING SAGEMAKER ENDPOINT")
    print("=" * 70)
    
    # Create boto3 client
    print(f"\n📡 Connecting to endpoint: {ENDPOINT_NAME}")
    print(f"   Region: {REGION}")
    client = boto3.client('sagemaker-runtime', region_name=REGION)
    
    # Prepare test payload
    payload = create_test_payload()
    print(f"\n📊 Test Data:")
    print(f"   Station: {payload['station_code']}")
    print(f"   Time: Tuesday 14:00 (2 PM)")
    print(f"   Current availability: {payload['historical_bikes'][-1]} bikes")
    print(f"   Station capacity: {payload['capacity']} bikes")
    print(f"   24h history: {payload['historical_bikes'][:8]}... (showing first 8 hours)")
    
    # Send request to endpoint
    print(f"\n⏳ Sending request to SageMaker...")
    print(f"   (First request will be COLD START - may take 5-10 seconds)")
    
    start_time = datetime.now()
    
    try:
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Parse response
        response_data = json.loads(response['Body'].read().decode())
        predictions = response_data['predictions']  # Extract nested predictions
        
        print(f"\n✅ SUCCESS! Received predictions in {latency_ms:.0f} ms")
        print("=" * 70)
        print("🔮 PREDICTIONS:")
        print("=" * 70)
        print(f"   Current (T=0):  {payload['historical_bikes'][-1]} bikes")
        print(f"   T+1h (15:00):   {predictions['T+1h']} bikes predicted")
        print(f"   T+2h (16:00):   {predictions['T+2h']} bikes predicted")
        print(f"   T+3h (17:00):   {predictions['T+3h']} bikes predicted")
        
        # Model metadata
        print(f"\n🤖 MODEL INFO:")
        print(f"   Version: {response_data['model_version']}")
        print(f"   Inference time: {response_data['inference_time_ms']:.2f} ms")
        print(f"   Timestamp: {response_data['timestamp']}")
        
        # Analysis
        print("\n📈 ANALYSIS:")
        change_1h = predictions['T+1h'] - payload['historical_bikes'][-1]
        change_3h = predictions['T+3h'] - payload['historical_bikes'][-1]
        
        print(f"   1-hour trend: {change_1h:+d} bikes ({'+' if change_1h > 0 else ''})")
        print(f"   3-hour trend: {change_3h:+d} bikes ({'+' if change_3h > 0 else ''})")
        
        if change_3h > 0:
            print(f"   📊 Prediction: Station filling up (more bikes arriving)")
        elif change_3h < 0:
            print(f"   📊 Prediction: Station emptying (more bikes leaving)")
        else:
            print(f"   📊 Prediction: Stable usage (no major change)")
        
        print("\n💰 COST ESTIMATE:")
        print(f"   This request cost: ~$0.0002 (0.02 cents)")
        print(f"   Monthly (100 requests): ~$0.02")
        
        print("\n" + "=" * 70)
        print("🎉 ENDPOINT TEST SUCCESSFUL!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n🔍 Troubleshooting:")
        print("   1. Check endpoint is 'InService' in AWS Console")
        print("   2. Verify AWS credentials are configured")
        print("   3. Check CloudWatch logs for container errors")
        return False

if __name__ == "__main__":
    success = test_endpoint()
    
    if success:
        print("\n✨ Next steps:")
        print("   1. Try running this script again (warm inference ~200ms)")
        print("   2. Check CloudWatch metrics in AWS Console")
        print("   3. Integrate endpoint into FastAPI backend")
        print("   4. Build Streamlit UI with live predictions")
