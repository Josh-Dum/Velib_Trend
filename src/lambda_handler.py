"""
AWS Lambda handler for Velib snapshot collection.

This function is designed to be triggered by EventBridge on a schedule.
It captures one Velib snapshot and uploads it to S3 with metadata indexing.
"""

import json
import os
import sys
from datetime import datetime, timezone

# Add the parent directory to sys.path so we can import our modules
sys.path.append('/opt')  # Lambda layer path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snapshot_velib import capture_snapshot
from src.snapshot_index import append_index_for_file


def lambda_handler(event, context):
    """
    Lambda entry point for scheduled Velib snapshot collection.
    
    Args:
        event: EventBridge event (contains schedule info)
        context: Lambda context object
        
    Returns:
        dict: Response with status and details
    """
    
    # Configuration from environment variables
    S3_BUCKET = os.environ.get('S3_BUCKET', 'velib-trend-josh-dum-2025')
    S3_PREFIX = os.environ.get('S3_PREFIX', 'velib')
    UPLOADER_VERSION = os.environ.get('UPLOADER_VERSION', 'lambda-v1.0')
    
    print(f"[lambda] Starting Velib snapshot collection at {datetime.utcnow().isoformat()}Z")
    print(f"[lambda] Target: s3://{S3_BUCKET}/{S3_PREFIX}/snapshots/")
    
    try:
        # Capture snapshot (will be written to /tmp in Lambda)
        result = capture_snapshot(
            out_dir="/tmp/snapshots",  # Lambda writable directory
            page_size=100,
            stale_threshold_sec=3600,  # 1 hour - matches Vélib refresh frequency
            gzip_enabled=True,
            include_empty=False
        )
        
        if result.get("skipped"):
            print("[lambda] Skipped - no records found")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'status': 'skipped',
                    'message': 'No records to process',
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                })
            }
        
        local_path = result["path"]
        records_count = result["records"]
        
        print(f"[lambda] Captured {records_count} records -> {local_path}")
        
        # Upload to S3
        import boto3
        from pathlib import Path
        
        s3 = boto3.client('s3')
        filename = Path(local_path).name
        s3_key = f"{S3_PREFIX}/snapshots/{filename}" if S3_PREFIX else f"snapshots/{filename}"
        
        # Upload the snapshot file
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"[lambda] Uploaded snapshot -> s3://{S3_BUCKET}/{s3_key}")
        
        # Upload index metadata
        try:
            index_s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/" if S3_PREFIX else f"s3://{S3_BUCKET}/"
            index_path = append_index_for_file(local_path, index_path=index_s3_uri)
            print(f"[lambda] Uploaded index -> {index_path}")
        except Exception as e:
            print(f"[lambda] Index upload warning: {e}")
        
        # Clean up local file (Lambda has limited /tmp space)
        os.unlink(local_path)
        print(f"[lambda] Cleaned up local file")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'records': records_count,
                'snapshot_key': s3_key,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        }
        
    except Exception as e:
        error_msg = f"Lambda execution failed: {str(e)}"
        print(f"[lambda] ERROR: {error_msg}")
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': error_msg,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        }


# For local testing
if __name__ == "__main__":
    # Mock event and context for testing
    test_event = {}
    test_context = type('Context', (), {
        'function_name': 'test-velib-collector',
        'aws_request_id': 'test-123'
    })()
    
    result = lambda_handler(test_event, test_context)
    print("Test result:", json.dumps(result, indent=2))