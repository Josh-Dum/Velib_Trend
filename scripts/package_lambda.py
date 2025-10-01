"""
Package the Velib collector for AWS Lambda deployment.

This script creates a ZIP file containing all necessary code and dependencies
that can be uploaded to AWS Lambda.
"""

import os
import shutil
import zipfile
from pathlib import Path
import subprocess
import sys

def create_lambda_package():
    """Create a Lambda deployment package."""
    
    print("Creating Lambda deployment package...")
    
    # Create temporary directory for packaging
    package_dir = Path("lambda_package")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    print(f"Package directory: {package_dir.absolute()}")
    
    # Copy source files
    src_files = [
        "src/lambda_handler.py",
        "src/snapshot_velib.py", 
        "src/fetch_live_velib.py",
        "src/snapshot_index.py",
        "src/data_utils.py"  # Added - needed by other modules
    ]
    
    # Create src directory in package
    (package_dir / "src").mkdir()
    
    for src_file in src_files:
        src_path = Path(src_file)
        if src_path.exists():
            dst_path = package_dir / src_file
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {src_file}")
        else:
            print(f"Warning: {src_file} not found")
    
    # Install dependencies in package directory
    print("Installing dependencies (requests with all dependencies)...")
    
    # Install requests directly - pip will include all sub-dependencies
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "requests",
            "-t", str(package_dir),
            "--upgrade"
        ], check=True, capture_output=True, text=True)
        print("Dependencies installed successfully")
        if result.stdout:
            print(f"Installed: {result.stdout[:200]}")
    except subprocess.CalledProcessError as e:
        print(f"Pip install failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    
    # Create ZIP file
    zip_path = Path("velib-lambda-deployment.zip")
    if zip_path.exists():
        zip_path.unlink()
    
    print(f"Creating ZIP file: {zip_path.absolute()}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                arc_path = file_path.relative_to(package_dir)
                zipf.write(file_path, arc_path)
                
    # Clean up temp directory
    shutil.rmtree(package_dir)
    
    zip_size = zip_path.stat().st_size / 1024 / 1024  # MB
    print(f"Package created: {zip_path} ({zip_size:.1f} MB)")
    
    if zip_size > 50:
        print("Warning: Package is larger than 50MB - consider using Lambda Layers")
    
    return zip_path

if __name__ == "__main__":
    try:
        zip_path = create_lambda_package()
        print(f"\n✅ Success! Upload {zip_path} to AWS Lambda")
        print("\nNext steps:")
        print("1. Go to AWS Lambda console")
        print("2. Create new function")
        print("3. Upload the ZIP file")
        print("4. Set handler to: src.lambda_handler.lambda_handler")
        print("5. Configure environment variables")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)