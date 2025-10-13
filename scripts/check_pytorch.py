"""
Quick script to check PyTorch installation and CUDA availability.
"""
import torch

print("=" * 80)
print("🔍 PYTORCH & CUDA CHECK")
print("=" * 80)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test a simple tensor operation on GPU
    try:
        x = torch.rand(3, 3).cuda()
        y = torch.rand(3, 3).cuda()
        z = x + y
        print(f"✅ GPU tensor operations working correctly")
        print(f"   Test tensor device: {z.device}")
    except Exception as e:
        print(f"❌ GPU tensor operations failed: {e}")
else:
    print("⚠️  CUDA not available - will train on CPU (slower)")
    print("   Consider installing PyTorch with CUDA support:")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("=" * 80)
