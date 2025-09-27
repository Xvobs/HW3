#!/usr/bin/env python3
"""
Mac M4 (Apple Silicon) GPU Setup Guide
=====================================

This script helps you set up PyTorch with Metal Performance Shaders (MPS) 
for GPU acceleration on Mac M4 and other Apple Silicon chips.
"""

import subprocess
import sys
import platform
import torch


def check_system():
    """Check system compatibility."""
    print("🔍 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Processor: {platform.processor()}")
    
    # Check if running on Apple Silicon
    is_apple_silicon = platform.machine() in ['arm64', 'aarch64']
    if is_apple_silicon:
        print("✅ Apple Silicon detected - MPS support available!")
    else:
        print("ℹ️  Not Apple Silicon - will use CPU or CUDA if available")
    
    return is_apple_silicon


def check_pytorch_mps():
    """Check PyTorch MPS availability."""
    print("\n🔍 PyTorch Configuration:")
    print(f"   PyTorch version: {torch.__version__}")
    
    # Check MPS availability
    if hasattr(torch.backends, 'mps'):
        print(f"   MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("✅ MPS is ready for GPU acceleration!")
            return True
        else:
            print("⚠️  MPS not available - check macOS version (requires 12.3+)")
    else:
        print("❌ MPS not supported in this PyTorch version")
        print("   You need PyTorch 1.12+ for MPS support")
    
    return False


def install_optimized_pytorch():
    """Install PyTorch optimized for Apple Silicon."""
    print("\n🛠️  Installing PyTorch with MPS support...")
    
    # Install command for Apple Silicon optimized PyTorch
    install_cmd = [
        sys.executable, "-m", "pip", "install", "--upgrade",
        "torch", "torchvision", "torchaudio"
    ]
    
    try:
        result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch installation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def test_gpu_performance():
    """Test GPU performance with a simple operation."""
    print("\n🧪 Testing GPU Performance...")
    
    # Test different devices
    devices_to_test = []
    
    if torch.cuda.is_available():
        devices_to_test.append(('cuda', 'NVIDIA GPU'))
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices_to_test.append(('mps', 'Apple Silicon GPU'))
    
    devices_to_test.append(('cpu', 'CPU'))
    
    # Simple matrix multiplication benchmark
    size = 2048
    iterations = 5
    
    for device_name, device_desc in devices_to_test:
        print(f"\n   Testing {device_desc} ({device_name})...")
        device = torch.device(device_name)
        
        try:
            # Create test tensors
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warm up
            for _ in range(2):
                c = torch.mm(a, b)
            
            if device.type in ['cuda', 'mps']:
                torch.cuda.synchronize() if device.type == 'cuda' else None
            
            # Benchmark
            import time
            start_time = time.time()
            
            for _ in range(iterations):
                c = torch.mm(a, b)
                
            if device.type in ['cuda', 'mps']:
                torch.cuda.synchronize() if device.type == 'cuda' else None
            
            end_time = time.time()
            avg_time = (end_time - start_time) / iterations
            
            print(f"   ⏱️  Average time: {avg_time:.3f}s")
            print(f"   🚀 Performance: {(size*size*size*2/avg_time/1e9):.1f} GFLOPS")
            
        except Exception as e:
            print(f"   ❌ Error testing {device_name}: {e}")


def create_optimized_training_script():
    """Create an optimized training script for Mac M4."""
    script_content = '''#!/usr/bin/env python3
"""
Mac M4 Optimized Training Script
"""

import torch
import os

# Optimize for Apple Silicon
if torch.backends.mps.is_available():
    print("🚀 Using Apple Silicon GPU acceleration!")
    
    # Set environment variables for optimal performance
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Use all available memory
    
    # Recommend optimal batch sizes for M4
    print("💡 Recommended settings for M4:")
    print("   - Batch size: 16-64 (depending on model size)")
    print("   - Image size: 224-512")
    print("   - Reduce num_workers to 2-4 for DataLoader")

# Example training command optimized for M4
training_command = """
python train_cnn_example.py \\
    --dataset_dir ./motif_images \\
    --epochs 20 \\
    --batch_size 32 \\
    --lr 0.001 \\
    --image_size 256 \\
    --save_model
"""

print("\\n🔧 Optimized training command:")
print(training_command)
'''
    
    with open('mac_m4_training.py', 'w') as f:
        f.write(script_content)
    
    print("\n✅ Created mac_m4_training.py with optimized settings")


def main():
    print("="*60)
    print("🍎 Mac M4 GPU Setup for Malware Image Classification")
    print("="*60)
    
    # Check system
    is_apple_silicon = check_system()
    
    # Check PyTorch MPS
    mps_available = check_pytorch_mps()
    
    if is_apple_silicon and not mps_available:
        print("\n🔄 Installing/updating PyTorch for Apple Silicon...")
        if install_optimized_pytorch():
            # Re-import torch after installation
            import importlib
            importlib.reload(torch)
            mps_available = check_pytorch_mps()
    
    # Test performance
    if mps_available:
        test_gpu_performance()
        create_optimized_training_script()
        
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("="*60)
        print("Your Mac M4 is ready for GPU-accelerated training!")
        print("\n🚀 Next steps:")
        print("1. Generate your malware images:")
        print("   python generate_image.py --data_dir /path/to/MOTIF --output_dir ./motif_images")
        print("2. Train with GPU acceleration:")
        print("   python train_cnn_example.py --dataset_dir ./motif_images --epochs 20")
        print("3. Monitor GPU usage with Activity Monitor")
        
    else:
        print("\n⚠️  GPU acceleration not available")
        print("   Your training will use CPU, which will be slower but still functional")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()