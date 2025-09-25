#!/bin/bash

# MOTIF Dataset Setup Script for VM Environment
# ==============================================
# 
# This script sets up the necessary environment and dependencies
# for generating malware images from the MOTIF dataset in a VM.
#
# Usage: chmod +x setup_vm.sh && ./setup_vm.sh

set -e  # Exit on any error

echo "=================================="
echo "MOTIF Dataset VM Setup Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   print_status "Please run as a regular user with sudo privileges"
   exit 1
fi

# Update system packages
print_status "Updating system packages..."
sudo apt-get update

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    unzip \
    htop \
    tree \
    vim \
    nano

# Install additional utilities for malware analysis
print_status "Installing malware analysis utilities..."
sudo apt-get install -y \
    hexdump \
    file \
    binutils \
    objdump \
    readelf \
    strings

# Create project directory
PROJECT_DIR="$HOME/motif_malware_analysis"
print_status "Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create Python virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python dependencies..."
cat > requirements.txt << 'EOF'
# Core dependencies for malware image generation
numpy>=1.21.0
Pillow>=8.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0

# Optional ML dependencies (uncomment if needed)
# torch>=1.10.0
# torchvision>=0.11.0
# tensorflow>=2.7.0

# Utilities
tqdm>=4.62.0
seaborn>=0.11.0
jupyter>=1.0.0

# For handling various file formats
python-magic>=0.4.24
pefile>=2021.9.3
EOF

pip install -r requirements.txt

# Create directory structure
print_status "Creating directory structure..."
mkdir -p data/{raw,processed,images}
mkdir -p scripts
mkdir -p notebooks
mkdir -p results/{train,test}
mkdir -p logs

# Create sample configuration file
print_status "Creating configuration file..."
cat > config.json << 'EOF'
{
    "dataset_config": {
        "name": "MOTIF",
        "raw_data_dir": "./data/raw",
        "processed_data_dir": "./data/processed", 
        "output_dir": "./data/images",
        "image_size": 256,
        "test_size": 0.2,
        "random_seed": 42
    },
    "processing_config": {
        "batch_size": 1000,
        "max_file_size": 10485760,
        "min_file_size": 1024,
        "supported_extensions": [".exe", ".dll", ".bin", ""],
        "skip_extensions": [".txt", ".log", ".jpg", ".png", ".gif", ".pdf"]
    },
    "logging_config": {
        "level": "INFO",
        "log_file": "./logs/processing.log"
    }
}
EOF

# Create a simple data validation script
print_status "Creating data validation script..."
cat > scripts/validate_motif_data.py << 'EOF'
#!/usr/bin/env python3
"""
Quick validation script for MOTIF dataset structure
"""

import os
import sys
from pathlib import Path

def validate_motif_dataset(data_dir):
    """Validate MOTIF dataset structure and files."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        return False
    
    print(f"📁 Checking dataset directory: {data_dir}")
    
    # Count files
    total_files = 0
    file_types = {}
    large_files = 0
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = Path(root) / file
            try:
                size = file_path.stat().st_size
                total_files += 1
                
                if size > 1024 * 1024:  # Files larger than 1MB
                    large_files += 1
                
                # Track file extensions
                ext = file_path.suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
                    
            except Exception as e:
                print(f"⚠️  Error checking file {file_path}: {e}")
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total files: {total_files}")
    print(f"   Large files (>1MB): {large_files}")
    print(f"   File types found:")
    
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        ext_name = ext if ext else "(no extension)"
        print(f"     {ext_name}: {count}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_motif_data.py <dataset_directory>")
        sys.exit(1)
    
    validate_motif_dataset(sys.argv[1])
EOF

chmod +x scripts/validate_motif_data.py

# Create helpful aliases
print_status "Creating helpful aliases..."
cat >> ~/.bashrc << 'EOF'

# MOTIF Project Aliases
alias motif-env='source ~/motif_malware_analysis/venv/bin/activate'
alias motif-dir='cd ~/motif_malware_analysis'
alias motif-generate='python ~/motif_malware_analysis/generate_image.py'
alias motif-validate='python ~/motif_malware_analysis/scripts/validate_motif_data.py'
EOF

# Create a simple startup script
print_status "Creating startup script..."
cat > start_motif_analysis.sh << 'EOF'
#!/bin/bash

echo "Starting MOTIF Malware Analysis Environment..."
echo "=============================================="

# Activate virtual environment
source venv/bin/activate

# Show current directory structure
echo "Current directory structure:"
tree -L 2

echo ""
echo "Available commands:"
echo "  python generate_image.py --help          # Generate malware images"
echo "  python visualize_dataset.py --help       # Visualize generated dataset"  
echo "  python dataset_loader.py --help          # Test dataset loading"
echo "  python scripts/validate_motif_data.py    # Validate raw data"
echo ""
echo "Environment ready! 🚀"
EOF

chmod +x start_motif_analysis.sh

# Create README file
print_status "Creating README file..."
cat > README.md << 'EOF'
# MOTIF Malware Image Generation

This environment is set up for converting MOTIF malware dataset files into grayscale images for machine learning classification.

## Quick Start

1. **Activate Environment:**
   ```bash
   ./start_motif_analysis.sh
   ```

2. **Validate Your MOTIF Dataset:**
   ```bash
   python scripts/validate_motif_data.py /path/to/your/motif/dataset
   ```

3. **Generate Images:**
   ```bash
   python generate_image.py --data_dir /path/to/motif/dataset --output_dir ./data/images
   ```

4. **Visualize Results:**
   ```bash
   python visualize_dataset.py --dataset_dir ./data/images --num_samples 16
   ```

5. **Test Data Loading:**
   ```bash
   python dataset_loader.py --dataset_dir ./data/images --test_loading
   ```

## Directory Structure

```
motif_malware_analysis/
├── venv/                  # Python virtual environment
├── data/
│   ├── raw/              # Place your MOTIF dataset here
│   ├── processed/        # Intermediate processing results
│   └── images/           # Generated image datasets
├── scripts/              # Utility scripts
├── notebooks/            # Jupyter notebooks (optional)
├── results/              # ML model results
├── logs/                 # Processing logs
├── generate_image.py     # Main image generation script
├── visualize_dataset.py  # Dataset visualization tool
├── dataset_loader.py     # PyTorch dataset loader
├── config.json           # Configuration settings
└── requirements.txt      # Python dependencies
```

## Configuration

Edit `config.json` to customize:
- Image size (default: 256x256)
- Train/test split ratio
- File size limits
- Processing batch size

## Tips for VM Usage

1. **Monitor Resources:**
   ```bash
   htop  # Monitor CPU and memory usage
   df -h # Check disk space
   ```

2. **Process Large Datasets in Batches:**
   - Use the batch processing features in the scripts
   - Monitor memory usage and adjust batch sizes if needed

3. **Free Up Space:**
   ```bash
   # Clean up intermediate files after processing
   rm -rf data/processed/*
   ```

4. **Backup Important Results:**
   ```bash
   # Compress and backup generated images
   tar -czf motif_images_backup.tar.gz data/images/
   ```

## Troubleshooting

- If you run out of memory, reduce the batch size in config.json
- If processing is slow, ensure you have enough disk space
- Check logs/processing.log for detailed error messages
- Use `python scripts/validate_motif_data.py` to check your input data

## Next Steps

After generating images, you can:
1. Train CNN models for malware classification
2. Experiment with different image sizes
3. Apply data augmentation techniques
4. Use transfer learning with pre-trained models
EOF

# Set permissions
print_status "Setting file permissions..."
find . -name "*.py" -exec chmod +x {} \;
find . -name "*.sh" -exec chmod +x {} \;

# Final setup summary
print_status "Setup completed successfully! 🎉"
echo ""
echo "=================================="
echo "SETUP SUMMARY"
echo "=================================="
echo "✅ System packages installed"
echo "✅ Python virtual environment created"
echo "✅ Required Python packages installed"
echo "✅ Project structure created"
echo "✅ Configuration files generated"
echo "✅ Utility scripts created"
echo ""
echo "📁 Project location: $PROJECT_DIR"
echo ""
echo "🚀 To get started:"
echo "   cd $PROJECT_DIR"
echo "   ./start_motif_analysis.sh"
echo ""
print_warning "Remember to:"
echo "   1. Place your MOTIF dataset in the data/raw/ directory"
echo "   2. Validate your dataset before processing"
echo "   3. Adjust config.json settings if needed"
echo ""
print_status "Happy malware analysis! 🔍"