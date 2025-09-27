# MOTIF M### Key Principles:
1. **Binary-to-Image Conversion**: Each byte in the malware binary (0-255) directly maps to a grayscale pixel intensity
2. **Spatial Locality**: Consecutive bytes in the binary become adjacent pixels, preserving structural patterns
3. **Grayscale Representation**: Images use single-channel grayscale (mode 'L') with uint8 pixel values
4. **Dimension Calculation**: Image dimensions are calculated to preserve the complete binary content

## ⚠️ Important Note: Binary Classification with MOTIF

**MOTIF contains only malware samples** (3,095 samples from 454 families). For binary classification (malware vs benign), you have several options:

1. **Add Benign Samples**: Include legitimate software samples in a separate directory
2. **Multi-class Classification**: Use MOTIF for malware family classification instead
3. **Artificial Binary Split**: Treat some families as "benign" class for training purposes

The scripts will work but will create a dataset with only malware samples unless you add benign files.

## 🚀 Quick StartImage Generation

This repository contains tools for converting malware binary files from the MOTIF dataset into grayscale images for machine learning classification, following the methodology from Nataraj et al. (2011).

## � Methodology

This implementation follows the approach from:
**"Malware images: visualization and automatic classification"** by Nataraj et al. (VizSec 2011)

### Key Principles:
1. **Binary-to-Image Conversion**: Each byte in the malware binary (0-255) directly maps to a grayscale pixel intensity
2. **Spatial Locality**: Consecutive bytes in the binary become adjacent pixels, preserving structural patterns
3. **Grayscale Representation**: Images use single-channel grayscale (mode 'L') with uint8 pixel values
4. **Dimension Calculation**: Image dimensions are calculated to preserve the complete binary content

## �🚀 Quick Start

### For VM Environment

### For MOTIF Dataset (Recommended)

The MOTIF dataset is directly supported! Follow these steps:

1. **Download the MOTIF dataset:**
   ```bash
   git lfs clone https://github.com/boozallen/MOTIF.git
   ```

2. **Extract the malware samples:**
   ```bash
   cd MOTIF
   7z x MOTIF.7z  # Password: i_assume_all_risk_opening_malware
   ```

3. **Setup GPU acceleration (Mac M4/Apple Silicon):**
   ```bash
   python setup_mac_m4.py  # Check and optimize PyTorch for Apple Silicon
   ```

4. **Analyze and setup your MOTIF dataset:**
   ```bash
   python setup_motif.py /path/to/MOTIF
   ```

5. **Generate malware images:**
   ```bash
   python generate_image.py --data_dir /path/to/MOTIF --output_dir ./motif_images
   ```

6. **Verify methodology compliance:**
   ```bash
   python verify_nataraj_images.py --image_dir ./motif_images/train/malware
   ```

7. **Train with GPU acceleration:**
   ```bash
   python train_cnn_example.py --dataset_dir ./motif_images --epochs 20 --batch_size 32
   ```

### For VM Environment

1. **Setup the environment:**
   ```bash
   chmod +x setup_vm.sh
   ./setup_vm.sh
   ```

2. **Place your MOTIF dataset in the data directory and generate images:**
   ```bash
   # Fixed-size images (256x256) - recommended for CNN training
   python generate_image.py --data_dir /path/to/motif/dataset --output_dir ./motif_images
   
   # Variable-size images (original Nataraj approach)
   python generate_image.py --data_dir /path/to/motif/dataset --output_dir ./motif_images --variable_size
   ```

3. **Verify methodology compliance:**
   ```bash
   python verify_nataraj_images.py --image_dir ./motif_images/train/malware
   ```

4. **Visualize the generated dataset:**
   ```bash
   python visualize_dataset.py --dataset_dir ./motif_images --num_samples 16
   ```

### For Local Environment

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate malware images:**
   ```bash
   python generate_image.py --data_dir /path/to/motif/dataset --output_dir ./dataset
   ```

## 📁 Files Description

### Core Scripts

- **`generate_image.py`** - Main script to convert malware binaries to grayscale images (Nataraj methodology)
- **`setup_motif.py`** - MOTIF dataset configuration and analysis helper
- **`setup_mac_m4.py`** - Mac M4/Apple Silicon GPU setup and optimization
- **`verify_nataraj_images.py`** - Verify that generated images follow the correct methodology
- **`visualize_dataset.py`** - Visualize and validate the generated image dataset  
- **`dataset_loader.py`** - PyTorch DataLoader for the generated images
- **`train_cnn_example.py`** - Example CNN training script using the generated malware images
- **`setup_vm.sh`** - Automated setup script for VM environments

### Configuration

- **`requirements.txt`** - Python package dependencies
- **`README.md`** - This file

## 🔧 Usage Examples

### Basic Image Generation (Nataraj Methodology)

```bash
# Generate fixed-size 256x256 images (recommended for CNN training)
python generate_image.py \
    --data_dir /path/to/motif/dataset \
    --output_dir ./malware_images \
    --image_size 256 \
    --test_size 0.2

# Generate variable-size images (original Nataraj approach)
python generate_image.py \
    --data_dir /path/to/motif/dataset \
    --output_dir ./malware_images \
    --variable_size \
    --test_size 0.2

# Generate high-resolution images for detailed analysis
python generate_image.py \
    --data_dir /path/to/motif/dataset \
    --output_dir ./malware_images_hires \
    --image_size 512 \
    --verbose
```

### Methodology Verification

```bash
# Verify that generated images follow Nataraj methodology
python verify_nataraj_images.py \
    --image_dir ./malware_images/train/malware \
    --sample_size 10

# Create detailed visualization of a specific malware image
python verify_nataraj_images.py \
    --visualize ./malware_images/train/malware/sample001.png \
    --output_dir ./analysis_plots
```

### Dataset Visualization

```bash
# Visualize 16 random samples from the dataset
python visualize_dataset.py \
    --dataset_dir ./malware_images \
    --num_samples 16 \
    --save_plots ./plots
```

### Testing Data Loading

```bash
# Test PyTorch data loading functionality
python dataset_loader.py \
    --dataset_dir ./malware_images \
    --batch_size 32

# Example CNN training with generated images
python train_cnn_example.py \
    --dataset_dir ./malware_images \
    --epochs 20 \
    --batch_size 32 \
    --save_model
```
    --batch_size 32 \
    --test_loading
```

## 🏗️ Generated Dataset Structure

```
dataset/
├── train/
│   ├── malware/
│   │   ├── malware_000001_a1b2c3d4.png
│   │   ├── malware_000002_e5f6g7h8.png
│   │   └── ...
│   └── benign/
│       ├── benign_000001_i9j0k1l2.png
│       └── ...
├── test/
│   ├── malware/
│   └── benign/
└── dataset_info.json
```

## ⚙️ Configuration Options

### Image Generation Options

- `--image_size`: Size of square images (default: 256)
- `--test_size`: Fraction for test split (default: 0.2)
- `--data_dir`: Path to MOTIF dataset
- `--output_dir`: Where to save generated images

### Dataset Structure Requirements

The script automatically detects malware vs benign files based on:
- Directory names containing: `malware`, `malicious`, `virus`, `trojan`, etc.
- Directory names containing: `benign`, `clean`, `goodware`, etc.
- File names with malware-related keywords

## 🔍 How It Works

1. **File Discovery**: Recursively scans the dataset directory to find binary files
2. **Classification**: Automatically categorizes files as malware or benign based on directory/file names
3. **Binary to Image**: Converts binary data to grayscale images:
   - Reads raw bytes from binary files
   - Converts bytes to pixel values (0-255)
   - Reshapes to square images
   - Pads or truncates to desired size
4. **Train/Test Split**: Randomly splits data maintaining class balance
5. **Image Export**: Saves as PNG images with organized directory structure

## 📊 Dataset Information

The generated `dataset_info.json` contains:
- Dataset statistics (file counts, splits)
- Processing parameters
- Class mappings
- Generation metadata

## 🐛 Troubleshooting

### Common Issues

1. **Empty dataset directory**: Ensure MOTIF files are in the specified directory
2. **Memory errors**: Reduce batch size or image size
3. **Permission errors**: Ensure write permissions to output directory
4. **No files detected**: Check directory structure matches expected patterns

### Getting Help

```bash
# Get detailed help for any script
python generate_image.py --help
python visualize_dataset.py --help
python dataset_loader.py --help
```

## � Complete Workflow

### For VM Environment (Recommended)

```bash
# 1. Setup environment
chmod +x setup_vm.sh
./setup_vm.sh

# 2. Generate malware images using Nataraj methodology
python generate_image.py \
    --data_dir /path/to/motif/dataset \
    --output_dir ./malware_images \
    --image_size 256 \
    --test_size 0.2 \
    --verbose

# 3. Verify methodology compliance
python verify_nataraj_images.py \
    --image_dir ./malware_images/train/malware \
    --sample_size 5

# 4. Visualize dataset
python visualize_dataset.py \
    --dataset_dir ./malware_images \
    --num_samples 16

# 5. Train example CNN model
python train_cnn_example.py \
    --dataset_dir ./malware_images \
    --epochs 20 \
    --batch_size 32 \
    --save_model
```

### Expected Output Structure

```
malware_images/
├── train/
│   ├── malware/           # Training malware images
│   └── benign/            # Training benign images
├── test/
│   ├── malware/           # Testing malware images
│   └── benign/            # Testing benign images
└── dataset_info.json      # Dataset statistics
```

## �🔬 Advanced Usage

### Custom File Classification

Modify the file discovery logic in `generate_image.py` to customize how files are classified as malware vs benign.

### Custom Image Preprocessing

Extend the `MalwareImageGenerator` class to implement custom preprocessing:
- Different normalization schemes
- Feature extraction
- Multi-channel representations

### Integration with ML Pipelines

Use `dataset_loader.py` as a starting point for integrating with your machine learning training pipelines:

```python
from dataset_loader import get_data_loaders

train_loader, test_loader = get_data_loaders('./dataset', batch_size=32)

# Use in your training loop
for images, labels in train_loader:
    # Your model training code
    pass
```

## 📋 System Requirements

- Python 3.6+
- 4GB+ RAM (for typical datasets)
- Storage space: ~2-3x the size of your original dataset
- Optional: CUDA-capable GPU for deep learning frameworks

## 📄 License

This project is provided for educational and research purposes. Ensure compliance with your institution's policies when working with malware datasets.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!