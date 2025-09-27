# MOTIF Multi-Class Family Classification

## Updated Dataset Structure

The MOTIF dataset contains **3,095 malware samples from 454 different malware families**, not a simple binary malware/benign classification. This project has been updated to properly handle the multi-class family classification task.

## Key Changes

### 1. Dataset Generation (`generate_image.py`)

**Previous Version:** 
- Treated MOTIF as binary classification (malware vs benign)
- Created only 2 classes
- Used generic malware/benign directories

**Updated Version:**
- Supports **454 malware families** from the MOTIF dataset
- Uses `motif_dataset.jsonl` to extract family labels
- Creates family-specific directories (e.g., `train/emotet/`, `train/zeus/`, etc.)
- Generates comprehensive dataset statistics per family

### 2. Training Script (`train_family_classifier.py`)

**New Features:**
- CNN architecture optimized for 454-class classification
- Handles class imbalance with weighted loss functions
- Includes top-1 and top-5 accuracy metrics
- Advanced regularization (batch normalization, dropout)
- Learning rate scheduling

## Dataset Structure

### MOTIF Dataset Information
- **Total Samples:** 3,095 malware samples
- **Families:** 454 distinct malware families
- **Source:** Based on threat intelligence reports from 2016-2021
- **Format:** Disarmed PE files converted to grayscale images

### Required Files
To use the multi-class family classification, you need:

1. **`motif_dataset.jsonl`** - Contains family labels for each sample
   ```json
   {
     "md5": "abc123...",
     "label": 42,
     "reported_family": "emotet",
     "aliases": ["emotet", "geodo"],
     ...
   }
   ```

2. **`MOTIF_defanged/`** - Directory containing binary files
   ```
   MOTIF_defanged/
   ├── MOTIF_abc123...
   ├── MOTIF_def456...
   └── ...
   ```

## Usage

### Step 1: Generate Multi-Class Dataset

```bash
# Generate family-based image dataset
python generate_image.py \\
    --data_dir /path/to/motif/dataset \\
    --output_dir ./motif_family_dataset \\
    --image_size 256 \\
    --test_size 0.2
```

This creates:
```
motif_family_dataset/
├── dataset_info.json
├── train/
│   ├── emotet/
│   ├── zeus/
│   ├── wannacry/
│   └── ... (454 families)
└── test/
    ├── emotet/
    ├── zeus/
    ├── wannacry/
    └── ... (454 families)
```

### Step 2: Train Family Classifier

```bash
# Train CNN for family classification
python train_family_classifier.py \\
    --dataset_dir motif_family_dataset/ \\
    --epochs 50 \\
    --batch_size 32 \\
    --lr 0.001 \\
    --use_weighted_loss \\
    --save_model
```

### Step 3: Evaluate Results

The training script provides:
- **Top-1 Accuracy:** Exact family match
- **Top-5 Accuracy:** Correct family in top 5 predictions
- **Per-family metrics:** Precision, recall, F1-score
- **Training visualizations:** Loss and accuracy curves

## Dataset Information File

The generated `dataset_info.json` contains:

```json
{
  "dataset_name": "MOTIF_Malware_Family_Images",
  "dataset_type": "multi_class_family_classification",
  "num_classes": 454,
  "classes": {
    "0": "emotet",
    "1": "zeus",
    "2": "wannacry",
    ...
  },
  "family_statistics": {
    "emotet": {"train": 45, "test": 12, "total": 57},
    "zeus": {"train": 32, "test": 8, "total": 40},
    ...
  }
}
```

## Architecture Details

### CNN Architecture for 454-Class Classification

```python
# Optimized for multi-class classification
- Input: 256×256 grayscale images
- 4 convolutional blocks with batch normalization
- Progressive feature maps: 64 → 128 → 256 → 512
- 3-layer classifier: 2048 → 1024 → 454
- Dropout regularization and batch normalization
```

### Handling Class Imbalance

The MOTIF dataset has significant class imbalance:
- Some families have 50+ samples
- Others have only 1-2 samples
- **Solution:** Weighted loss function and class balancing

## Performance Expectations

### Realistic Expectations for 454-Class Classification:
- **Top-1 Accuracy:** 15-30% (challenging with 454 classes)
- **Top-5 Accuracy:** 35-60% (more practical metric)
- **Top-10 Accuracy:** 50-75% (useful for threat analysis)

### Why Multi-Class is Challenging:
1. **High number of classes** (454 families)
2. **Class imbalance** (some families have few samples)
3. **Visual similarity** between some families
4. **Limited training data** per family

## Comparison with Binary Classification

| Aspect | Binary (Old) | Multi-Class Family (New) |
|--------|-------------|--------------------------|
| Classes | 2 (malware/benign) | 454 (malware families) |
| Samples | 3,095 malware, 0 benign | 3,095 across 454 families |
| Accuracy | ~95%+ expected | ~20-30% top-1, ~50-60% top-5 |
| Use Case | Malware detection | Malware family attribution |
| Real-world Value | Limited (no benign samples) | High (threat intelligence) |

## Research Applications

This multi-class setup enables:

1. **Malware Family Attribution:** Identify which threat group created malware
2. **Campaign Tracking:** Track evolution of malware families
3. **Threat Intelligence:** Link attacks to specific groups
4. **Zero-day Detection:** Identify variants of known families

## Files Overview

- **`generate_image.py`** - Updated to support 454 families
- **`train_family_classifier.py`** - Multi-class CNN training
- **`dataset_loader.py`** - PyTorch dataset loader (unchanged)
- **`train_cnn_example.py`** - Original binary classifier
- **`README.md`** - This documentation

## Troubleshooting

### Common Issues:

1. **"No MOTIF-specific structure found"**
   - Ensure you have `motif_dataset.jsonl` and `MOTIF_defanged/` directory
   - Check file paths in the command line arguments

2. **"Dataset type not supported"**
   - Regenerate dataset with updated `generate_image.py`
   - Delete old `dataset_info.json` if switching from binary to multi-class

3. **Out of memory during training**
   - Reduce batch size: `--batch_size 16` or `--batch_size 8`
   - Reduce image size: `--image_size 128`

4. **Low accuracy**
   - This is expected for 454-class classification
   - Focus on top-5 or top-10 accuracy
   - Consider using weighted loss: `--use_weighted_loss`

## Next Steps

1. **Download MOTIF Dataset:**
   ```bash
   git lfs clone https://github.com/boozallen/MOTIF.git
   ```

2. **Extract the dataset:**
   ```bash
   7z x MOTIF.7z  # Password: i_assume_all_risk_opening_malware
   ```

3. **Generate multi-class dataset:**
   ```bash
   python generate_image.py --data_dir MOTIF/ --output_dir motif_family_dataset/
   ```

4. **Train family classifier:**
   ```bash
   python train_family_classifier.py --dataset_dir motif_family_dataset/ --epochs 50 --use_weighted_loss --save_model
   ```

## References

- [MOTIF Dataset Paper](https://arxiv.org/abs/2111.15031)
- [MOTIF GitHub Repository](https://github.com/boozallen/MOTIF)
- [Nataraj et al. Malware Images Paper](https://ieeexplore.ieee.org/document/6016774)