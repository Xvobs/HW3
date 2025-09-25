#!/usr/bin/env python3
"""
MOTIF Dataset Loader for Machine Learning
==========================================

This module provides a PyTorch DataLoader for the generated MOTIF malware image dataset.
Use this in your machine learning training scripts.

Usage:
    from dataset_loader import MOTIFDataset, get_data_loaders
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders('./motif_dataset', batch_size=32)
    
    # Use in training loop
    for images, labels in train_loader:
        # Your training code here
        pass
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, Any
import logging

# Try to import PyTorch, fall back to basic functionality if not available
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Some functionality will be limited.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MOTIFDataset:
    """Dataset class for MOTIF malware images."""
    
    def __init__(self, 
                 dataset_dir: str, 
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize MOTIF dataset.
        
        Args:
            dataset_dir: Path to dataset directory
            split: 'train' or 'test'
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load dataset info
        info_path = self.dataset_dir / "dataset_info.json"
        self.dataset_info = {}
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.dataset_info = json.load(f)
        
        # Define class mapping
        self.class_to_idx = {"benign": 0, "malware": 1}
        self.idx_to_class = {0: "benign", 1: "malware"}
        
        # Load file paths and labels
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from {split} split")
    
    def _load_samples(self) -> list:
        """Load all sample file paths and labels."""
        samples = []
        split_dir = self.dataset_dir / self.split
        
        for class_name in ["benign", "malware"]:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            label = self.class_to_idx[class_name]
            
            # Get all PNG files in the class directory
            for img_path in class_dir.glob("*.png"):
                samples.append((str(img_path), label))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Get a sample from the dataset."""
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image_size = self.dataset_info.get('image_size', 256)
            image = Image.new('L', (image_size, image_size), 0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count of samples per class."""
        counts = {"benign": 0, "malware": 0}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            counts[class_name] += 1
        return counts


def get_default_transforms(image_size: int = 256, normalize: bool = True) -> Dict[str, Callable]:
    """Get default transforms for the dataset."""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available for transforms")
        return {"train": None, "test": None}
    
    # Basic transforms
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]
    
    if normalize:
        # Normalize to [0, 1] range for grayscale images
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    
    basic_transform = transforms.Compose(transform_list)
    
    # Training transforms with augmentation
    train_transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor()
    ]
    
    if normalize:
        train_transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    
    train_transform = transforms.Compose(train_transform_list)
    
    return {
        "train": train_transform,
        "test": basic_transform
    }


def get_data_loaders(dataset_dir: str, 
                    batch_size: int = 32,
                    image_size: int = 256,
                    num_workers: int = 4,
                    normalize: bool = True,
                    pin_memory: bool = True,
                    shuffle_train: bool = True) -> Tuple[Optional['DataLoader'], Optional['DataLoader']]:
    """
    Get PyTorch DataLoaders for train and test splits.
    
    Args:
        dataset_dir: Path to dataset directory
        batch_size: Batch size for data loaders
        image_size: Size to resize images to
        num_workers: Number of worker processes for data loading
        normalize: Whether to normalize images
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available. Cannot create DataLoaders.")
        return None, None
    
    # Get transforms
    transforms_dict = get_default_transforms(image_size, normalize)
    
    # Create datasets
    train_dataset = MOTIFDataset(
        dataset_dir=dataset_dir,
        split='train',
        transform=transforms_dict['train']
    )
    
    test_dataset = MOTIFDataset(
        dataset_dir=dataset_dir,
        split='test', 
        transform=transforms_dict['test']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Print dataset info
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Testing samples: {len(test_dataset)}")
    logger.info(f"Training class distribution: {train_dataset.get_class_counts()}")
    logger.info(f"Testing class distribution: {test_dataset.get_class_counts()}")
    
    return train_loader, test_loader


def analyze_dataset_balance(dataset_dir: str) -> Dict[str, Any]:
    """Analyze class balance in the dataset."""
    results = {}
    
    for split in ['train', 'test']:
        try:
            dataset = MOTIFDataset(dataset_dir, split=split)
            counts = dataset.get_class_counts()
            total = sum(counts.values())
            
            results[split] = {
                'counts': counts,
                'total': total,
                'balance_ratio': counts['malware'] / counts['benign'] if counts['benign'] > 0 else float('inf'),
                'malware_percentage': (counts['malware'] / total * 100) if total > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing {split} split: {e}")
            results[split] = {'error': str(e)}
    
    return results


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MOTIF dataset loader")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--test_loading', action='store_true', help='Test data loading')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        logger.error(f"Dataset directory not found: {args.dataset_dir}")
        exit(1)
    
    # Analyze dataset balance
    balance_analysis = analyze_dataset_balance(args.dataset_dir)
    
    print("\n" + "="*50)
    print("DATASET BALANCE ANALYSIS")
    print("="*50)
    
    for split, info in balance_analysis.items():
        if 'error' in info:
            print(f"{split.capitalize()} split: Error - {info['error']}")
        else:
            print(f"{split.capitalize()} split:")
            print(f"  Total samples: {info['total']}")
            print(f"  Malware: {info['counts']['malware']} ({info['malware_percentage']:.1f}%)")
            print(f"  Benign: {info['counts']['benign']} ({100-info['malware_percentage']:.1f}%)")
            print(f"  Balance ratio (malware/benign): {info['balance_ratio']:.2f}")
            print()
    
    if args.test_loading and TORCH_AVAILABLE:
        print("Testing data loading...")
        
        try:
            train_loader, test_loader = get_data_loaders(args.dataset_dir, batch_size=args.batch_size)
            
            if train_loader and test_loader:
                # Test loading a batch
                train_batch = next(iter(train_loader))
                test_batch = next(iter(test_loader))
                
                print(f"✅ Successfully loaded batches:")
                print(f"  Train batch shape: {train_batch[0].shape}")
                print(f"  Test batch shape: {test_batch[0].shape}")
                print(f"  Train labels shape: {train_batch[1].shape}")
                print(f"  Test labels shape: {test_batch[1].shape}")
                
                # Show some label statistics
                train_labels = train_batch[1].numpy()
                test_labels = test_batch[1].numpy()
                
                print(f"  Train batch - Malware: {(train_labels == 1).sum()}, Benign: {(train_labels == 0).sum()}")
                print(f"  Test batch - Malware: {(test_labels == 1).sum()}, Benign: {(test_labels == 0).sum()}")
            else:
                print("❌ Failed to create data loaders")
                
        except Exception as e:
            logger.error(f"Error testing data loading: {e}")
    elif args.test_loading:
        print("⚠️  PyTorch not available. Cannot test data loading.")