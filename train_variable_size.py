#!/usr/bin/env python3
"""
Variable Size Image Training for MOTIF Dataset
==============================================

This script demonstrates how to train CNNs on variable-sized malware images
following the original Nataraj et al. approach as closely as possible.

Key approaches:
1. Preserve variable sizes with adaptive pooling
2. Intelligent padding/cropping strategies  
3. Multi-scale training approaches
4. True variable size handling with global pooling

Usage:
    # Train with adaptive padding (recommended)
    python train_variable_size.py --dataset_dir ./motif_images_variable_size --approach adaptive_pad
    
    # Train with preserved variable sizes
    python train_variable_size.py --dataset_dir ./motif_images_variable_size --approach preserve --model_type variable
    
    # Multi-scale training
    python train_variable_size.py --dataset_dir ./motif_images_variable_size --approach multi_scale
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
import time
from collections import defaultdict

# Import our custom modules
from dataset_loader import MOTIFDataset
from variable_size_transforms import get_variable_size_transforms, create_nataraj_transforms
from variable_size_models import create_variable_size_model, get_model_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

class VariableSizeDataLoader:
    """Custom data loader for variable sized images."""
    
    def __init__(self, dataset, batch_size=16, shuffle=True, num_workers=4):
        """
        Initialize variable size data loader.
        
        For true variable size training, we need to handle batching carefully
        since images have different dimensions.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Check if we're dealing with true variable sizes
        sample_img, _ = dataset[0]
        self.variable_size = len(sample_img.shape) == 3 and sample_img.shape[1:] != sample_img.shape[1:]
        
        if self.variable_size:
            logger.info("Using variable size batching - will pad batches dynamically")
            self.dataloader = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self._variable_collate_fn,
                pin_memory=True
            )
        else:
            # Standard data loader for fixed-size images
            self.dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle, 
                num_workers=num_workers,
                pin_memory=True
            )
    
    def _variable_collate_fn(self, batch):
        """Custom collate function for variable sized images."""
        images, labels = zip(*batch)
        
        # Find maximum dimensions in batch
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        
        # Pad all images to max dimensions
        padded_images = []
        for img in images:
            c, h, w = img.shape
            pad_h = max_h - h
            pad_w = max_w - w
            
            # Pad with zeros (black pixels)
            padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)
            padded_images.append(padded)
        
        # Stack into batch tensor
        batch_images = torch.stack(padded_images, dim=0)
        batch_labels = torch.tensor(labels)
        
        return batch_images, batch_labels
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def train_epoch_variable(model, dataloader, criterion, optimizer, device, log_interval=50):
    """Training epoch for variable size images."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping for stability with variable sizes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % log_interval == 0:
            logger.info(f'Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.6f} '
                       f'Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate_epoch_variable(model, dataloader, criterion, device):
    """Validation epoch for variable size images."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def analyze_image_sizes(dataset_dir):
    """Analyze the distribution of image sizes in the dataset."""
    logger.info("Analyzing image sizes in dataset...")
    
    from PIL import Image
    sizes = []
    aspects = []
    
    dataset_path = Path(dataset_dir)
    
    # Sample images from train and test
    for split in ['train', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
            
        # Sample a few images from each family
        count = 0
        for family_dir in split_dir.iterdir():
            if not family_dir.is_dir():
                continue
                
            for img_file in list(family_dir.glob('*.png'))[:5]:  # Sample 5 per family
                try:
                    with Image.open(img_file) as img:
                        w, h = img.size
                        sizes.append((w, h))
                        aspects.append(w / h)
                        count += 1
                        
                        if count >= 500:  # Limit total samples for speed
                            break
                except:
                    continue
            
            if count >= 500:
                break
        
        if count >= 500:
            break
    
    if sizes:
        widths, heights = zip(*sizes)
        
        stats = {
            'count': len(sizes),
            'width_stats': {
                'min': min(widths),
                'max': max(widths),
                'mean': np.mean(widths),
                'median': np.median(widths)
            },
            'height_stats': {
                'min': min(heights),
                'max': max(heights),
                'mean': np.mean(heights),
                'median': np.median(heights)
            },
            'aspect_ratio_stats': {
                'min': min(aspects),
                'max': max(aspects),
                'mean': np.mean(aspects),
                'median': np.median(aspects)
            }
        }
        
        logger.info(f"Image size analysis ({stats['count']} samples):")
        logger.info(f"  Width: {stats['width_stats']['min']}-{stats['width_stats']['max']} "
                   f"(mean: {stats['width_stats']['mean']:.1f})")
        logger.info(f"  Height: {stats['height_stats']['min']}-{stats['height_stats']['max']} "
                   f"(mean: {stats['height_stats']['mean']:.1f})")
        logger.info(f"  Aspect ratio: {stats['aspect_ratio_stats']['min']:.2f}-{stats['aspect_ratio_stats']['max']:.2f} "
                   f"(mean: {stats['aspect_ratio_stats']['mean']:.2f})")
        
        return stats
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Train CNN on variable-sized malware images')
    
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to the variable size dataset directory')
    parser.add_argument('--approach', type=str, default='adaptive_pad',
                       choices=['preserve', 'adaptive_pad', 'center_crop', 'resize', 'multi_scale'],
                       help='Approach for handling variable sizes')
    parser.add_argument('--model_type', type=str, default='nataraj',
                       choices=['nataraj', 'variable', 'pyramid'],
                       help='Type of CNN model to use')
    parser.add_argument('--target_size', type=int, default=256,
                       help='Target size for fixed-size approaches')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--analyze_sizes', action='store_true',
                       help='Analyze image size distribution before training')
    
    args = parser.parse_args()
    
    # Load dataset info
    dataset_path = Path(args.dataset_dir)
    info_path = dataset_path / 'dataset_info.json'
    
    if not info_path.exists():
        logger.error(f"Dataset info file not found: {info_path}")
        sys.exit(1)
    
    with open(info_path, 'r') as f:
        dataset_info = json.load(f)
    
    num_classes = dataset_info.get("num_classes", 502)
    logger.info(f"Training with {num_classes} malware families")
    
    # Analyze image sizes if requested
    if args.analyze_sizes:
        size_stats = analyze_image_sizes(args.dataset_dir)
    
    # Set up device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Create transforms based on approach
    logger.info(f"Using {args.approach} approach for variable size handling")
    
    if args.approach == 'preserve':
        # True variable size training
        train_transform = get_variable_size_transforms(
            approach='preserve',
            training=True,
            max_size=1024,
            min_size=64
        )
        val_transform = get_variable_size_transforms(
            approach='preserve', 
            training=False,
            max_size=1024,
            min_size=64
        )
        # Use smaller batch size for variable sizes
        if args.batch_size > 16:
            args.batch_size = 16
            logger.info("Reduced batch size to 16 for variable size training")
            
    else:
        # Fixed size approaches
        train_transform = get_variable_size_transforms(
            approach=args.approach,
            target_size=args.target_size,
            training=True
        )
        val_transform = get_variable_size_transforms(
            approach=args.approach,
            target_size=args.target_size,
            training=False
        )
    
    # Create datasets
    train_dataset = MOTIFDataset(
        args.dataset_dir, 
        split='train', 
        transform=train_transform
    )
    
    val_dataset = MOTIFDataset(
        args.dataset_dir, 
        split='test', 
        transform=val_transform
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    if args.approach == 'preserve':
        train_loader = VariableSizeDataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        val_loader = VariableSizeDataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # Create model
    logger.info(f"Creating {args.model_type} model")
    model = create_variable_size_model(
        model_type=args.model_type,
        num_classes=num_classes
    ).to(device)
    
    # Print model info
    model_info = get_model_info(model)
    logger.info(f"Model parameters: {model_info['total_parameters']:,}")
    logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 50)
        
        # Training
        train_loss, train_acc = train_epoch_variable(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation
        val_loss, val_acc = validate_epoch_variable(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'approach': args.approach,
                'model_type': args.model_type
            }, f'best_variable_size_{args.approach}_{args.model_type}.pth')
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")
        logger.info(f"Epoch time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time/60:.2f} minutes")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'Loss - {args.approach} approach')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'Accuracy - {args.approach} approach')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'variable_size_training_{args.approach}_{args.model_type}.png', 
                dpi=150, bbox_inches='tight')
    logger.info(f"Training history saved to variable_size_training_{args.approach}_{args.model_type}.png")


if __name__ == "__main__":
    main()