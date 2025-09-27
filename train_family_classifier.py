#!/usr/bin/env python3
"""
MOTIF Multi-Class Family Classification Example
==============================================

This script demonstrates how to train a CNN for multi-class malware family classification
using the updated MOTIF dataset generator that supports 454 malware families.

Key differences from binary classification:
- Supports 454 malware families instead of malware/benign binary classification
- Uses family-specific directories and labels
- Includes top-k accuracy evaluation for multi-class scenarios
- Handles class imbalance through weighted loss and sampling strategies

Usage:
    python train_family_classifier.py --dataset_dir motif_family_dataset/ --epochs 50
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import json
import logging
from collections import Counter

# Import custom dataset loader
from dataset_loader import MOTIFDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FamilyClassifierCNN(nn.Module):
    """
    CNN architecture optimized for multi-class malware family classification.
    
    Design considerations for 454-class classification:
    - Deeper network to handle more complex feature extraction
    - Dropout regularization to prevent overfitting
    - Batch normalization for stable training
    - Larger final layer to support 454 classes
    """
    
    def __init__(self, num_classes=454, input_size=256, dropout_rate=0.5):
        super(FamilyClassifierCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256 -> 128
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
        )
        
        # Calculate the size of flattened features
        feature_size = 512 * (input_size // 16) * (input_size // 16)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_size, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),  # Lower dropout for final layer
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def create_data_transforms(image_size=256):
    """
    Create data transformations with light augmentation for family classification.
    
    Note: We use minimal augmentation to preserve the binary structure visualization
    while adding some variety to help with generalization across families.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # Light augmentation that preserves binary patterns
        transforms.RandomRotation(degrees=5),  # Small rotation
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translation
        transforms.ToTensor(),
        # Optional normalization - can help with training stability
        transforms.Normalize([0.485], [0.229])  # ImageNet single-channel stats
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    
    return train_transform, val_transform


def calculate_class_weights(dataset_dir: Path, dataset_info: dict) -> torch.Tensor:
    """Calculate class weights for handling imbalanced dataset."""
    
    # Count samples per family from dataset info
    family_counts = {}
    split_info = dataset_info.get("split_info", {})
    train_info = split_info.get("train", {}).get("families", {})
    
    for family_name, family_data in train_info.items():
        count = family_data.get("count", 0)
        class_id = int(family_data.get("class_id", 0))
        family_counts[class_id] = count
    
    # Calculate weights (inverse frequency)
    num_classes = len(family_counts)
    total_samples = sum(family_counts.values())
    
    weights = torch.zeros(num_classes)
    for class_id, count in family_counts.items():
        if count > 0:
            weights[class_id] = total_samples / (num_classes * count)
        else:
            weights[class_id] = 1.0  # Fallback for empty classes
    
    logger.info(f"Calculated class weights for {num_classes} families")
    logger.info(f"Weight range: {weights.min():.4f} to {weights.max():.4f}")
    
    return weights


def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    """Train for one epoch with multi-class metrics."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    top5_correct = 0
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Calculate top-5 accuracy if we have enough classes
        if num_classes >= 5:
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()
        
        if batch_idx % 50 == 0:
            logger.info(f'Batch {batch_idx}/{len(dataloader)}, '
                       f'Loss: {loss.item():.4f}, '
                       f'Acc: {100.*correct/total:.2f}%')
    
    top1_acc = 100. * correct / total
    top5_acc = 100. * top5_correct / total if num_classes >= 5 else 0
    
    return total_loss / len(dataloader), top1_acc, top5_acc


def validate_epoch(model, dataloader, criterion, device, num_classes):
    """Validate for one epoch with multi-class metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    top5_correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Calculate top-5 accuracy
            if num_classes >= 5:
                _, top5_pred = outputs.topk(5, 1, True, True)
                top5_correct += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    top1_acc = 100. * correct / total
    top5_acc = 100. * top5_correct / total if num_classes >= 5 else 0
    
    return total_loss / len(dataloader), top1_acc, top5_acc, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description='Train CNN for MOTIF family classification')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to the generated family dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    parser.add_argument('--save_model', action='store_true',
                       help='Save the trained model')
    parser.add_argument('--use_weighted_loss', action='store_true',
                       help='Use weighted loss for class imbalance')
    
    args = parser.parse_args()
    
    # Load dataset info
    dataset_path = Path(args.dataset_dir)
    info_path = dataset_path / 'dataset_info.json'
    
    if not info_path.exists():
        logger.error(f"Dataset info file not found: {info_path}")
        logger.error("Please run generate_image.py with the updated multi-class version first")
        sys.exit(1)
    
    with open(info_path, 'r') as f:
        dataset_info = json.load(f)
    
    # Check if this is a multi-class family dataset
    dataset_type = dataset_info.get("dataset_type", "unknown")
    if dataset_type != "multi_class_family_classification":
        logger.error(f"Dataset type '{dataset_type}' is not supported by this script")
        logger.error("Please regenerate the dataset with the updated generate_image.py script")
        sys.exit(1)
    
    num_classes = dataset_info.get("num_classes", 2)
    logger.info(f"Training multi-class classifier with {num_classes} malware families")
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Create data transforms
    train_transform, val_transform = create_data_transforms(args.image_size)
    
    # Create datasets
    train_dataset = MOTIFDataset(args.dataset_dir, split='train', transform=train_transform)
    val_dataset = MOTIFDataset(args.dataset_dir, split='test', transform=val_transform)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Calculate class weights for imbalanced dataset
    if args.use_weighted_loss:
        class_weights = calculate_class_weights(dataset_path, dataset_info)
        class_weights = class_weights.to(device)
    else:
        class_weights = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4 if torch.cuda.is_available() else 2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4 if torch.cuda.is_available() else 2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    model = FamilyClassifierCNN(
        num_classes=num_classes, 
        input_size=args.image_size,
        dropout_rate=0.5
    ).to(device)
    
    # Create loss function and optimizer
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using weighted cross-entropy loss")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard cross-entropy loss")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    logger.info("Starting training...")
    train_losses = []
    train_accs = []
    train_top5_accs = []
    val_losses = []
    val_accs = []
    val_top5_accs = []
    
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        logger.info(f"\\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 50)
        
        # Training
        train_loss, train_acc, train_top5_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, num_classes
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_top5_accs.append(train_top5_acc)
        
        # Validation
        val_loss, val_acc, val_top5_acc, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device, num_classes
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_top5_accs.append(val_top5_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if num_classes >= 5:
            logger.info(f"Train Top-5 Acc: {train_top5_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if num_classes >= 5:
            logger.info(f"Val Top-5 Acc: {val_top5_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'num_classes': num_classes,
                }, 'best_family_classifier.pth')
    
    # Final evaluation
    logger.info("\\n" + "="*50)
    logger.info("FINAL EVALUATION")
    logger.info("="*50)
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Print classification report for top families only (to avoid clutter)
    class_names = [dataset_info["classes"][str(i)] for i in range(num_classes)]
    
    # Show results for families with samples in validation set
    unique_targets = sorted(list(set(val_targets)))
    active_class_names = [class_names[i] for i in unique_targets]
    
    print("\\nClassification Report (families with validation samples):")
    print(classification_report(
        val_targets, val_preds,
        labels=unique_targets,
        target_names=active_class_names,
        zero_division=0
    ))
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Top-1 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    if num_classes >= 5:
        plt.subplot(1, 3, 3)
        plt.plot(train_top5_accs, label='Train')
        plt.plot(val_top5_accs, label='Validation')
        plt.title('Top-5 Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('family_classification_history.png', dpi=150, bbox_inches='tight')
    logger.info("Training history saved to family_classification_history.png")
    
    # Save final model
    if args.save_model:
        final_model_path = 'final_family_classifier.pth'
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_accs[-1],
            'val_acc': val_accs[-1],
            'num_classes': num_classes,
            'dataset_info': dataset_info
        }, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()