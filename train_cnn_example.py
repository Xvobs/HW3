#!/usr/bin/env python3
"""
CNN Training Example with Nataraj Malware Images
===============================================

This script demonstrates how to use the generated malware images (following Nataraj methodology)
to train a Convolutional Neural Network for malware classification.

This serves as a starting point for your malware classification experiments.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Import custom dataset loader
from dataset_loader import MalwareDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MalwareCNN(nn.Module):
    """
    Simple CNN architecture for malware classification using Nataraj images.
    
    This is a basic architecture - you should experiment with:
    - Different architectures (ResNet, EfficientNet, etc.)
    - Transfer learning from ImageNet pre-trained models
    - Data augmentation techniques
    - Different optimization strategies
    """
    
    def __init__(self, num_classes=2, input_size=256):
        super(MalwareCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1 channel for grayscale
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256 -> 128
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16
        )
        
        # Calculate the size of flattened features
        feature_size = 256 * (input_size // 16) * (input_size // 16)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def create_data_transforms(image_size=256):
    """
    Create data transformations for training and validation.
    
    Note: For Nataraj images, we keep transformations minimal to preserve
    the original binary structure visualization.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # Note: No normalization as we want to preserve the 0-255 range from bytes
        # transforms.Normalize([0.5], [0.5])  # Uncomment if you want normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])  # Uncomment if you want normalization
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 50 == 0:
            logger.info(f'Batch {batch_idx}/{len(dataloader)}, '
                       f'Loss: {loss.item():.4f}, '
                       f'Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(dataloader), 100. * correct / total


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
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
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return (total_loss / len(dataloader), 
            100. * correct / total,
            all_preds, 
            all_targets)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train CNN on Nataraj malware images')
    parser.add_argument('--dataset_dir', required=True, help='Path to generated image dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    
    args = parser.parse_args()
    
    # Check if dataset directory exists
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data transforms
    train_transform, val_transform = create_data_transforms(args.image_size)
    
    # Datasets and dataloaders
    logger.info("Loading datasets...")
    train_dataset = MalwareDataset(
        dataset_path / 'train', 
        transform=train_transform
    )
    val_dataset = MalwareDataset(
        dataset_path / 'test', 
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Model, loss, optimizer
    model = MalwareCNN(num_classes=2, input_size=args.image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(args.epochs):
        logger.info(f"\\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Final evaluation
    logger.info("\\n" + "="*50)
    logger.info("FINAL EVALUATION")
    logger.info("="*50)
    
    print("\\nClassification Report:")
    print(classification_report(val_targets, val_preds, 
                              target_names=['Benign', 'Malware']))
    
    print("\\nConfusion Matrix:")
    print(confusion_matrix(val_targets, val_preds))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    logger.info("Training history saved to training_history.png")
    plt.show()
    
    # Save model
    if args.save_model:
        model_path = 'malware_cnn_model.pth'
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_accs[-1],
            'val_acc': val_accs[-1],
        }, model_path)
        logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()