#!/usr/bin/env python3
"""
Variable Size CNN Models for MOTIF Dataset
==========================================

This module provides CNN architectures optimized for variable-sized malware images
following the original Nataraj et al. approach.

Key features:
1. Adaptive pooling for handling variable input sizes
2. Models that preserve spatial relationships in binary data
3. Architectures inspired by the original malware visualization research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class NatarajStyleCNN(nn.Module):
    """
    Smaller, faster CNN for variable-size malware images.
    """
    
    def __init__(self, 
                 num_classes: int = 502,
                 input_channels: int = 1,
                 adaptive_pool_size: Tuple[int, int] = (4, 4),
                 dropout_rate: float = 0.2):
        """
        Args:
            num_classes: Number of malware families
            input_channels: Number of input channels (1 for grayscale)
            adaptive_pool_size: Size for adaptive pooling layer
            dropout_rate: Dropout rate for regularization
        """
        super(NatarajStyleCNN, self).__init__()
        
        self.adaptive_pool_size = adaptive_pool_size
        
        # Feature extraction layers - designed for binary pattern recognition
        self.features = nn.Sequential(
            # Block 1: Initial feature detection
            # Small kernels to capture local byte patterns
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: Medium-scale pattern detection
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: Larger pattern detection
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_pool_size)
        
        # Calculate classifier input size
        classifier_input_size = 64 * adaptive_pool_size[0] * adaptive_pool_size[1]
        
        # Classifier for malware family classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_size, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using best practices for binary pattern recognition."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VariableSizeCNN(nn.Module):
    """
    Smaller, faster variable-size CNN.
    """
    
    def __init__(self, 
                 num_classes: int = 502,
                 input_channels: int = 1,
                 base_features: int = 16):
        """
        Args:
            num_classes: Number of malware families
            input_channels: Number of input channels
            base_features: Base number of feature channels
        """
        super(VariableSizeCNN, self).__init__()
        
        # Feature extraction with adaptive behavior
        self.conv1 = nn.Conv2d(input_channels, base_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_features)
        
        self.conv2 = nn.Conv2d(base_features, base_features * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_features * 2)
        
        self.conv3 = nn.Conv2d(base_features * 2, base_features * 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_features * 4)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier using combined global features
        final_features = base_features * 4 * 2  # avg + max pooling
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Global pooling - handles any remaining spatial size
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        
        # Combine average and max pooling features
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.classifier(x)
        
        return x


class PyramidPoolingCNN(nn.Module):
    """
    CNN with pyramid pooling to capture multi-scale features from variable sized images.
    Inspired by PSPNet but adapted for malware binary patterns.
    """
    
    def __init__(self, 
                 num_classes: int = 502,
                 input_channels: int = 1,
                 pyramid_levels: List[int] = [1, 2, 3, 6]):
        """
        Args:
            num_classes: Number of malware families
            input_channels: Number of input channels
            pyramid_levels: Sizes for pyramid pooling
        """
        super(PyramidPoolingCNN, self).__init__()
        
        self.pyramid_levels = pyramid_levels
        
        # Backbone feature extractor
        self.backbone = nn.Sequential(
            # Initial layers
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Residual-like blocks for binary pattern extraction
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Pyramid pooling modules
        self.pyramid_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(level) for level in pyramid_levels
        ])
        
        # Feature fusion
        pyramid_features = 512 * len(pyramid_levels)
        self.fusion = nn.Sequential(
            nn.Conv2d(pyramid_features, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pyramid pooling."""
        
        # Extract backbone features
        features = self.backbone(x)
        
        # Pyramid pooling
        h, w = features.size(2), features.size(3)
        pyramid_features = []
        
        for pool in self.pyramid_pools:
            pooled = pool(features)
            # Upsample back to original feature map size
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)
        
        # Concatenate pyramid features
        pyramid_concat = torch.cat(pyramid_features, dim=1)
        
        # Fuse features
        fused = self.fusion(pyramid_concat)
        
        # Global pooling and classification
        x = self.global_pool(fused).view(fused.size(0), -1)
        x = self.classifier(x)
        
        return x


def create_variable_size_model(model_type: str = 'nataraj',
                             num_classes: int = 502,
                             **kwargs):
    """
    Create a model optimized for variable-sized malware images.
    
    Args:
        model_type: Type of model to create:
            - 'nataraj': NatarajStyleCNN (closest to original paper)
            - 'variable': VariableSizeCNN (true variable size handling)
            - 'pyramid': PyramidPoolingCNN (multi-scale features)
        num_classes: Number of malware families
        **kwargs: Additional model parameters
        
    Returns:
        PyTorch model
    """
    
    if model_type == 'nataraj':
        return NatarajStyleCNN(
            num_classes=num_classes,
            adaptive_pool_size=kwargs.get('adaptive_pool_size', (8, 8)),
            dropout_rate=kwargs.get('dropout_rate', 0.5)
        )
    
    elif model_type == 'variable':
        return VariableSizeCNN(
            num_classes=num_classes,
            base_features=kwargs.get('base_features', 64)
        )
    
    elif model_type == 'pyramid':
        return PyramidPoolingCNN(
            num_classes=num_classes,
            pyramid_levels=kwargs.get('pyramid_levels', [1, 2, 3, 6])
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Utility functions
def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model):
    """Get detailed information about a model."""
    total_params = count_parameters(model)
    
    # Calculate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': total_params,
        'model_size_mb': size_mb,
        'parameter_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024
    }