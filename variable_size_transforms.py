#!/usr/bin/env python3
"""
Variable Size Image Transforms for MOTIF Dataset
===============================================

This module provides transforms for handling variable-sized malware images
following the original Nataraj et al. approach as closely as possible.

The original Nataraj approach:
1. Converts binary files to images with dimensions based on file size
2. Uses square or near-square images (width = ceil(sqrt(file_size)))
3. Preserves spatial locality of bytes in the binary file
4. Each byte maps directly to pixel intensity (0-255)

For CNN training, we provide several strategies to handle variable sizes:
1. Adaptive pooling to fixed feature maps
2. Padding/cropping to standard sizes
3. Multi-scale training approaches
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import random
from typing import Tuple, Optional, Union, List
import math


class VariableSizePreservingTransform:
    """
    Transform that preserves the variable size nature of Nataraj images
    while making them compatible with CNN training.
    
    This follows the original approach most closely by maintaining
    the natural dimensions determined by file size.
    """
    
    def __init__(self, 
                 max_size: int = 1024,
                 min_size: int = 64,
                 normalize: bool = True,
                 augment: bool = False):
        """
        Args:
            max_size: Maximum dimension for images (for memory constraints)
            min_size: Minimum dimension for images (for meaningful features)
            normalize: Whether to normalize pixel values
            augment: Whether to apply minimal augmentations
        """
        self.max_size = max_size
        self.min_size = min_size
        self.normalize = normalize
        self.augment = augment
        
        # Minimal augmentations that preserve binary structure
        if augment:
            self.augment_transforms = transforms.Compose([
                # Very subtle noise to simulate minor byte variations
                transforms.Lambda(self._add_subtle_noise),
                # Minimal rotation (max 1 degree) to add robustness
                transforms.RandomRotation(degrees=1, fill=0),
            ])
    
    def _add_subtle_noise(self, img):
        """Add very subtle noise to simulate minor variations in binary data."""
        if random.random() < 0.1:  # Apply noise rarely
            img_array = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 1, img_array.shape)  # Very low noise
            img_array = np.clip(img_array + noise, 0, 255)
            return Image.fromarray(img_array.astype(np.uint8), mode='L')
        return img
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Transform image while preserving variable size characteristics.
        
        Args:
            img: PIL Image (grayscale)
            
        Returns:
            Tensor with shape [1, H, W] where H and W are variable
        """
        # Ensure grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Get original dimensions
        width, height = img.size
        
        # Clamp dimensions to reasonable bounds
        if width > self.max_size or height > self.max_size:
            # Resize keeping aspect ratio
            img.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)
        
        # Ensure minimum size
        width, height = img.size
        if width < self.min_size or height < self.min_size:
            scale = max(self.min_size / width, self.min_size / height)
            new_width = max(self.min_size, int(width * scale))
            new_height = max(self.min_size, int(height * scale))
            img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
        
        # Apply minimal augmentations if enabled
        if self.augment:
            img = self.augment_transforms(img)
        
        # Convert to tensor
        img_tensor = F.to_tensor(img)
        
        # Normalize to [-1, 1] range (simple normalization preserving byte patterns)
        if self.normalize:
            img_tensor = (img_tensor - 0.5) / 0.5
        
        return img_tensor


class AdaptiveSizeTransform:
    """
    Transform that adapts variable sized images to work with standard CNNs
    while preserving as much of the original structure as possible.
    
    Uses intelligent padding/cropping strategies.
    """
    
    def __init__(self, 
                 target_size: Union[int, Tuple[int, int]] = 256,
                 strategy: str = 'adaptive_pad',
                 normalize: bool = True):
        """
        Args:
            target_size: Target size for output images
            strategy: How to handle size adaptation:
                - 'adaptive_pad': Pad smaller dimension to make square
                - 'center_crop': Crop to target size from center  
                - 'resize': Resize to exact target (may distort aspect ratio)
                - 'pad_resize': Pad to square then resize
            normalize: Whether to normalize pixel values
        """
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
        
        self.strategy = strategy
        self.normalize = normalize
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Transform variable size image to fixed size."""
        
        if img.mode != 'L':
            img = img.convert('L')
        
        if self.strategy == 'adaptive_pad':
            img = self._adaptive_pad(img)
        elif self.strategy == 'center_crop':
            img = self._center_crop(img)
        elif self.strategy == 'resize':
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
        elif self.strategy == 'pad_resize':
            img = self._pad_to_square(img)
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor
        img_tensor = F.to_tensor(img)
        
        if self.normalize:
            img_tensor = (img_tensor - 0.5) / 0.5
        
        return img_tensor
    
    def _adaptive_pad(self, img: Image.Image) -> Image.Image:
        """Pad image to target size, preserving aspect ratio."""
        width, height = img.size
        target_w, target_h = self.target_size
        
        # Calculate padding needed
        pad_w = max(0, target_w - width)
        pad_h = max(0, target_h - height)
        
        # Pad symmetrically
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        
        if pad_w > 0 or pad_h > 0:
            img = F.pad(img, padding, fill=0, padding_mode='constant')
        
        # Crop if image is larger than target
        if width > target_w or height > target_h:
            img = F.center_crop(img, self.target_size)
        
        return img
    
    def _center_crop(self, img: Image.Image) -> Image.Image:
        """Crop image from center to target size."""
        return F.center_crop(img, self.target_size)
    
    def _pad_to_square(self, img: Image.Image) -> Image.Image:
        """Pad image to make it square."""
        width, height = img.size
        max_dim = max(width, height)
        
        pad_w = max_dim - width
        pad_h = max_dim - height
        
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        
        return F.pad(img, padding, fill=0, padding_mode='constant')


class MultiScaleTransform:
    """
    Multi-scale transform that creates multiple scales of the same image
    for multi-scale CNN training (closer to original research approaches).
    """
    
    def __init__(self, 
                 scales: List[int] = [224, 256, 288],
                 random_scale: bool = True,
                 normalize: bool = True):
        """
        Args:
            scales: List of target scales
            random_scale: Whether to randomly select scale during training
            normalize: Whether to normalize pixel values
        """
        self.scales = scales
        self.random_scale = random_scale
        self.normalize = normalize
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Transform to random scale."""
        if img.mode != 'L':
            img = img.convert('L')
        
        # Select scale
        if self.random_scale:
            target_size = random.choice(self.scales)
        else:
            target_size = self.scales[0]
        
        # Resize maintaining aspect ratio, then center crop
        img = self._resize_and_crop(img, target_size)
        
        # Convert to tensor
        img_tensor = F.to_tensor(img)
        
        if self.normalize:
            img_tensor = (img_tensor - 0.5) / 0.5
        
        return img_tensor
    
    def _resize_and_crop(self, img: Image.Image, target_size: int) -> Image.Image:
        """Resize image maintaining aspect ratio, then center crop."""
        width, height = img.size
        
        # Calculate scale to make smaller dimension equal to target_size
        scale = target_size / min(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop to exact target size
        img = F.center_crop(img, (target_size, target_size))
        
        return img


def get_variable_size_transforms(approach: str = 'adaptive_pad',
                               target_size: int = 256,
                               training: bool = True,
                               **kwargs):
    """
    Get transforms for variable size image handling.
    
    Args:
        approach: Approach to handle variable sizes:
            - 'preserve': Preserve variable sizes (requires special CNN)
            - 'adaptive_pad': Intelligently pad/crop to fixed size
            - 'center_crop': Center crop to fixed size
            - 'resize': Resize to fixed size (may distort)
            - 'multi_scale': Multi-scale training
        target_size: Target size for fixed-size approaches
        training: Whether this is for training (enables augmentation)
        **kwargs: Additional arguments for specific transforms
        
    Returns:
        Transform function
    """
    
    if approach == 'preserve':
        return VariableSizePreservingTransform(
            max_size=kwargs.get('max_size', 1024),
            min_size=kwargs.get('min_size', 64),
            augment=training,
            normalize=kwargs.get('normalize', True)
        )
    
    elif approach == 'adaptive_pad':
        return AdaptiveSizeTransform(
            target_size=target_size,
            strategy='adaptive_pad',
            normalize=kwargs.get('normalize', True)
        )
    
    elif approach == 'center_crop':
        return AdaptiveSizeTransform(
            target_size=target_size,
            strategy='center_crop',
            normalize=kwargs.get('normalize', True)
        )
    
    elif approach == 'resize':
        return AdaptiveSizeTransform(
            target_size=target_size,
            strategy='resize',
            normalize=kwargs.get('normalize', True)
        )
    
    elif approach == 'multi_scale':
        scales = kwargs.get('scales', [224, 256, 288])
        return MultiScaleTransform(
            scales=scales,
            random_scale=training,
            normalize=kwargs.get('normalize', True)
        )
    
    else:
        raise ValueError(f"Unknown approach: {approach}")


# Convenience function for creating standard variable size transforms
def create_nataraj_transforms(image_size: int = 256, 
                            preserve_aspect: bool = True,
                            minimal_augment: bool = False):
    """
    Create transforms that follow the Nataraj approach as closely as possible.
    
    Args:
        image_size: Target image size for CNN compatibility
        preserve_aspect: Whether to preserve aspect ratio
        minimal_augment: Whether to apply minimal augmentations
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    
    if preserve_aspect:
        # Use adaptive padding to preserve aspect ratio
        train_transform = get_variable_size_transforms(
            approach='adaptive_pad',
            target_size=image_size,
            training=True,
            normalize=True
        )
        
        val_transform = get_variable_size_transforms(
            approach='adaptive_pad', 
            target_size=image_size,
            training=False,
            normalize=True
        )
    else:
        # Standard resize approach
        train_transform = get_variable_size_transforms(
            approach='resize',
            target_size=image_size,
            training=True,
            normalize=True
        )
        
        val_transform = get_variable_size_transforms(
            approach='resize',
            target_size=image_size, 
            training=False,
            normalize=True
        )
    
    return train_transform, val_transform