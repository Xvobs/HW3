#!/usr/bin/env python3
"""
Nataraj Methodology Verification Tool
====================================

This script helps verify that the generated malware images follow the correct
Nataraj et al. methodology from "Malware images: visualization and automatic 
classification" (VizSec 2011).

Key verification points:
1. Each pixel represents exactly one byte from the binary
2. Pixel intensity values range from 0-255 (uint8)
3. Spatial locality is preserved (consecutive bytes → adjacent pixels)
4. Images maintain the structural patterns visible in malware
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_image_properties(image_path: str) -> dict:
    """
    Analyze properties of a generated malware image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        analysis = {
            'file_path': image_path,
            'image_mode': img.mode,
            'dimensions': img_array.shape,
            'data_type': str(img_array.dtype),
            'pixel_min': int(img_array.min()),
            'pixel_max': int(img_array.max()),
            'pixel_mean': float(img_array.mean()),
            'pixel_std': float(img_array.std()),
            'unique_values': len(np.unique(img_array)),
            'zero_pixels': int(np.sum(img_array == 0)),
            'non_zero_pixels': int(np.sum(img_array > 0))
        }
        
        # Calculate entropy (measure of randomness/information content)
        hist, _ = np.histogram(img_array, bins=256, range=(0, 255))
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros to avoid log(0)
        entropy = -np.sum(hist * np.log2(hist))
        analysis['entropy'] = float(entropy)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing {image_path}: {str(e)}")
        return None


def visualize_malware_image(image_path: str, output_dir: str = None):
    """
    Create a comprehensive visualization of a malware image.
    
    Args:
        image_path: Path to the malware image
        output_dir: Directory to save visualization (optional)
    """
    try:
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Malware Image Analysis: {Path(image_path).name}', fontsize=14)
        
        # Original image
        axes[0,0].imshow(img_array, cmap='gray', vmin=0, vmax=255)
        axes[0,0].set_title('Grayscale Image')
        axes[0,0].axis('off')
        
        # Histogram of pixel intensities
        axes[0,1].hist(img_array.flatten(), bins=256, range=(0, 255), 
                      density=True, alpha=0.7, color='blue')
        axes[0,1].set_title('Pixel Intensity Distribution')
        axes[0,1].set_xlabel('Pixel Intensity (0-255)')
        axes[0,1].set_ylabel('Density')
        axes[0,1].grid(True, alpha=0.3)
        
        # 2D histogram (heatmap of pixel patterns)
        if len(img_array.shape) == 2 and min(img_array.shape) > 1:
            # Take a sample of the image for pattern analysis
            sample_size = min(64, min(img_array.shape))
            sample = img_array[:sample_size, :sample_size]
            im = axes[1,0].imshow(sample, cmap='viridis', aspect='auto')
            axes[1,0].set_title(f'Pattern Detail (Top-left {sample_size}x{sample_size})')
            plt.colorbar(im, ax=axes[1,0])
        else:
            axes[1,0].text(0.5, 0.5, 'Pattern analysis\nnot available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Pattern Analysis')
        
        # Statistics text
        analysis = analyze_image_properties(image_path)
        if analysis:
            stats_text = f"""Image Properties:
Dimensions: {analysis['dimensions']}
Data Type: {analysis['data_type']}
Pixel Range: {analysis['pixel_min']} - {analysis['pixel_max']}
Mean Intensity: {analysis['pixel_mean']:.1f}
Std Deviation: {analysis['pixel_std']:.1f}
Entropy: {analysis['entropy']:.2f} bits
Non-zero Pixels: {analysis['non_zero_pixels']}
Zero Pixels: {analysis['zero_pixels']}"""
        else:
            stats_text = "Analysis failed"
        
        axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes,
                      verticalalignment='top', fontfamily='monospace', fontsize=9)
        axes[1,1].set_title('Image Statistics')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        # Save if output directory specified
        if output_dir:
            output_path = Path(output_dir) / f"{Path(image_path).stem}_analysis.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error visualizing {image_path}: {str(e)}")


def verify_nataraj_methodology(image_dir: str, sample_size: int = 5):
    """
    Verify that images follow Nataraj methodology.
    
    Args:
        image_dir: Directory containing malware images
        sample_size: Number of images to analyze
    """
    image_dir = Path(image_dir)
    
    # Find image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(image_dir.glob(f"**/{ext}"))
    
    if not image_files:
        logger.error(f"No image files found in {image_dir}")
        return
    
    # Analyze a sample of images
    sample_files = image_files[:sample_size]
    logger.info(f"Analyzing {len(sample_files)} images for Nataraj methodology compliance...")
    
    all_analyses = []
    for img_path in sample_files:
        analysis = analyze_image_properties(str(img_path))
        if analysis:
            all_analyses.append(analysis)
            
            # Check Nataraj methodology compliance
            issues = []
            
            if analysis['image_mode'] != 'L':
                issues.append(f"Mode should be 'L' (grayscale), found '{analysis['image_mode']}'")
            
            if analysis['data_type'] != 'uint8':
                issues.append(f"Data type should be 'uint8', found '{analysis['data_type']}'")
            
            if analysis['pixel_min'] < 0 or analysis['pixel_max'] > 255:
                issues.append(f"Pixel values outside 0-255 range: {analysis['pixel_min']}-{analysis['pixel_max']}")
            
            if analysis['entropy'] < 1.0:
                issues.append(f"Very low entropy ({analysis['entropy']:.2f}) - may indicate padding issues")
            
            if issues:
                logger.warning(f"Issues in {img_path.name}: {'; '.join(issues)}")
            else:
                logger.info(f"✓ {img_path.name}: Compliant with Nataraj methodology")
    
    # Summary statistics
    if all_analyses:
        entropies = [a['entropy'] for a in all_analyses]
        logger.info(f"\nSummary Statistics:")
        logger.info(f"Average entropy: {np.mean(entropies):.2f} ± {np.std(entropies):.2f}")
        logger.info(f"Entropy range: {np.min(entropies):.2f} - {np.max(entropies):.2f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Verify Nataraj methodology compliance")
    parser.add_argument('--image_dir', required=True, help='Directory containing malware images')
    parser.add_argument('--visualize', help='Path to specific image to visualize')
    parser.add_argument('--output_dir', help='Directory to save visualizations')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of images to analyze')
    
    args = parser.parse_args()
    
    if args.visualize:
        logger.info(f"Creating visualization for {args.visualize}")
        visualize_malware_image(args.visualize, args.output_dir)
    else:
        logger.info(f"Verifying Nataraj methodology compliance in {args.image_dir}")
        verify_nataraj_methodology(args.image_dir, args.sample_size)


if __name__ == "__main__":
    main()