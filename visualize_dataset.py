#!/usr/bin/env python3
"""
Dataset Visualization and Validation Tool
==========================================

This script provides utilities to visualize and validate the generated malware image dataset.
Use this to verify that your images were generated correctly and to explore the dataset.

Usage:
    python visualize_dataset.py --dataset_dir ./motif_dataset --num_samples 10
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetVisualizer:
    """Visualize and validate the generated malware image dataset."""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test"
        
        # Load dataset info if available
        info_path = self.dataset_dir / "dataset_info.json"
        self.dataset_info = None
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.dataset_info = json.load(f)
    
    def validate_dataset(self) -> dict:
        """Validate the dataset structure and count files."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Check directory structure
        required_dirs = [
            self.train_dir / "malware",
            self.train_dir / "benign",
            self.test_dir / "malware", 
            self.test_dir / "benign"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                validation_results["errors"].append(f"Missing directory: {dir_path}")
                validation_results["valid"] = False
        
        if not validation_results["valid"]:
            return validation_results
        
        # Count files in each directory
        stats = {}
        total_files = 0
        
        for split in ["train", "test"]:
            stats[split] = {}
            split_dir = self.dataset_dir / split
            
            for class_name in ["malware", "benign"]:
                class_dir = split_dir / class_name
                png_files = list(class_dir.glob("*.png"))
                count = len(png_files)
                stats[split][class_name] = count
                total_files += count
                
                # Check if any files exist
                if count == 0:
                    validation_results["warnings"].append(f"No images found in {class_dir}")
        
        stats["total"] = total_files
        validation_results["statistics"] = stats
        
        # Validate a few random images
        sample_images = []
        for split in ["train", "test"]:
            for class_name in ["malware", "benign"]:
                class_dir = self.dataset_dir / split / class_name
                png_files = list(class_dir.glob("*.png"))
                if png_files:
                    sample_images.extend(random.sample(png_files, min(3, len(png_files))))
        
        corrupted_images = []
        for img_path in sample_images:
            try:
                img = Image.open(img_path)
                img.verify()  # Verify image integrity
            except Exception as e:
                corrupted_images.append(str(img_path))
                validation_results["warnings"].append(f"Corrupted image: {img_path} - {str(e)}")
        
        if corrupted_images:
            validation_results["warnings"].append(f"Found {len(corrupted_images)} corrupted images")
        
        return validation_results
    
    def get_sample_images(self, num_samples: int = 8) -> List[Tuple[str, str, str]]:
        """Get random sample images from the dataset."""
        samples = []
        
        for split in ["train", "test"]:
            for class_name in ["malware", "benign"]:
                class_dir = self.dataset_dir / split / class_name
                png_files = list(class_dir.glob("*.png"))
                
                if png_files:
                    sample_count = min(num_samples // 4, len(png_files))
                    selected_files = random.sample(png_files, sample_count)
                    
                    for img_path in selected_files:
                        samples.append((str(img_path), class_name, split))
        
        return samples
    
    def visualize_samples(self, num_samples: int = 16, save_path: str = None):
        """Visualize random samples from the dataset."""
        samples = self.get_sample_images(num_samples)
        
        if not samples:
            logger.error("No sample images found")
            return
        
        # Calculate grid dimensions
        cols = 4
        rows = (len(samples) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle("MOTIF Dataset Sample Images", fontsize=16)
        
        for idx, (img_path, class_name, split) in enumerate(samples):
            row = idx // cols
            col = idx % cols
            
            try:
                # Load and display image
                img = Image.open(img_path)
                img_array = np.array(img)
                
                axes[row, col].imshow(img_array, cmap='gray')
                axes[row, col].set_title(f"{class_name} ({split})", fontsize=10)
                axes[row, col].axis('off')
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f"Error loading\n{class_name} ({split})", 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
        
        # Hide unused subplots
        for idx in range(len(samples), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def plot_dataset_statistics(self, save_path: str = None):
        """Plot dataset statistics."""
        validation_results = self.validate_dataset()
        stats = validation_results["statistics"]
        
        if not stats:
            logger.error("No statistics available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("MOTIF Dataset Statistics", fontsize=16)
        
        # 1. Train vs Test split
        train_total = stats["train"]["malware"] + stats["train"]["benign"]
        test_total = stats["test"]["malware"] + stats["test"]["benign"]
        
        ax1.pie([train_total, test_total], labels=['Train', 'Test'], autopct='%1.1f%%')
        ax1.set_title('Train/Test Split')
        
        # 2. Class distribution in training set
        train_malware = stats["train"]["malware"]
        train_benign = stats["train"]["benign"]
        
        ax2.pie([train_malware, train_benign], labels=['Malware', 'Benign'], autopct='%1.1f%%')
        ax2.set_title('Training Set Class Distribution')
        
        # 3. Class distribution in test set  
        test_malware = stats["test"]["malware"]
        test_benign = stats["test"]["benign"]
        
        ax3.pie([test_malware, test_benign], labels=['Malware', 'Benign'], autopct='%1.1f%%')
        ax3.set_title('Test Set Class Distribution')
        
        # 4. Overall counts bar chart
        categories = ['Train\nMalware', 'Train\nBenign', 'Test\nMalware', 'Test\nBenign']
        counts = [train_malware, train_benign, test_malware, test_benign]
        colors = ['red', 'green', 'darkred', 'darkgreen']
        
        bars = ax4.bar(categories, counts, color=colors, alpha=0.7)
        ax4.set_title('Sample Counts by Category')
        ax4.set_ylabel('Number of Images')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Statistics plot saved to {save_path}")
        
        plt.show()
    
    def print_dataset_summary(self):
        """Print a summary of the dataset."""
        validation_results = self.validate_dataset()
        
        print("\n" + "="*60)
        print("MOTIF DATASET SUMMARY")
        print("="*60)
        
        if validation_results["valid"]:
            print("✅ Dataset structure is valid")
        else:
            print("❌ Dataset structure has errors:")
            for error in validation_results["errors"]:
                print(f"   - {error}")
        
        if validation_results["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in validation_results["warnings"]:
                print(f"   - {warning}")
        
        stats = validation_results["statistics"]
        if stats:
            print(f"\n📊 Dataset Statistics:")
            print(f"   Total images: {stats['total']}")
            print(f"   Training images: {stats['train']['malware'] + stats['train']['benign']}")
            print(f"     - Malware: {stats['train']['malware']}")
            print(f"     - Benign: {stats['train']['benign']}")
            print(f"   Testing images: {stats['test']['malware'] + stats['test']['benign']}")
            print(f"     - Malware: {stats['test']['malware']}")
            print(f"     - Benign: {stats['test']['benign']}")
        
        if self.dataset_info:
            print(f"\n🔧 Generation Settings:")
            print(f"   Image size: {self.dataset_info.get('image_size', 'Unknown')}")
            print(f"   Test split: {self.dataset_info.get('test_size', 'Unknown')}")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize and validate MOTIF malware image dataset")
    
    parser.add_argument(
        '--dataset_dir', 
        type=str, 
        required=True,
        help='Path to the generated dataset directory'
    )
    
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=16,
        help='Number of sample images to visualize (default: 16)'
    )
    
    parser.add_argument(
        '--save_plots', 
        type=str,
        help='Directory to save visualization plots'
    )
    
    parser.add_argument(
        '--no_display', 
        action='store_true',
        help='Skip displaying plots (useful for headless environments)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        logger.error(f"Dataset directory does not exist: {args.dataset_dir}")
        return
    
    visualizer = DatasetVisualizer(args.dataset_dir)
    
    # Print summary
    visualizer.print_dataset_summary()
    
    if not args.no_display:
        # Set matplotlib backend for headless environments
        if args.save_plots and not os.environ.get('DISPLAY'):
            import matplotlib
            matplotlib.use('Agg')
        
        # Create plots
        save_dir = Path(args.save_plots) if args.save_plots else None
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            sample_path = save_dir / "sample_images.png"
            stats_path = save_dir / "dataset_statistics.png"
        else:
            sample_path = stats_path = None
        
        # Generate visualizations
        visualizer.visualize_samples(args.num_samples, sample_path)
        visualizer.plot_dataset_statistics(stats_path)


if __name__ == "__main__":
    main()