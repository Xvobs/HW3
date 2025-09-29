#!/usr/bin/env python3
"""
Dataset Visualization and Validation Tool
==========================================

This script provides utilities to visualize and validate the generated malware image dataset.
Supports both binary classification and multi-class family classification datasets.

For multi-class datasets:
- Family distribution analysis
- Class imbalance visualization  
- Top families by sample count
- Comprehensive family statistics

Usage:
    # Basic visualization
    python visualize_dataset.py --dataset_dir ./motif_dataset --num_samples 16
    
    # Multi-class family analysis
    python visualize_dataset.py --dataset_dir ./motif_family_dataset --family_distribution --top_families 50
    
    # Save plots without display (headless)
    python visualize_dataset.py --dataset_dir ./motif_family_dataset --save_plots ./plots --no_display
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
        
        # Determine dataset type
        self.dataset_type = self.dataset_info.get("dataset_type", "binary_classification") if self.dataset_info else "binary_classification"
        self.is_multi_class = self.dataset_type == "multi_class_family_classification"
        
        if self.is_multi_class:
            self.num_classes = self.dataset_info.get("num_classes", 0)
            self.classes = self.dataset_info.get("classes", {})
            self.class_to_id = self.dataset_info.get("class_to_id", {})
            logger.info(f"Multi-class dataset with {self.num_classes} families")
        else:
            self.num_classes = 2
            self.classes = {0: "benign", 1: "malware"}
            logger.info("Binary classification dataset")
    
    def validate_dataset(self) -> dict:
        """Validate the dataset structure and count files."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        if self.is_multi_class:
            # Multi-class validation
            validation_results = self._validate_multiclass_dataset()
        else:
            # Binary classification validation
            validation_results = self._validate_binary_dataset()
        
        return validation_results
    
    def _validate_binary_dataset(self) -> dict:
        """Validate binary classification dataset."""
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
        
        return validation_results
    
    def _validate_multiclass_dataset(self) -> dict:
        """Validate multi-class family classification dataset."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Check if train/test directories exist
        for split_dir in [self.train_dir, self.test_dir]:
            if not split_dir.exists():
                validation_results["errors"].append(f"Missing directory: {split_dir}")
                validation_results["valid"] = False
        
        if not validation_results["valid"]:
            return validation_results
        
        # Count files in each family directory
        stats = {
            "train": {"families": {}, "total": 0},
            "test": {"families": {}, "total": 0},
            "total": 0,
            "families": {}
        }
        
        all_families = set()
        
        for split in ["train", "test"]:
            split_dir = self.dataset_dir / split
            
            # Get all family directories
            if split_dir.exists():
                family_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                
                for family_dir in family_dirs:
                    family_name = family_dir.name
                    all_families.add(family_name)
                    
                    png_files = list(family_dir.glob("*.png"))
                    count = len(png_files)
                    
                    stats[split]["families"][family_name] = count
                    stats[split]["total"] += count
                    
                    if count == 0:
                        validation_results["warnings"].append(f"No images found in {family_dir}")
        
        # Calculate overall family statistics
        for family_name in all_families:
            train_count = stats["train"]["families"].get(family_name, 0)
            test_count = stats["test"]["families"].get(family_name, 0)
            total_count = train_count + test_count
            
            stats["families"][family_name] = {
                "train": train_count,
                "test": test_count,
                "total": total_count
            }
        
        stats["total"] = stats["train"]["total"] + stats["test"]["total"]
        stats["num_families"] = len(all_families)
        
        validation_results["statistics"] = stats
        
        # Validate a few random images
        sample_images = []
        for split in ["train", "test"]:
            split_dir = self.dataset_dir / split
            if split_dir.exists():
                family_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                for family_dir in random.sample(family_dirs, min(5, len(family_dirs))):
                    png_files = list(family_dir.glob("*.png"))
                    if png_files:
                        sample_images.extend(random.sample(png_files, min(2, len(png_files))))
        
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
        
        if self.is_multi_class:
            # Multi-class: sample from various families
            for split in ["train", "test"]:
                split_dir = self.dataset_dir / split
                if split_dir.exists():
                    family_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                    
                    # Sample families to show diversity
                    selected_families = random.sample(family_dirs, min(num_samples // 2, len(family_dirs)))
                    
                    for family_dir in selected_families:
                        family_name = family_dir.name
                        png_files = list(family_dir.glob("*.png"))
                        
                        if png_files:
                            # Take 1-2 samples from each family
                            sample_count = min(2, len(png_files))
                            selected_files = random.sample(png_files, sample_count)
                            
                            for img_path in selected_files:
                                samples.append((str(img_path), family_name, split))
                                if len(samples) >= num_samples:
                                    break
                        
                        if len(samples) >= num_samples:
                            break
                    
                    if len(samples) >= num_samples:
                        break
        else:
            # Binary classification: sample from malware/benign
            for split in ["train", "test"]:
                for class_name in ["malware", "benign"]:
                    class_dir = self.dataset_dir / split / class_name
                    png_files = list(class_dir.glob("*.png"))
                    
                    if png_files:
                        sample_count = min(num_samples // 4, len(png_files))
                        selected_files = random.sample(png_files, sample_count)
                        
                        for img_path in selected_files:
                            samples.append((str(img_path), class_name, split))
        
        return samples[:num_samples]
    
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
        
        fig.suptitle(f"MOTIF Dataset Sample Images ({self.dataset_type.replace('_', ' ').title()})", fontsize=16)
        
        for idx, (img_path, class_name, split) in enumerate(samples):
            row = idx // cols
            col = idx % cols
            
            try:
                # Load and display image
                img = Image.open(img_path)
                img_array = np.array(img)
                
                axes[row, col].imshow(img_array, cmap='gray')
                
                # Truncate long family names for display
                display_name = class_name if len(class_name) <= 12 else class_name[:12] + "..."
                axes[row, col].set_title(f"{display_name} ({split})", fontsize=9)
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
        
        if self.is_multi_class:
            self._plot_multiclass_statistics(stats, save_path)
        else:
            self._plot_binary_statistics(stats, save_path)
    
    def _plot_binary_statistics(self, stats: dict, save_path: str = None):
        """Plot statistics for binary classification."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("MOTIF Binary Classification Dataset Statistics", fontsize=16)
        
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
    
    def _plot_multiclass_statistics(self, stats: dict, save_path: str = None):
        """Plot statistics for multi-class family classification."""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f"MOTIF Multi-Class Family Dataset Statistics ({stats['num_families']} families)", fontsize=16)
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Train vs Test split (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        train_total = stats["train"]["total"]
        test_total = stats["test"]["total"]
        
        ax1.pie([train_total, test_total], labels=['Train', 'Test'], autopct='%1.1f%%')
        ax1.set_title('Train/Test Split')
        
        # 2. Top 20 families by total count (top middle and right)
        ax2 = fig.add_subplot(gs[0, 1:])
        family_counts = [(name, data["total"]) for name, data in stats["families"].items()]
        family_counts.sort(key=lambda x: x[1], reverse=True)
        
        top_20 = family_counts[:20]
        families, counts = zip(*top_20) if top_20 else ([], [])
        
        bars = ax2.bar(range(len(families)), counts, alpha=0.7)
        ax2.set_title('Top 20 Families by Sample Count')
        ax2.set_ylabel('Number of Samples')
        ax2.set_xticks(range(len(families)))
        ax2.set_xticklabels([f[:10] + "..." if len(f) > 10 else f for f in families], 
                           rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=8)
        
        # 3. Distribution of samples per family (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        sample_counts = [data["total"] for data in stats["families"].values()]
        ax3.hist(sample_counts, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_title('Distribution of Samples per Family')
        ax3.set_xlabel('Number of Samples')
        ax3.set_ylabel('Number of Families')
        
        # 4. Train/Test split by family size (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Categorize families by size
        size_categories = {'1-2 samples': 0, '3-5 samples': 0, '6-10 samples': 0, '11-20 samples': 0, '20+ samples': 0}
        for family_data in stats["families"].values():
            total = family_data["total"]
            if total <= 2:
                size_categories['1-2 samples'] += 1
            elif total <= 5:
                size_categories['3-5 samples'] += 1
            elif total <= 10:
                size_categories['6-10 samples'] += 1
            elif total <= 20:
                size_categories['11-20 samples'] += 1
            else:
                size_categories['20+ samples'] += 1
        
        labels, values = zip(*size_categories.items())
        ax4.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Families by Sample Count')
        
        # 5. Train vs Test samples comparison (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        train_counts = []
        test_counts = []
        family_names = []
        
        # Show top 15 families for readability
        for name, data in sorted(stats["families"].items(), key=lambda x: x[1]["total"], reverse=True)[:15]:
            family_names.append(name[:8] + "..." if len(name) > 8 else name)
            train_counts.append(data["train"])
            test_counts.append(data["test"])
        
        x = np.arange(len(family_names))
        width = 0.35
        
        ax5.bar(x - width/2, train_counts, width, label='Train', alpha=0.7)
        ax5.bar(x + width/2, test_counts, width, label='Test', alpha=0.7)
        
        ax5.set_title('Train/Test Split for Top 15 Families')
        ax5.set_ylabel('Number of Samples')
        ax5.set_xticks(x)
        ax5.set_xticklabels(family_names, rotation=45, ha='right')
        ax5.legend()
        
        # 6. Summary statistics (bottom)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Calculate statistics
        total_samples = stats["total"]
        num_families = stats["num_families"]
        avg_samples_per_family = total_samples / num_families if num_families > 0 else 0
        
        families_with_1_sample = sum(1 for data in stats["families"].values() if data["total"] == 1)
        families_with_10plus = sum(1 for data in stats["families"].values() if data["total"] >= 10)
        
        largest_family = max(stats["families"].items(), key=lambda x: x[1]["total"]) if stats["families"] else ("None", {"total": 0})
        
        summary_text = f"""
        Dataset Summary Statistics:
        • Total Samples: {total_samples:,}
        • Number of Families: {num_families:,}
        • Average Samples per Family: {avg_samples_per_family:.1f}
        • Families with Only 1 Sample: {families_with_1_sample} ({families_with_1_sample/num_families*100:.1f}%)
        • Families with 10+ Samples: {families_with_10plus} ({families_with_10plus/num_families*100:.1f}%)
        • Largest Family: {largest_family[0]} ({largest_family[1]["total"]} samples)
        • Train/Test Ratio: {train_total/test_total:.2f}:1
        """
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Statistics plot saved to {save_path}")
        
        plt.show()
    
    def visualize_family_distribution(self, top_n: int = 30, save_path: str = None):
        """Visualize the distribution of samples across malware families (multi-class only)."""
        if not self.is_multi_class:
            logger.warning("Family distribution visualization is only available for multi-class datasets")
            return
        
        validation_results = self.validate_dataset()
        stats = validation_results["statistics"]
        
        if not stats or not stats.get("families"):
            logger.error("No family statistics available")
            return
        
        # Prepare data for visualization
        family_counts = [(name, data["total"]) for name, data in stats["families"].items()]
        family_counts.sort(key=lambda x: x[1], reverse=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f"MOTIF Family Distribution Analysis ({stats['num_families']} families)", fontsize=16)
        
        # 1. Top N families bar chart
        top_families = family_counts[:top_n]
        if top_families:
            families, counts = zip(*top_families)
            
            bars = ax1.barh(range(len(families)), counts, alpha=0.7)
            ax1.set_title(f'Top {len(families)} Families by Sample Count')
            ax1.set_xlabel('Number of Samples')
            ax1.set_yticks(range(len(families)))
            ax1.set_yticklabels([f[:20] + "..." if len(f) > 20 else f for f in families])
            ax1.invert_yaxis()  # Top family at the top
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax1.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{count}', va='center', fontsize=8)
        
        # 2. Histogram of family sizes
        all_counts = [count for _, count in family_counts]
        ax2.hist(all_counts, bins=min(30, max(10, len(set(all_counts)))), alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Family Sizes')
        ax2.set_xlabel('Number of Samples per Family')
        ax2.set_ylabel('Number of Families')
        ax2.set_yscale('log')  # Log scale for better visualization
        
        # 3. Cumulative distribution
        sorted_counts = sorted(all_counts, reverse=True)
        cumulative_samples = np.cumsum(sorted_counts)
        cumulative_percent = cumulative_samples / cumulative_samples[-1] * 100
        
        ax3.plot(range(1, len(cumulative_percent) + 1), cumulative_percent, marker='o', markersize=2)
        ax3.set_title('Cumulative Sample Distribution')
        ax3.set_xlabel('Number of Families (ranked by size)')
        ax3.set_ylabel('Cumulative Percentage of Samples')
        ax3.grid(True, alpha=0.3)
        
        # Add some reference lines
        ax3.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50% of samples')
        ax3.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% of samples')
        ax3.legend()
        
        # 4. Size category pie chart
        size_categories = {
            '1 sample': sum(1 for count in all_counts if count == 1),
            '2-3 samples': sum(1 for count in all_counts if 2 <= count <= 3),
            '4-10 samples': sum(1 for count in all_counts if 4 <= count <= 10),
            '11-20 samples': sum(1 for count in all_counts if 11 <= count <= 20),
            '21+ samples': sum(1 for count in all_counts if count > 20)
        }
        
        # Remove empty categories
        size_categories = {k: v for k, v in size_categories.items() if v > 0}
        
        if size_categories:
            labels, values = zip(*size_categories.items())
            wedges, texts, autotexts = ax4.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Families by Size Category')
            
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Family distribution plot saved to {save_path}")
        
        plt.show()
        
        # Print some interesting statistics
        print(f"\n📊 Family Distribution Analysis:")
        print(f"   • Top 10% of families ({len(family_counts)//10} families) contain {sum(count for _, count in family_counts[:len(family_counts)//10])} samples")
        print(f"   • Top 50% of families contain {sum(count for _, count in family_counts[:len(family_counts)//2])} samples")
        single_sample_families = sum(1 for _, count in family_counts if count == 1)
        print(f"   • {single_sample_families} families have only 1 sample ({single_sample_families/len(family_counts)*100:.1f}%)")
        large_families = sum(1 for _, count in family_counts if count >= 10)
        print(f"   • {large_families} families have 10+ samples ({large_families/len(family_counts)*100:.1f}%)")
    
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
            if self.is_multi_class:
                self._print_multiclass_summary(stats)
            else:
                self._print_binary_summary(stats)
        
        if self.dataset_info:
            print(f"\n� Generation Settings:")
            print(f"   Image size: {self.dataset_info.get('image_size', 'Unknown')}")
            print(f"   Test split: {self.dataset_info.get('test_size', 'Unknown')}")
            print(f"   Dataset type: {self.dataset_info.get('dataset_type', 'Unknown')}")
        
        print("\n" + "="*60)
    
    def _print_binary_summary(self, stats: dict):
        """Print summary for binary classification."""
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total images: {stats['total']}")
        print(f"   Training images: {stats['train']['malware'] + stats['train']['benign']}")
        print(f"     - Malware: {stats['train']['malware']}")
        print(f"     - Benign: {stats['train']['benign']}")
        print(f"   Testing images: {stats['test']['malware'] + stats['test']['benign']}")
        print(f"     - Malware: {stats['test']['malware']}")
        print(f"     - Benign: {stats['test']['benign']}")
    
    def _print_multiclass_summary(self, stats: dict):
        """Print summary for multi-class family classification."""
        print(f"\n📊 Multi-Class Family Dataset Statistics:")
        print(f"   Total images: {stats['total']:,}")
        print(f"   Number of malware families: {stats['num_families']}")
        print(f"   Training images: {stats['train']['total']:,}")
        print(f"   Testing images: {stats['test']['total']:,}")
        
        # Calculate and display statistics about family distribution
        family_sizes = [data["total"] for data in stats["families"].values()]
        if family_sizes:
            avg_size = sum(family_sizes) / len(family_sizes)
            min_size = min(family_sizes)
            max_size = max(family_sizes)
            
            print(f"\n📈 Family Distribution:")
            print(f"   Average samples per family: {avg_size:.1f}")
            print(f"   Smallest family: {min_size} samples")
            print(f"   Largest family: {max_size} samples")
            
            # Count families by size categories
            single_sample = sum(1 for size in family_sizes if size == 1)
            small_families = sum(1 for size in family_sizes if 2 <= size <= 5)
            medium_families = sum(1 for size in family_sizes if 6 <= size <= 20)
            large_families = sum(1 for size in family_sizes if size > 20)
            
            print(f"   Families with 1 sample: {single_sample} ({single_sample/len(family_sizes)*100:.1f}%)")
            print(f"   Families with 2-5 samples: {small_families} ({small_families/len(family_sizes)*100:.1f}%)")
            print(f"   Families with 6-20 samples: {medium_families} ({medium_families/len(family_sizes)*100:.1f}%)")
            print(f"   Families with 20+ samples: {large_families} ({large_families/len(family_sizes)*100:.1f}%)")
        
        # Show top 10 families
        family_counts = [(name, data["total"]) for name, data in stats["families"].items()]
        family_counts.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔝 Top 10 Families by Sample Count:")
        for i, (family_name, count) in enumerate(family_counts[:10]):
            train_count = stats["families"][family_name]["train"]
            test_count = stats["families"][family_name]["test"]
            print(f"   {i+1}. {family_name}: {count} samples (train: {train_count}, test: {test_count})")
        
        if len(family_counts) > 10:
            remaining = len(family_counts) - 10
            print(f"   ... and {remaining} more families")


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
    
    parser.add_argument(
        '--family_distribution',
        action='store_true',
        help='Generate detailed family distribution analysis (multi-class only)'
    )
    
    parser.add_argument(
        '--top_families',
        type=int,
        default=30,
        help='Number of top families to show in distribution analysis (default: 30)'
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
            family_dist_path = save_dir / "family_distribution.png" if args.family_distribution else None
        else:
            sample_path = stats_path = family_dist_path = None
        
        # Generate visualizations
        visualizer.visualize_samples(args.num_samples, sample_path)
        visualizer.plot_dataset_statistics(stats_path)
        
        # Generate family distribution analysis if requested
        if args.family_distribution:
            visualizer.visualize_family_distribution(args.top_families, family_dist_path)


if __name__ == "__main__":
    main()