#!/usr/bin/env python3
"""
MOTIF Dataset Image Generation Tool
==================================

This script converts malware binary files from the MOTIF dataset into grayscale images
for static malware classification. It processes both malware and benign files,
creating training and testing datasets.

Requirements:
- Python 3.6+
- numpy
- PIL (Pillow)
- sklearn
- matplotlib (optional for visualization)

Usage:
    python generate_image.py --data_dir /path/to/motif/dataset --output_dir ./datasets
"""

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
import math
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import hashlib
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MalwareImageGenerator:
    """
    Generate grayscale images from malware binary files.
    
    Implementation follows the methodology from Nataraj et al. (2011):
    "Malware images: visualization and automatic classification"
    """
    
    def __init__(self, image_size: int = 256, fixed_size: bool = False):
        """
        Initialize the malware image generator.
        
        Args:
            image_size: Size of the output square image (image_size x image_size)
            fixed_size: If True, all images will be resized to image_size x image_size.
                       If False, images will have variable dimensions based on file size
                       (following original Nataraj approach more closely)
        """
        self.image_size = image_size
        self.fixed_size = fixed_size
        
    def binary_to_image(self, file_path: str, output_path: str = None) -> Optional[np.ndarray]:
        """
        Convert a binary file to a grayscale image following Nataraj et al. methodology.
        
        Implementation follows "Malware images: visualization and automatic classification" (VizSec 2011)
        Key aspects:
        1. Read binary as 8-bit unsigned integers (0-255 range)
        2. Arrange into 2D matrix preserving spatial locality of bytes
        3. Use square or near-square dimensions for visualization
        4. Each byte value directly maps to grayscale pixel intensity
        
        Handles both compressed (gzip) and uncompressed binary files (MOTIF compatibility).
        
        Args:
            file_path: Path to the binary file
            output_path: Path to save the image (optional)
            
        Returns:
            numpy array representing the image, or None if failed
        """
        try:
            # Try to read as gzipped file first (MOTIF compatibility)
            binary_data = None
            try:
                import gzip
                with gzip.open(file_path, 'rb') as f:
                    binary_data = f.read()
                logger.debug(f"Successfully read gzipped file: {file_path}")
            except (OSError, gzip.BadGzipFile):
                # Not a gzipped file, read as regular binary
                with open(file_path, 'rb') as f:
                    binary_data = f.read()
                logger.debug(f"Successfully read regular binary file: {file_path}")
            
            if len(binary_data) == 0:
                logger.warning(f"Empty file: {file_path}")
                return None
                
            # Convert binary data to numpy array of uint8 (following Nataraj method)
            data_array = np.frombuffer(binary_data, dtype=np.uint8)
            
            # Calculate optimal dimensions following Nataraj approach
            # Use square root to get approximately square image for better visualization
            file_size = len(data_array)
            width = int(math.ceil(math.sqrt(file_size)))
            
            # If we want a fixed size output, calculate accordingly
            if hasattr(self, 'fixed_size') and self.fixed_size:
                # For fixed size (e.g., 256x256), pad or truncate
                total_pixels = self.image_size * self.image_size
                
                if file_size < total_pixels:
                    # Pad with zeros (black pixels) - common approach
                    padded_data = np.zeros(total_pixels, dtype=np.uint8)
                    padded_data[:file_size] = data_array
                    data_array = padded_data
                elif file_size > total_pixels:
                    # Truncate to fit - preserve beginning of file
                    data_array = data_array[:total_pixels]
                
                image_array = data_array.reshape(self.image_size, self.image_size)
            else:
                # Variable size based on file size (original Nataraj approach)
                height = int(math.ceil(file_size / width))
                total_pixels = width * height
                
                if file_size < total_pixels:
                    # Pad the last row if necessary
                    padded_data = np.zeros(total_pixels, dtype=np.uint8)
                    padded_data[:file_size] = data_array
                    data_array = padded_data
                
                image_array = data_array.reshape(height, width)
            
            # Save image if output path is provided
            if output_path:
                image = Image.fromarray(image_array, 'L')  # 'L' for grayscale
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path, format='PNG')
                logger.debug(f"Saved image: {output_path} (dimensions: {image_array.shape})")
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None


class MOTIFDatasetProcessor:
    """Process MOTIF dataset and generate train/test splits."""
    
    def __init__(self, data_dir: str, output_dir: str, image_size: int = 256, test_size: float = 0.2, fixed_size: bool = True):
        """
        Initialize the MOTIF dataset processor.
        
        Args:
            data_dir: Path to the MOTIF dataset directory
            output_dir: Path to save processed datasets
            image_size: Size of generated images (only used if fixed_size=True)
            test_size: Fraction of data to use for testing
            fixed_size: Whether to use fixed-size images or variable size (Nataraj original)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.test_size = test_size
        self.image_generator = MalwareImageGenerator(image_size, fixed_size)
        self.family_mapping = {}  # Will store label_id -> family_name mapping
        
        # Create output directories
        self.train_dir = self.output_dir / "train"
        self.test_dir = self.output_dir / "test"
        
        # Will create class directories dynamically based on discovered families
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash for a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return "unknown"
    
    def discover_files(self) -> Dict[str, List[str]]:
        """
        Discover malware and benign files in the MOTIF dataset.
        
        For MOTIF dataset, this method:
        1. Reads motif_dataset.jsonl for labels
        2. Looks for MOTIF_defanged folder with binary files
        3. Maps files using MD5 hashes
        
        Returns:
            Dictionary with 'malware' and 'benign' keys containing file paths
        """
        files = {"malware": [], "benign": []}
        
        logger.info(f"Scanning MOTIF dataset directory: {self.data_dir}")
        
        # Look for MOTIF dataset structure
        motif_jsonl = self.data_dir / "dataset" / "motif_dataset.jsonl"
        motif_binaries_dir = None
        
        # Find the directory with binary files
        for possible_dir in [
            self.data_dir / "MOTIF_defanged",
            self.data_dir / "dataset" / "MOTIF_defanged", 
            self.data_dir / "motif_defanged",
            self.data_dir  # In case files are directly in data_dir
        ]:
            if possible_dir.exists():
                # Check if it contains MOTIF_ files
                motif_files = list(possible_dir.glob("MOTIF_*"))
                if motif_files:
                    motif_binaries_dir = possible_dir
                    logger.info(f"Found MOTIF binaries in: {motif_binaries_dir}")
                    break
        
        if motif_jsonl.exists() and motif_binaries_dir:
            # MOTIF dataset with jsonl labels
            logger.info("Processing MOTIF dataset with jsonl labels")
            files = self._process_motif_dataset(motif_jsonl, motif_binaries_dir)
        else:
            # Check if we have existing family structure (from previous run)
            if self._has_existing_family_structure():
                logger.info("Found existing family structure, using for re-generation")
                files = self._process_existing_family_structure()
            else:
                # Fallback to generic processing
                logger.info("No MOTIF-specific structure found, using generic file discovery")
                files = self._discover_files_generic()
        
        logger.info(f"Found {len(files['malware'])} malware files")
        logger.info(f"Found {len(files['benign'])} benign files")
        
        return files
    
    def _process_motif_dataset(self, jsonl_path: Path, binaries_dir: Path) -> Dict[str, List[str]]:
        """Process MOTIF dataset using the jsonl file for labels."""
        import json
        
        # MOTIF dataset has 454 malware families - use multi-class approach
        files = {}  # Will contain family_name: [file_paths] mapping
        family_mapping = {}  # label_id: family_name mapping
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    md5_hash = entry.get('md5', '')
                    family_label = entry.get('label', None)  # Numeric family label
                    reported_family = entry.get('reported_family', f'family_{family_label}')
                    
                    # Look for corresponding binary file
                    binary_file = binaries_dir / f"MOTIF_{md5_hash}"
                    
                    if binary_file.exists() and family_label is not None:
                        # Clean family name for directory structure
                        clean_family_name = self._clean_family_name(reported_family)
                        
                        # Store family mapping
                        family_mapping[family_label] = clean_family_name
                        
                        # Group files by family
                        if clean_family_name not in files:
                            files[clean_family_name] = []
                        files[clean_family_name].append(str(binary_file))
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing JSON line: {e}")
                except Exception as e:
                    logger.warning(f"Error processing entry: {e}")
        
        # Store family mapping for later use
        self.family_mapping = family_mapping
        
        logger.info(f"Found {len(files)} malware families")
        for family_name, file_list in files.items():
            logger.info(f"  {family_name}: {len(file_list)} samples")
        
        return files
    
    def _clean_family_name(self, family_name: str) -> str:
        """Clean family name for use as directory name."""
        import re
        # Remove special characters and normalize
        cleaned = re.sub(r'[^\w\-_]', '_', family_name.lower())
        # Remove multiple underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        return cleaned if cleaned else 'unknown'
    
    def _discover_files_generic(self) -> Dict[str, List[str]]:
        """Generic file discovery for non-MOTIF datasets or when JSONL is not available."""
        files = {}
        
        # For MOTIF without JSONL, treat each file as separate family based on naming
        for root, dirs, filenames in os.walk(self.data_dir):
            root_path = Path(root)
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                
                # Skip non-executable files and common non-binary files
                if self._is_valid_binary_file(filename):
                    # For MOTIF files, extract potential family info from filename or use generic
                    if filename.startswith("MOTIF_"):
                        # Use hash as family identifier when no other info available
                        family_name = f"unknown_family_{filename.split('_')[1][:8]}"
                    else:
                        # Generic classification attempt based on path
                        family_name = self._infer_family_from_path(root_path)
                    
                    if family_name not in files:
                        files[family_name] = []
                    files[family_name].append(file_path)
        
        return files
    
    def _infer_family_from_path(self, path: Path) -> str:
        """Infer family name from directory path."""
        path_parts = path.parts
        
        # Common patterns for malware dataset organization
        malware_patterns = [
            "malware", "malicious", "virus", "trojan", "worm", "backdoor",
            "adware", "spyware", "ransomware", "rootkit"
        ]
        
        # Look for family indicators in path
        for part in reversed(path_parts):
            part_lower = part.lower()
            if any(pattern in part_lower for pattern in malware_patterns):
                return self._clean_family_name(part)
        
        return "unknown_family"
    
    def _has_existing_family_structure(self) -> bool:
        """Check if we have an existing family-based directory structure."""
        import json
        
        # Look for the current dataset structure
        existing_dataset_info = self.output_dir / "dataset_info.json"
        if existing_dataset_info.exists():
            try:
                with open(existing_dataset_info, 'r') as f:
                    info = json.load(f)
                    return info.get("dataset_type") == "multi_class_family_classification"
            except:
                pass
        
        # Also check if train/test directories have family subdirectories
        if self.train_dir.exists():
            subdirs = [d for d in self.train_dir.iterdir() if d.is_dir()]
            if len(subdirs) > 2:  # More than just malware/benign
                return True
        
        return False
    
    def _process_existing_family_structure(self) -> Dict[str, List[str]]:
        """Process existing family structure from previous generation."""
        files = {}
        
        # This is a placeholder - in practice, you'd need the original binary files
        # For now, we'll create a warning that re-generation needs original data
        logger.warning("Found existing family structure but no original binary files")
        logger.warning("To regenerate with family classification, you need:")
        logger.warning("1. The original MOTIF binary files (MOTIF_defanged directory)")
        logger.warning("2. The motif_dataset.jsonl file with family labels")
        
        return files
    
    def _is_valid_binary_file(self, filename: str) -> bool:
        """Check if a file is likely a binary executable file."""
        # Skip image files, text files, etc.
        skip_extensions = ['.txt', '.log', '.jpg', '.png', '.gif', '.pdf', '.json', '.csv', '.xml', '.html']
        filename_lower = filename.lower()
        
        if any(filename_lower.endswith(ext) for ext in skip_extensions):
            return False
            
        # MOTIF files are valid
        if filename.startswith("MOTIF_"):
            return True
            
        # Skip very short filenames that are likely not executables
        if len(filename) < 3:
            return False
            
        return True
    
    def process_files(self, files: Dict[str, List[str]]) -> Dict[str, any]:
        """
        Process files and create train/test splits with images.
        
        Args:
            files: Dictionary containing family_name -> [file_paths] mapping
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "train_count": 0,
            "test_count": 0,
            "families": {},
            "num_families": len(files)
        }
        
        # Create directories for all families
        for family_name in files.keys():
            for split_dir in [self.train_dir, self.test_dir]:
                (split_dir / family_name).mkdir(parents=True, exist_ok=True)
            stats["families"][family_name] = {"train": 0, "test": 0, "total": 0}
        
        # Process each family
        for family_name, file_list in files.items():
            if not file_list:
                logger.warning(f"No {family_name} files found")
                continue
                
            logger.info(f"Processing {len(file_list)} {family_name} files...")
            
            # Create train/test split for this family
            if len(file_list) == 1:
                # If only one file, put it in training
                train_files = file_list
                test_files = []
                logger.warning(f"Family {family_name} has only 1 sample, placing in training set")
            else:
                train_files, test_files = train_test_split(
                    file_list, 
                    test_size=self.test_size, 
                    random_state=42,
                    shuffle=True
                )
            
            # Process training files
            for i, file_path in enumerate(train_files):
                output_name = f"{family_name}_{i:06d}_{self.get_file_hash(file_path)[:8]}.png"
                output_path = self.train_dir / family_name / output_name
                
                image_array = self.image_generator.binary_to_image(file_path, str(output_path))
                
                if image_array is not None:
                    stats["successful"] += 1
                    stats["train_count"] += 1
                    stats["families"][family_name]["train"] += 1
                    stats["families"][family_name]["total"] += 1
                else:
                    stats["failed"] += 1
                
                stats["total_processed"] += 1
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(train_files)} {family_name} training files")
            
            # Process testing files
            for i, file_path in enumerate(test_files):
                output_name = f"{family_name}_{i:06d}_{self.get_file_hash(file_path)[:8]}.png"
                output_path = self.test_dir / family_name / output_name
                
                image_array = self.image_generator.binary_to_image(file_path, str(output_path))
                
                if image_array is not None:
                    stats["successful"] += 1
                    stats["test_count"] += 1
                    stats["families"][family_name]["test"] += 1
                    stats["families"][family_name]["total"] += 1
                else:
                    stats["failed"] += 1
                
                stats["total_processed"] += 1
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_files)} {family_name} testing files")
            
            logger.info(f"Completed {family_name}: {stats['families'][family_name]['total']} total samples")
        
        return stats
    
    def create_dataset_info(self, stats: Dict[str, any]) -> None:
        """Create dataset information file."""
        # Create class mapping (family_name -> class_id)
        family_names = sorted(stats["families"].keys())
        classes = {str(i): family_name for i, family_name in enumerate(family_names)}
        class_to_id = {family_name: str(i) for i, family_name in enumerate(family_names)}
        
        # Create detailed split information
        split_info = {
            "train": {"total": stats["train_count"], "families": {}},
            "test": {"total": stats["test_count"], "families": {}},
        }
        
        for family_name, family_stats in stats["families"].items():
            split_info["train"]["families"][family_name] = {
                "count": family_stats["train"],
                "class_id": class_to_id[family_name]
            }
            split_info["test"]["families"][family_name] = {
                "count": family_stats["test"],
                "class_id": class_to_id[family_name]
            }
        
        dataset_info = {
            "dataset_name": "MOTIF_Malware_Family_Images",
            "dataset_type": "multi_class_family_classification",
            "image_size": self.image_size,
            "test_size": self.test_size,
            "num_classes": len(family_names),
            "statistics": {
                "total_processed": stats["total_processed"],
                "successful": stats["successful"],
                "failed": stats["failed"],
                "train_count": stats["train_count"],
                "test_count": stats["test_count"],
                "num_families": stats["num_families"]
            },
            "classes": classes,  # class_id -> family_name mapping
            "class_to_id": class_to_id,  # family_name -> class_id mapping
            "split_info": split_info,
            "family_statistics": stats["families"]
        }
        
        # Add family mapping if available
        if hasattr(self, 'family_mapping') and self.family_mapping:
            dataset_info["original_family_mapping"] = self.family_mapping
        
        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Dataset information saved to {info_path}")
        logger.info(f"Multi-class dataset with {len(family_names)} malware families")
    
    def generate_dataset(self) -> None:
        """Main method to generate the complete dataset."""
        logger.info("Starting MOTIF dataset processing...")
        logger.info(f"Input directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Image size: {self.image_size}x{self.image_size}")
        logger.info(f"Test split: {self.test_size}")
        
        # Discover files
        files = self.discover_files()
        
        if not files["malware"] and not files["benign"]:
            logger.error("No suitable files found in the dataset directory")
            sys.exit(1)
        
        # Process files and create images
        stats = self.process_files(files)
        
        # Create dataset information
        self.create_dataset_info(stats)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("DATASET GENERATION COMPLETE")
        logger.info("="*50)
        logger.info(f"Total files processed: {stats['total_processed']}")
        logger.info(f"Successfully converted: {stats['successful']}")
        logger.info(f"Failed conversions: {stats['failed']}")
        logger.info(f"Training images: {stats['train_count']}")
        logger.info(f"Testing images: {stats['test_count']}")
        logger.info(f"Number of malware families: {stats['num_families']}")
        
        # Show top families by sample count
        family_counts = [(name, data["total"]) for name, data in stats["families"].items()]
        family_counts.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("\nTop 10 families by sample count:")
        for i, (family_name, count) in enumerate(family_counts[:10]):
            logger.info(f"  {i+1}. {family_name}: {count} samples")
        
        if len(family_counts) > 10:
            logger.info(f"  ... and {len(family_counts) - 10} more families")
        
        logger.info(f"\nDataset saved to: {self.output_dir}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate image dataset from MOTIF malware dataset for multi-class family classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MOTIF Dataset Information:
    The MOTIF dataset contains 3,095 malware samples from 454 different families.
    This script generates images for multi-class family classification, not binary malware/benign classification.
    
    Required files:
    - motif_dataset.jsonl: Contains family labels for each sample
    - MOTIF_defanged/: Directory containing the disarmed binary files
    
Examples:
    # Basic usage for family classification
    python generate_image.py --data_dir /path/to/motif --output_dir ./family_dataset
    
    # Custom image size and test split
    python generate_image.py --data_dir /path/to/motif --output_dir ./family_dataset --image_size 512 --test_size 0.3
    
    # Verbose logging
    python generate_image.py --data_dir /path/to/motif --output_dir ./family_dataset --verbose
        """
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True,
        help='Path to the MOTIF dataset directory containing motif_dataset.jsonl and MOTIF_defanged/ folder'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./motif_family_dataset',
        help='Directory to save the generated multi-class family image dataset (default: ./motif_family_dataset)'
    )
    
    parser.add_argument(
        '--image_size', 
        type=int, 
        default=256,
        help='Size of generated square images in pixels (default: 256)'
    )
    
    parser.add_argument(
        '--test_size', 
        type=float, 
        default=0.2,
        help='Fraction of data to use for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--variable_size', 
        action='store_true',
        help='Use variable image dimensions based on file size (original Nataraj method). '
             'Default is fixed size for consistent CNN input.'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    if args.image_size <= 0:
        logger.error("Image size must be positive")
        sys.exit(1)
    
    if not 0 < args.test_size < 1:
        logger.error("Test size must be between 0 and 1")
        sys.exit(1)
    
    # Create processor and generate dataset
    processor = MOTIFDatasetProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        test_size=args.test_size,
        fixed_size=not args.variable_size  # Invert because variable_size flag means NOT fixed_size
    )
    
    processor.generate_dataset()


if __name__ == "__main__":
    main()
