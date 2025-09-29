#!/usr/bin/env python3
"""
Generate Variable Size MOTIF Dataset
===================================

This script demonstrates how to generate a variable-sized malware image dataset
following the original Nataraj et al. approach exactly.

The original Nataraj method:
1. Reads binary file as sequence of bytes
2. Calculates width = ceil(sqrt(file_size))
3. Arranges bytes into 2D matrix with natural dimensions
4. Each byte value becomes a pixel intensity (0-255)
5. Preserves spatial locality of bytes in the binary

Usage:
    python generate_variable_size_dataset.py --data_dir /path/to/motif --output_dir ./motif_variable_size
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# Use the existing generate_image.py functionality
from generate_image import MOTIFDatasetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate variable-sized malware image dataset following original Nataraj approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Original Nataraj Approach:
    1. Binary file → byte sequence
    2. Width = ceil(sqrt(file_size))  
    3. Height = ceil(file_size / width)
    4. Arrange bytes into width × height matrix
    5. Each byte value → pixel intensity
    
This preserves:
    - Natural dimensions based on file size
    - Spatial locality of bytes
    - Direct byte-to-pixel mapping
    
Examples:
    # Generate variable size dataset
    python generate_variable_size_dataset.py --data_dir /path/to/motif --output_dir ./variable_size_dataset
    
    # With custom test split
    python generate_variable_size_dataset.py --data_dir /path/to/motif --output_dir ./variable_size_dataset --test_size 0.3
        """
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to MOTIF dataset directory')
    parser.add_argument('--output_dir', type=str, default='./motif_images_variable_size', 
                       help='Output directory for variable size images')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data for testing')
    parser.add_argument('--max_size', type=int, default=2048,
                       help='Maximum image dimension (for memory constraints)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("GENERATING VARIABLE SIZE MALWARE IMAGES")
    logger.info("="*60)
    logger.info("Following original Nataraj et al. methodology:")
    logger.info("- Variable dimensions based on file size")
    logger.info("- Width = ceil(sqrt(file_size))")
    logger.info("- Preserves spatial locality of bytes")
    logger.info("- Direct byte-to-pixel mapping")
    logger.info("="*60)
    
    # Create processor with variable size setting
    processor = MOTIFDatasetProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=None,  # Not used for variable size
        test_size=args.test_size,
        fixed_size=False  # KEY: This enables variable size generation
    )
    
    # Override max size constraint in image generator
    processor.image_generator.max_size = args.max_size
    
    # Generate the dataset
    processor.generate_dataset()
    
    logger.info("\n" + "="*60)
    logger.info("VARIABLE SIZE DATASET GENERATION COMPLETE")
    logger.info("="*60)
    logger.info("Key characteristics of generated images:")
    logger.info("- Dimensions vary naturally based on file size")
    logger.info("- Preserves original Nataraj spatial relationships")
    logger.info("- Ready for variable size CNN training")
    logger.info("="*60)
    
    # Show some example usage
    logger.info("\nNext steps - Train with variable sizes:")
    logger.info("1. Adaptive padding approach (recommended):")
    logger.info(f"   python train_variable_size.py --dataset_dir {args.output_dir} --approach adaptive_pad")
    logger.info("\n2. True variable size approach (original Nataraj):")
    logger.info(f"   python train_variable_size.py --dataset_dir {args.output_dir} --approach preserve --model_type variable")
    logger.info("\n3. Multi-scale training:")
    logger.info(f"   python train_variable_size.py --dataset_dir {args.output_dir} --approach multi_scale")


if __name__ == "__main__":
    main()