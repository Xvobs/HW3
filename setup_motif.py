#!/usr/bin/env python3
"""
MOTIF Dataset Configuration and Usage Guide
==========================================

This script helps you set up and use the MOTIF dataset with the Nataraj image generation tools.

Key information about MOTIF dataset:
- 3,095 disarmed PE malware samples from 454 families
- Files are named MOTIF_[MD5] format
- Labels available in motif_dataset.jsonl
- Binary files may be compressed with gzip
- All samples are malware (different families)

For binary classification, you'll need to either:
1. Add benign/legitimate software samples
2. Treat some families as "benign" 
3. Use it for multi-class family classification instead
"""

import os
import sys
import json
import argparse
from pathlib import Path


def analyze_motif_dataset(motif_dir: str):
    """Analyze the MOTIF dataset structure and provide recommendations."""
    motif_path = Path(motif_dir)
    
    print("=== MOTIF Dataset Analysis ===")
    print(f"Dataset directory: {motif_path}")
    
    # Check for expected files
    dataset_dir = motif_path / "dataset"
    jsonl_file = dataset_dir / "motif_dataset.jsonl" 
    defanged_dir = motif_path / "MOTIF_defanged"
    
    # Alternative locations
    if not defanged_dir.exists():
        for alt_name in ["motif_defanged", "MOTIF", "defanged"]:
            alt_dir = motif_path / alt_name
            if alt_dir.exists():
                defanged_dir = alt_dir
                break
    
    print(f"\\nDataset structure:")
    print(f"  📁 Main directory: {motif_path.exists()}")
    print(f"  📁 dataset/: {dataset_dir.exists()}")
    print(f"  📄 motif_dataset.jsonl: {jsonl_file.exists()}")
    print(f"  📁 MOTIF_defanged/: {defanged_dir.exists()}")
    
    if jsonl_file.exists():
        # Analyze the jsonl file
        families = set()
        sample_count = 0
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    families.add(entry.get('label', 'unknown'))
                    sample_count += 1
                except:
                    pass
        
        print(f"\\n📊 Dataset Statistics:")
        print(f"  Total samples: {sample_count}")
        print(f"  Malware families: {len(families)}")
        print(f"  Sample families: {sorted(list(families))[:10]}..." if len(families) > 10 else f"  All families: {sorted(families)}")
    
    if defanged_dir.exists():
        motif_files = list(defanged_dir.glob("MOTIF_*"))
        print(f"\\n📁 Binary Files:")
        print(f"  MOTIF binary files found: {len(motif_files)}")
        if motif_files:
            print(f"  Sample files: {[f.name for f in motif_files[:5]]}")
    
    return {
        'has_jsonl': jsonl_file.exists(),
        'has_binaries': defanged_dir.exists() and len(list(defanged_dir.glob("MOTIF_*"))) > 0,
        'jsonl_path': str(jsonl_file) if jsonl_file.exists() else None,
        'binaries_path': str(defanged_dir) if defanged_dir.exists() else None
    }


def generate_usage_commands(analysis_result: dict, motif_dir: str):
    """Generate the appropriate commands to run with MOTIF dataset."""
    print("\\n=== Usage Recommendations ===")
    
    if analysis_result['has_jsonl'] and analysis_result['has_binaries']:
        print("✅ Complete MOTIF dataset found!")
        print("\\n🔧 Recommended Commands:")
        
        print("\\n1. Generate grayscale images (Nataraj methodology):")
        print(f"   python generate_image.py --data_dir {motif_dir} --output_dir ./motif_images --image_size 256")
        
        print("\\n2. Generate variable-size images (original Nataraj):")
        print(f"   python generate_image.py --data_dir {motif_dir} --output_dir ./motif_images_var --variable_size")
        
        print("\\n3. Verify methodology compliance:")
        print("   python verify_nataraj_images.py --image_dir ./motif_images/train/malware")
        
        print("\\n4. Visualize samples:")
        print("   python visualize_dataset.py --dataset_dir ./motif_images --num_samples 16")
        
        print("\\n⚠️  IMPORTANT NOTES FOR BINARY CLASSIFICATION:")
        print("   - MOTIF contains only malware samples (454 families)")
        print("   - For binary classification, you need benign samples")
        print("   - Consider one of these approaches:")
        print("     a) Add legitimate software to create 'benign' class")
        print("     b) Use for multi-class family classification instead")
        print("     c) Treat some families as 'benign' for binary classification")
        
    elif analysis_result['has_jsonl']:
        print("⚠️  Found labels but missing binary files")
        print("   - Extract MOTIF.7z with password: i_assume_all_risk_opening_malware")
        print("   - Ensure MOTIF_defanged directory contains binary files")
        
    elif analysis_result['has_binaries']:
        print("⚠️  Found binary files but missing labels")
        print("   - Missing motif_dataset.jsonl file")
        print("   - Will use generic file discovery (all files treated as malware)")
        
    else:
        print("❌ MOTIF dataset structure not found")
        print("   - Ensure you've downloaded and extracted the complete dataset")
        print("   - Check the directory structure")


def create_motif_script(output_path: str, motif_dir: str):
    """Create a ready-to-run script for MOTIF processing."""
    script_content = f'''#!/usr/bin/env python3
"""
Auto-generated MOTIF Processing Script
"""

import subprocess
import sys

# MOTIF dataset directory
MOTIF_DIR = "{motif_dir}"

def run_command(cmd, description):
    print(f"\\n{'='*50}")
    print(f"{{description}}")
    print(f"{'='*50}")
    print(f"Running: {{' '.join(cmd)}}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Warning: Command failed with exit code {{result.returncode}}")
    return result.returncode == 0

def main():
    print("MOTIF Dataset Processing Pipeline")
    
    # Step 1: Generate images
    success = run_command([
        sys.executable, "generate_image.py",
        "--data_dir", MOTIF_DIR,
        "--output_dir", "./motif_images", 
        "--image_size", "256",
        "--test_size", "0.2",
        "--verbose"
    ], "Step 1: Generating malware images using Nataraj methodology")
    
    if not success:
        print("Failed to generate images. Check your dataset path and try again.")
        return
    
    # Step 2: Verify methodology
    run_command([
        sys.executable, "verify_nataraj_images.py",
        "--image_dir", "./motif_images/train/malware",
        "--sample_size", "5"
    ], "Step 2: Verifying Nataraj methodology compliance")
    
    # Step 3: Visualize dataset
    run_command([
        sys.executable, "visualize_dataset.py", 
        "--dataset_dir", "./motif_images",
        "--num_samples", "16"
    ], "Step 3: Visualizing dataset samples")
    
    print("\\n" + "="*50)
    print("MOTIF processing complete!")
    print("Check ./motif_images/ for generated dataset")
    print("="*50)

if __name__ == "__main__":
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(output_path, 0o755)
    print(f"\\n✅ Created automated script: {output_path}")
    print(f"   Run with: python {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MOTIF Dataset Configuration Helper")
    parser.add_argument('motif_dir', help='Path to MOTIF dataset directory')
    parser.add_argument('--create-script', help='Create automated processing script', metavar='SCRIPT_PATH')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.motif_dir):
        print(f"Error: Directory not found: {args.motif_dir}")
        sys.exit(1)
    
    # Analyze dataset
    analysis = analyze_motif_dataset(args.motif_dir)
    
    # Generate usage recommendations
    generate_usage_commands(analysis, args.motif_dir)
    
    # Create script if requested
    if args.create_script:
        create_motif_script(args.create_script, args.motif_dir)
    
    print("\\n" + "="*60)
    print("Ready to process MOTIF dataset with Nataraj methodology!")
    print("="*60)


if __name__ == "__main__":
    main()