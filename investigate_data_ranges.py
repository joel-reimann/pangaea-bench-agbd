#!/usr/bin/env python3
"""
Investigate AGBD data ranges and compare with other datasets.
Based on supervisor meeting notes about suspicious low data ranges.
"""

import numpy as np
import yaml
from pathlib import Path

def analyze_dataset_ranges():
    """Compare data ranges across datasets"""
    
    configs_dir = Path("configs/dataset")
    datasets = {}
    
    # Read all dataset configs
    for yaml_file in configs_dir.glob("*.yaml"):
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
                
            dataset_name = yaml_file.stem
            
            # Extract data ranges if available
            if 'data_mean' in config:
                datasets[dataset_name] = {
                    'data_mean': config.get('data_mean', {}),
                    'data_std': config.get('data_std', {}),
                    'data_min': config.get('data_min', {}),
                    'data_max': config.get('data_max', {}),
                    'img_size': config.get('img_size', 'N/A')
                }
        except Exception as e:
            print(f"Could not parse {yaml_file}: {e}")
    
    print("=== DATASET DATA RANGE COMPARISON ===\n")
    
    # Compare optical data ranges
    print("OPTICAL DATA RANGES:")
    print("=" * 60)
    for name, data in datasets.items():
        if 'optical' in data.get('data_mean', {}):
            optical_mean = data['data_mean']['optical']
            optical_max = data.get('data_max', {}).get('optical', 'N/A')
            optical_min = data.get('data_min', {}).get('optical', 'N/A')
            
            if isinstance(optical_mean, list) and len(optical_mean) > 0:
                mean_range = f"[{min(optical_mean):.4f}, {max(optical_mean):.4f}]"
            else:
                mean_range = str(optical_mean)
                
            if isinstance(optical_max, list) and len(optical_max) > 0:
                max_range = f"[{min(optical_max):.4f}, {max(optical_max):.4f}]"
            else:
                max_range = str(optical_max)
                
            print(f"{name:20} | img_size: {data['img_size']:>3} | mean: {mean_range:20} | max: {max_range}")
    
    print("\n" + "=" * 60)
    
    # Focus on AGBD specifically
    if 'agbd' in datasets:
        agbd = datasets['agbd']
        print("\nAGBD DETAILED ANALYSIS:")
        print("=" * 40)
        print(f"Image size: {agbd['img_size']} (ISSUE: not divisible by 24!)")
        
        optical_max = agbd.get('data_max', {}).get('optical', [])
        optical_min = agbd.get('data_min', {}).get('optical', [])
        
        if optical_max and optical_min:
            print(f"Optical range: [{min(optical_min):.4f}, {max(optical_max):.4f}]")
            print("Bands and their ranges:")
            bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
            for i, band in enumerate(bands):
                if i < len(optical_min) and i < len(optical_max):
                    print(f"  {band:>3}: [{optical_min[i]:.4f}, {optical_max[i]:.4f}]")
        
        # Check if values seem suspiciously low
        if optical_max:
            max_val = max(optical_max)
            if max_val < 10:  # Sentinel-2 is usually 0-10000 but can be normalized to 0-1
                print(f"\n⚠️  WARNING: Max optical value {max_val:.4f} seems low.")
                print("   This might indicate pre-normalized data or scaling issues.")
                print("   Typical Sentinel-2 ranges:")
                print("   - Raw DN: 0-10000+ (digital numbers)")  
                print("   - TOA reflectance: 0-1.0")
                print("   - Surface reflectance: 0-1.0 (can exceed 1.0)")
        
        sar_max = agbd.get('data_max', {}).get('sar', [])
        sar_min = agbd.get('data_min', {}).get('sar', [])
        
        if sar_max and sar_min:
            print(f"\nSAR range: [{min(sar_min):.1f}, {max(sar_max):.1f}] dB")
            print("SAR bands and their ranges:")
            sar_bands = ['HH', 'HV']
            for i, band in enumerate(sar_bands):
                if i < len(sar_min) and i < len(sar_max):
                    print(f"  {band:>2}: [{sar_min[i]:.1f}, {sar_max[i]:.1f}] dB")
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. AGBD img_size should be 24 (not 25) for ViT token alignment")
    print("2. Check if AGBD optical values need rescaling compared to other datasets")
    print("3. Verify Sentinel-2 preprocessing matches other benchmarks")
    print("4. Consider using larger patch sizes (48x48, 96x96) for better context")

if __name__ == "__main__":
    analyze_dataset_ranges()
