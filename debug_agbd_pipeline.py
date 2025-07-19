#!/usr/bin/env python3
"""
Comprehensive diagnostic script to trace AGBD normalization/scaling pipeline
and compare with original AGBD repo expectations.

This script will:
1. Load raw AGBD patches and examine value ranges
2. Trace preprocessing normalization step by step
3. Compare with expected ranges from original repo
4. Test different normalization strategies
5. Examine target biomass values and scaling
"""

import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path

# Add pangaea to path
sys.path.insert(0, '/scratch/final2/pangaea-bench-agbd')

from pangaea.datasets.agbd import AGBD
from pangaea.engine.data_preprocessor import RandomCropToEncoder
from omegaconf import OmegaConf

def load_agbd_config():
    """Load AGBD dataset configuration."""
    config_path = "/scratch/final2/pangaea-bench-agbd/configs/dataset/agbd.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def analyze_raw_data():
    """Analyze raw AGBD data before any preprocessing."""
    print("=" * 80)
    print("ANALYZING RAW AGBD DATA")
    print("=" * 80)
    
    # Load dataset config
    config = load_agbd_config()
    print(f"Config img_size: {config.get('img_size', 'NOT SET')}")
    print(f"Config bands: {config.get('bands', 'NOT SET')}")
    
    # Try to create dataset instance
    try:
        dataset = AGBD(
            split='test',
            dataset_name='AGBD',
            root_path='/scratch/final2/pangaea-bench-agbd',
            **config
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Sample a few patches
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            
            print(f"\nSample {i}:")
            print(f"  Keys: {list(sample.keys())}")
            
            if 'image' in sample:
                for modality, data in sample['image'].items():
                    print(f"  {modality} shape: {data.shape}")
                    print(f"  {modality} range: [{data.min():.6f}, {data.max():.6f}]")
                    print(f"  {modality} mean: {data.mean():.6f}")
            
            if 'target' in sample:
                print(f"  Target shape: {sample['target'].shape}")
                if sample['target'].numel() == 1:
                    print(f"  Target value: {sample['target'].item():.6f}")
                else:
                    print(f"  Target range: [{sample['target'].min():.6f}, {sample['target'].max():.6f}]")
                    print(f"  Target center: {sample['target'][12, 12].item():.6f}")
            
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

def test_normalization_strategies():
    """Test different normalization strategies from the config."""
    print("\n" + "=" * 80)
    print("TESTING CURRENT NORMALIZATION")
    print("=" * 80)
    
    config = load_agbd_config()
    
    print(f"Current data_mean: {config.get('data_mean', 'NOT SET')}")
    print(f"Current data_std: {config.get('data_std', 'NOT SET')}")
    print(f"Current data_min: {config.get('data_min', 'NOT SET')}")
    print(f"Current data_max: {config.get('data_max', 'NOT SET')}")
    
    # Compare with expected surface reflectance values
    print("\nEXPECTED SURFACE REFLECTANCE RANGES:")
    print("- Visible bands (B2,B3,B4): 0.05-0.4")
    print("- NIR band (B8): 0.3-0.8 (vegetation)")
    print("- SWIR bands (B11,B12): 0.1-0.4")
    print("- Red edge bands: 0.1-0.6")
    
    # Analyze if our config values are reasonable
    if 'data_mean' in config and 'optical' in config['data_mean']:
        means = config['data_mean']['optical']
        print(f"\nOur optical means: {means}")
        print(f"Mean range: [{min(means):.4f}, {max(means):.4f}]")
        
        # Check if these are reasonable for surface reflectance
        if max(means) < 0.05:
            print("⚠️  WARNING: Mean values are very low for surface reflectance!")
        elif max(means) > 0.5:
            print("⚠️  WARNING: Mean values are high for surface reflectance!")
        else:
            print("✅ Mean values look reasonable for surface reflectance")
    
    if 'data_std' in config and 'optical' in config['data_std']:
        stds = config['data_std']['optical']
        print(f"\nOur optical stds: {stds}")
        print(f"Std range: [{min(stds):.4f}, {max(stds):.4f}]")
        
        # Check if these are reasonable
        if max(stds) < 0.01:
            print("⚠️  WARNING: Std values are very low!")
        elif max(stds) > 0.2:
            print("⚠️  WARNING: Std values are high!")
        else:
            print("✅ Std values look reasonable")

def test_preprocessor_effects():
    """Test how the preprocessor affects the data."""
    print("\n" + "=" * 80)
    print("TESTING PREPROCESSOR EFFECTS")
    print("=" * 80)
    
    config = load_agbd_config()
    
    # Create dataset
    try:
        dataset = AGBD(
            split='test',
            dataset_name='AGBD',
            root_path='/scratch/final2/pangaea-bench-agbd',
            **config
        )
        
        sample = dataset[0]
        
        print("RAW DATASET OUTPUT:")
        print(f"Keys: {list(sample.keys())}")
        
        if 'image' in sample:
            for modality, data in sample['image'].items():
                print(f"{modality} shape: {data.shape}")
                print(f"{modality} range: [{data.min():.6f}, {data.max():.6f}]")
                print(f"{modality} mean: {data.mean():.6f}")
        
        if 'target' in sample:
            print(f"Target shape: {sample['target'].shape}")
            if sample['target'].numel() == 1:
                print(f"Target value: {sample['target'].item():.6f}")
            else:
                print(f"Target center: {sample['target'][12, 12].item():.6f}")
        
        # Test with RandomCropToEncoder (the preprocessor used in training)
        print("\nTESTING RandomCropToEncoder (32x32):")
        
        meta = {
            'encoder_input_size': 32,
            'data_mean': {'optical': torch.zeros(12)},
            'ignore_index': -1,
        }
        
        preprocessor = RandomCropToEncoder(pad_if_needed=True, **meta)
        
        # Apply preprocessing
        processed_sample = preprocessor(sample)
        
        print("AFTER PREPROCESSING:")
        if 'image' in processed_sample:
            for modality, data in processed_sample['image'].items():
                print(f"{modality} shape: {data.shape}")
                print(f"{modality} range: [{data.min():.6f}, {data.max():.6f}]")
                print(f"{modality} mean: {data.mean():.6f}")
        
        if 'target' in processed_sample:
            print(f"Target shape: {processed_sample['target'].shape}")
            if processed_sample['target'].numel() == 1:
                print(f"Target value: {processed_sample['target'].item():.6f}")
            else:
                print(f"Target center: {processed_sample['target'][15, 15].item():.6f}")  # Center in 32x32
        
    except Exception as e:
        print(f"Error in preprocessor test: {e}")
        import traceback
        traceback.print_exc()

def compare_with_original_repo_expectations():
    """Compare with what we know about the original AGBD repo."""
    print("\n" + "=" * 80)
    print("COMPARING WITH ORIGINAL AGBD REPO EXPECTATIONS")
    print("=" * 80)
    
    print("Based on the original AGBD repo analysis:")
    print("1. Sentinel-2 bands should be in surface reflectance (0-1 range)")
    print("2. Biomass targets should be in Mg/ha (typically 0-500+ range)")
    print("3. Original repo likely used different normalization strategy")
    print("4. Input patches are 25x25, likely padded to match model requirements")
    
    # Check current normalization config
    config = load_agbd_config()
    norm_config = config.get('normalization', {})
    
    print(f"\nCurrent PANGAEA normalization config:")
    print(f"Type: {norm_config.get('type', 'NOT SET')}")
    
    if norm_config.get('type') == 'mean_std':
        print(f"Mean: {norm_config.get('mean', 'NOT SET')}")
        print(f"Std: {norm_config.get('std', 'NOT SET')}")
        
        # These values look like they're for normalized/scaled data, not raw surface reflectance
        mean_values = norm_config.get('mean', [])
        if mean_values:
            print(f"Mean values range: [{min(mean_values):.4f}, {max(mean_values):.4f}]")
            print("NOTE: These means are quite low for surface reflectance (should be ~0.1-0.3)")
    
    print(f"\nTarget transform: {config.get('target_transform', 'NOT SET')}")
    
    # Test what surface reflectance values should look like
    print("\nEXPECTED SURFACE REFLECTANCE RANGES:")
    print("- Visible bands (B2,B3,B4): 0.05-0.4")
    print("- NIR band (B8): 0.3-0.8 (vegetation)")
    print("- SWIR bands (B11,B12): 0.1-0.4")
    print("- Red edge bands: 0.1-0.6")

def main():
    """Run comprehensive diagnostic analysis."""
    print("AGBD NORMALIZATION/SCALING DIAGNOSTIC")
    print("=" * 80)
    
    try:
        # 1. Analyze raw data
        analyze_raw_data()
        
        # 2. Test current normalization
        test_normalization_strategies()
        
        # 3. Test preprocessor effects
        test_preprocessor_effects()
        
        # 4. Compare with original repo expectations
        compare_with_original_repo_expectations()
        
        print("\n" + "=" * 80)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during diagnostic: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
