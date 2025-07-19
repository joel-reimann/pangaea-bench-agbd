#!/usr/bin/env python3
"""
Test script to validate AGBD padding behavior and ignore_index handling.
Verifies that center pixels are preserved and padded regions are handled correctly.
"""

import sys
import os
sys.path.append('/scratch/final2/pangaea-bench-agbd')

import torch
import numpy as np
from pangaea.engine.data_preprocessor import RandomCropToEncoder

def test_padding_behavior():
    """Test padding behavior with different input sizes"""
    
    print("=== AGBD PADDING BEHAVIOR TEST ===\n")
    
    # Create mock AGBD data (25x25 with biomass value at center)
    biomass_value = 123.456
    
    # Mock image data (optical bands)
    mock_image = {
        'optical': torch.randn(12, 1, 25, 25)  # 12 bands, 1 temporal, 25x25 spatial
    }
    
    # Mock target (25x25 filled with biomass value)
    mock_target = torch.full((25, 25), biomass_value, dtype=torch.float32)
    
    mock_data = {
        'image': mock_image,
        'target': mock_target
    }
    
    print(f"Original data:")
    print(f"  Image shape: {mock_image['optical'].shape}")
    print(f"  Target shape: {mock_target.shape}")
    print(f"  Center pixel (12,12): {mock_target[12, 12].item():.6f}")
    print(f"  Target range: [{mock_target.min().item():.6f}, {mock_target.max().item():.6f}]\n")
    
    # Test different target sizes
    target_sizes = [32, 48, 224]
    
    for target_size in target_sizes:
        print(f"=== Testing padding to {target_size}x{target_size} ===")
        
        # Create mock metadata for preprocessor
        meta = {
            'encoder_input_size': target_size,
            'data_mean': {'optical': torch.zeros(12)},  # Mock mean values
            'ignore_index': -9999,  # Standard ignore index
        }
        
        # Create preprocessor
        preprocessor = RandomCropToEncoder(pad_if_needed=True, **meta)
        
        # Apply preprocessing
        try:
            processed_data = preprocessor(mock_data.copy())
            
            # Analyze results
            processed_target = processed_data['target']
            processed_image = processed_data['image']['optical']
        except Exception as e:
            print(f"  ERROR during preprocessing: {e}")
            continue
        
        print(f"  Processed image shape: {processed_image.shape}")
        print(f"  Processed target shape: {processed_target.shape}")
        
        # Calculate new center position
        pad_total = target_size - 25
        pad_each = pad_total // 2
        new_center_y = 12 + pad_each
        new_center_x = 12 + pad_each
        
        print(f"  Expected new center: ({new_center_y}, {new_center_x})")
        print(f"  Center pixel value: {processed_target[new_center_y, new_center_x].item():.6f}")
        print(f"  Original biomass preserved: {abs(processed_target[new_center_y, new_center_x].item() - biomass_value) < 1e-6}")
        
        # Check padding values
        # Top-left corner should be padded
        pad_value = processed_target[0, 0].item()
        print(f"  Padding value (top-left): {pad_value:.6f}")
        print(f"  Is padding value = center value: {abs(pad_value - biomass_value) < 1e-6}")
        
        # Check ViT alignment
        alignment_16 = target_size % 16 == 0
        alignment_32 = target_size % 32 == 0
        print(f"  ViT-16 aligned: {alignment_16}")
        print(f"  ViT-32 aligned: {alignment_32}")
        
        print()

def test_ignore_index_handling():
    """Test that ignore_index values are handled correctly"""
    
    print("=== IGNORE INDEX HANDLING TEST ===\n")
    
    ignore_value = -9999
    valid_value = 50.0
    
    # Create target with some ignore_index values
    mock_target = torch.full((25, 25), valid_value, dtype=torch.float32)
    # Add some ignore_index values in corners
    mock_target[0:3, 0:3] = ignore_value
    mock_target[-3:, -3:] = ignore_value
    
    mock_image = {
        'optical': torch.randn(12, 1, 25, 25)
    }
    
    mock_data = {
        'image': mock_image,
        'target': mock_target
    }
    
    print(f"Original target stats:")
    print(f"  Shape: {mock_target.shape}")
    print(f"  Valid pixels: {(mock_target != ignore_value).sum().item()}")
    print(f"  Ignore pixels: {(mock_target == ignore_value).sum().item()}")
    print(f"  Center value: {mock_target[12, 12].item():.6f}")
    
    # Test with 32x32 padding
    meta = {
        'encoder_input_size': 32,
        'data_mean': {'optical': torch.zeros(12)},
        'ignore_index': ignore_value,
    }
    
    preprocessor = RandomCropToEncoder(pad_if_needed=True, **meta)
    processed_data = preprocessor(mock_data.copy())
    processed_target = processed_data['target']
    
    print(f"\nProcessed target stats:")
    print(f"  Shape: {processed_target.shape}")
    print(f"  Center value (15,15): {processed_target[15, 15].item():.6f}")
    print(f"  Min value: {processed_target.min().item():.6f}")
    print(f"  Max value: {processed_target.max().item():.6f}")
    
    # Check if any ignore_index values remain
    ignore_count = (processed_target == ignore_value).sum().item()
    print(f"  Ignore pixels after padding: {ignore_count}")
    
    if ignore_count > 0:
        print("  🚨 WARNING: ignore_index values still present after padding!")
    else:
        print("  ✅ No ignore_index values in padded result")

def test_center_pixel_preservation():
    """Specifically test center pixel preservation across different scenarios"""
    
    print("=== CENTER PIXEL PRESERVATION TEST ===\n")
    
    test_values = [0.0, 50.5, 123.456, 999.999]
    
    for biomass in test_values:
        print(f"Testing biomass value: {biomass}")
        
        # Create 25x25 target with specific center value
        mock_target = torch.full((25, 25), biomass, dtype=torch.float32)
        mock_image = {'optical': torch.randn(12, 1, 25, 25)}
        mock_data = {'image': mock_image, 'target': mock_target}
        
        # Test 32x32 padding
        meta = {
            'encoder_input_size': 32,
            'data_mean': {'optical': torch.zeros(12)},
            'ignore_index': -9999,
        }
        
        preprocessor = RandomCropToEncoder(pad_if_needed=True, **meta)
        processed = preprocessor(mock_data.copy())
        
        # Check center preservation
        new_center = processed['target'][15, 15].item()
        preserved = abs(new_center - biomass) < 1e-6
        
        print(f"  Original center (12,12): {biomass}")
        print(f"  New center (15,15): {new_center:.6f}")
        print(f"  Preserved: {preserved}")
        
        if not preserved:
            print(f"  🚨 ERROR: Center pixel not preserved!")
        
        print()

if __name__ == "__main__":
    test_padding_behavior()
    test_ignore_index_handling()
    test_center_pixel_preservation()
    
    print("=== SUMMARY ===")
    print("1. Test padding behavior for different target sizes")
    print("2. Test ignore_index handling in padded regions") 
    print("3. Test center pixel preservation with various biomass values")
    print("\nNext step: Update agbd.yaml with img_size: 32 and test with real models!")
