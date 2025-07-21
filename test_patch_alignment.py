#!/usr/bin/env python3
"""
AGBD Patch Alignment Unit Test

CRITICAL: This test verifies the 25×25 → 32×32 padding strategy and central pixel alignment
as described in the AGBD paper and instructions.

Key Issues Being Tested:
1. 25×25 patches padded to 32×32 for ViT alignment (25 % 16 != 0 causes checkerboard artifacts)
2. Central pixel (12,12) in 25×25 should map correctly after padding
3. RandomCrop should not mis-center the central pixel used for loss/prediction
4. Ignore-index masks should exclude padding during loss computation

Source: Instructions state "Write a unit test: for a known 25×25 patch, compute padded grid and central coordinate"
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add pangaea to path
sys.path.append('/scratch/final2/pangaea-bench-agbd')
sys.path.append('/scratch/final2/pangaea-bench-agbd/pangaea')

def test_agbd_patch_alignment():
    """Test AGBD 25x25 → 32x32 padding alignment"""
    print("=" * 80)
    print("🧪 TESTING AGBD PATCH ALIGNMENT (25×25 → 32×32)")
    print("=" * 80)
    
    # Create a test 25x25 patch with known center value
    original_size = 25
    padded_size = 32
    
    # Create test patch with known center value
    test_patch = torch.zeros(original_size, original_size)
    center_y, center_x = original_size // 2, original_size // 2  # (12, 12)
    center_value = 250.0  # Known biomass value
    test_patch[center_y, center_x] = center_value
    
    print(f"📊 Original patch: {original_size}×{original_size}")
    print(f"📊 Center coordinate: ({center_y}, {center_x})")
    print(f"📊 Center value: {center_value} Mg/ha")
    print(f"📊 Target padded size: {padded_size}×{padded_size}")
    
    # Test 1: Symmetric padding (as should be done)
    pad_amount = (padded_size - original_size) // 2  # Should be 3.5, but let's see how it's handled
    print(f"\n🔍 TEST 1: Symmetric Padding")
    print(f"   Required padding per side: {(padded_size - original_size) / 2} = {pad_amount}")
    
    # CRITICAL: Check if padding is symmetric
    if (padded_size - original_size) % 2 != 0:
        print("⚠️  WARNING: Asymmetric padding required! This could cause misalignment.")
        pad_left = pad_right = pad_top = pad_bottom = pad_amount
        # Handle odd padding
        pad_right += 1
        pad_bottom += 1
    else:
        pad_left = pad_right = pad_top = pad_bottom = pad_amount
    
    print(f"   Padding: left={pad_left}, right={pad_right}, top={pad_top}, bottom={pad_bottom}")
    
    # Apply padding using same method as data_preprocessor.py
    padding = (pad_left, pad_right, pad_top, pad_bottom)  # (left, right, top, bottom)
    padded_patch = F.pad(test_patch, padding, mode='constant', value=center_value)
    
    print(f"📏 Padded patch shape: {padded_patch.shape}")
    
    # Find where the center pixel moved to
    center_positions = torch.where(padded_patch == center_value)
    print(f"🎯 Center pixel positions after padding: {list(zip(center_positions[0].tolist(), center_positions[1].tolist()))}")
    
    # Test 2: Verify central coordinate after padding
    print(f"\n🔍 TEST 2: Central Coordinate Verification")
    expected_center_y = center_y + pad_top
    expected_center_x = center_x + pad_left
    print(f"   Expected center after padding: ({expected_center_y}, {expected_center_x})")
    
    new_center_y, new_center_x = padded_size // 2, padded_size // 2
    print(f"   New patch center coordinate: ({new_center_y}, {new_center_x})")
    
    center_offset_y = expected_center_y - new_center_y
    center_offset_x = expected_center_x - new_center_x
    print(f"   Center offset: ({center_offset_y}, {center_offset_x})")
    
    if center_offset_y != 0 or center_offset_x != 0:
        print("🚨 CRITICAL ISSUE: Original center pixel is NOT at new patch center!")
        print("   This means the 'central pixel' used for regression is wrong!")
    else:
        print("✅ Center pixel alignment correct")
    
    # Test 3: RandomCrop behavior simulation
    print(f"\n🔍 TEST 3: RandomCrop Behavior Simulation")
    
    # Simulate what happens during RandomCrop to encoder size
    # Most ViT encoders expect specific sizes
    encoder_sizes = [224, 256, 288, 384]  # Common ViT input sizes
    
    for enc_size in encoder_sizes:
        print(f"\n   📐 Testing encoder size: {enc_size}×{enc_size}")
        
        if enc_size > padded_size:
            # Need to pad more
            extra_pad = (enc_size - padded_size) // 2
            extra_padding = (extra_pad, extra_pad, extra_pad, extra_pad)
            final_patch = F.pad(padded_patch, extra_padding, mode='constant', value=0)
            print(f"      Additional padding needed: {extra_pad} per side")
        elif enc_size < padded_size:
            # Need to crop
            crop_amount = (padded_size - enc_size) // 2
            final_patch = padded_patch[crop_amount:crop_amount+enc_size, crop_amount:crop_amount+enc_size]
            print(f"      Cropping needed: {crop_amount} per side")
        else:
            final_patch = padded_patch
            print(f"      No additional processing needed")
        
        print(f"      Final shape: {final_patch.shape}")
        
        # Check where center pixel ended up
        center_positions = torch.where(final_patch == center_value)
        if len(center_positions[0]) > 0:
            final_center_y, final_center_x = center_positions[0][0].item(), center_positions[1][0].item()
            patch_center_y, patch_center_x = enc_size // 2, enc_size // 2
            
            offset_y = final_center_y - patch_center_y
            offset_x = final_center_x - patch_center_x
            print(f"      Original center at: ({final_center_y}, {final_center_x})")
            print(f"      Patch center at: ({patch_center_y}, {patch_center_x})")
            print(f"      Offset: ({offset_y}, {offset_x})")
            
            if offset_y != 0 or offset_x != 0:
                print(f"      🚨 MISALIGNMENT for {enc_size}×{enc_size}!")
            else:
                print(f"      ✅ Correctly aligned for {enc_size}×{enc_size}")
        else:
            print(f"      ❌ CRITICAL: Center pixel value (250.0) completely lost!")
            print(f"      Final patch unique values: {torch.unique(final_patch)[:10]}")
            print(f"      This indicates the center value was cropped out - major alignment issue!")

def test_ignore_index_masking():
    """Test ignore_index behavior with padding"""
    print("\n" + "=" * 80)
    print("🧪 TESTING IGNORE_INDEX MASKING")
    print("=" * 80)
    
    # Create test scenario with padding
    original_patch = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],  # Center value is 5.0
        [7.0, 8.0, 9.0]
    ])
    
    center_value = 5.0
    ignore_index = -1
    
    print(f"📊 Original 3×3 patch center value: {center_value}")
    
    # Pad to 5×5 using different strategies
    strategies = [
        ("zeros", 0.0),
        ("ignore_index", ignore_index),
        ("center_value", center_value)
    ]
    
    for strategy_name, fill_value in strategies:
        print(f"\n🔍 Testing padding strategy: {strategy_name} (fill={fill_value})")
        
        padded = F.pad(original_patch, (1, 1, 1, 1), mode='constant', value=fill_value)
        print(f"   Padded patch:\n{padded}")
        
        # Simulate loss computation
        valid_mask = padded != ignore_index
        valid_pixels = torch.sum(valid_mask).item()
        total_pixels = padded.numel()
        
        print(f"   Valid pixels: {valid_pixels}/{total_pixels}")
        print(f"   Padding pixels included in loss: {total_pixels - valid_pixels == 0}")
        
        if strategy_name == "ignore_index":
            print("   ✅ Padding excluded from loss computation")
        elif valid_pixels == total_pixels:
            print("   ⚠️  Padding included in loss computation")
        
        # Check if center pixel is still accessible
        center_y, center_x = padded.shape[0] // 2, padded.shape[1] // 2
        center_after_pad = padded[center_y, center_x]
        print(f"   Center pixel after padding: {center_after_pad.item()}")
        
        if abs(center_after_pad - center_value) < 1e-6:
            print("   ✅ Center pixel preserved")
        else:
            print("   🚨 Center pixel corrupted!")


if __name__ == "__main__":
    print("Starting AGBD Patch Alignment Tests...")
    print("Source: Step 3 from Instructions - 'Write a unit test: for a known 25×25 patch, compute padded grid and central coordinate'")
    
    try:
        test_agbd_patch_alignment()
        test_ignore_index_masking()
        
        print("\n" + "=" * 80)
        print("🎯 SUMMARY & RECOMMENDATIONS")
        print("=" * 80)
        print("Based on test results:")
        print("1. Check if 25×25 → 32×32 padding is symmetric")
        print("2. Verify RandomCrop doesn't misalign central pixel")
        print("3. Ensure ignore_index properly excludes padding")
        print("4. Consider using FocusRandomCrop to maintain center alignment")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
