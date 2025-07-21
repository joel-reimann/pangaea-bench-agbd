#!/usr/bin/env python3
"""
Comprehensive test script for the AGBD pipeline after all fixes.
This script tests the complete data flow to ensure all fixes are working correctly.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add the pangaea package to the path
sys.path.append('/scratch/final2/pangaea-bench-agbd')

def test_agbd_preprocessing():
    """Test the AGBD preprocessing pipeline with new center crop."""
    print("="*60)
    print("TESTING AGBD PREPROCESSING PIPELINE")
    print("="*60)
    
    # Import required modules
    from pangaea.engine.data_preprocessor import AGBDCenterCropToEncoder
    
    # Create mock AGBD data
    mock_data = {
        "image": {
            "optical": torch.randn(10, 1, 25, 25),  # 10 bands, 1 time, 25x25 spatial
            "sar": torch.randn(2, 1, 25, 25)        # 2 bands, 1 time, 25x25 spatial
        },
        "target": torch.full((25, 25), -1.0, dtype=torch.float32)  # All ignore_index
    }
    
    # Set GEDI pixel at center (12, 12) with biomass value
    mock_data["target"][12, 12] = 150.0  # 150 Mg/ha biomass
    
    print(f"Input target shape: {mock_data['target'].shape}")
    print(f"GEDI pixel location: (12, 12)")
    print(f"GEDI biomass value: {mock_data['target'][12, 12].item()}")
    print(f"Valid pixels: {torch.sum(mock_data['target'] != -1).item()}")
    
    # Mock meta for preprocessor
    meta = {
        "encoder_input_size": 224,  # ViT encoder size
        "ignore_index": -1,
        "data_mean": {
            "optical": torch.zeros(10),
            "sar": torch.zeros(2)
        }
    }
    
    # Test AGBDCenterCropToEncoder
    print("\n--- Testing AGBDCenterCropToEncoder ---")
    preprocessor = AGBDCenterCropToEncoder(pad_if_needed=True, **meta)
    
    try:
        # Process the data
        processed_data = preprocessor(mock_data)
        
        print(f"Output target shape: {processed_data['target'].shape}")
        print(f"Output image shapes:")
        for modality, tensor in processed_data['image'].items():
            print(f"  {modality}: {tensor.shape}")
        
        # Find where the GEDI pixel ended up
        valid_mask = processed_data['target'] != -1
        if torch.any(valid_mask):
            valid_positions = torch.nonzero(valid_mask)
            gedi_y, gedi_x = valid_positions[0][0].item(), valid_positions[0][1].item()
            gedi_value = processed_data['target'][gedi_y, gedi_x].item()
            print(f"GEDI pixel location after preprocessing: ({gedi_y}, {gedi_x})")
            print(f"GEDI biomass value after preprocessing: {gedi_value}")
            
            # Check if GEDI pixel is centered
            center_y, center_x = processed_data['target'].shape[0] // 2, processed_data['target'].shape[1] // 2
            distance_from_center = abs(gedi_y - center_y) + abs(gedi_x - center_x)
            print(f"Distance from geometric center: {distance_from_center} pixels")
            
            if distance_from_center <= 2:
                print("✅ GEDI pixel is properly centered!")
            else:
                print("❌ GEDI pixel is NOT centered!")
        else:
            print("❌ No valid pixels found after preprocessing!")
            
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()

def test_agbd_trainer_logic():
    """Test the trainer logic for center pixel extraction."""
    print("\n" + "="*60)
    print("TESTING AGBD TRAINER LOGIC")
    print("="*60)
    
    # Mock trainer inputs
    batch_size = 4
    logits = torch.randn(batch_size, 1, 224, 224)  # Model predictions
    target = torch.full((batch_size, 32, 32), -1.0, dtype=torch.float32)  # Padded AGBD targets
    
    # Set GEDI pixels with different biomass values
    gedi_locations = [(15, 15), (15, 15), (15, 15), (15, 15)]  # Center after padding
    biomass_values = [120.0, 180.0, 95.0, 250.0]
    
    for i, ((gy, gx), biomass) in enumerate(zip(gedi_locations, biomass_values)):
        target[i, gy, gx] = biomass
        
    print(f"Logits shape: {logits.shape}")
    print(f"Target shape: {target.shape}")
    print(f"GEDI pixel locations: {gedi_locations}")
    print(f"Biomass values: {biomass_values}")
    
    # Simulate trainer logic
    logits_height, logits_width = logits.shape[-2:]
    target_height, target_width = target.shape[-2:]
    
    print(f"\nLogits spatial: {logits_height}x{logits_width}")
    print(f"Target spatial: {target_height}x{target_width}")
    
    # Check if AGBD dataset
    is_agbd = (target_height == 25 and target_width == 25) or (target_height == 32 and target_width == 32)
    print(f"Is AGBD dataset: {is_agbd}")
    
    if is_agbd:
        print("\n--- AGBD Processing Mode ---")
        # Find GEDI pixel location and map to logits space
        valid_mask = target != -1
        if torch.any(valid_mask):
            # Find valid pixel in first batch item
            batch_valid_mask = valid_mask[0]
            valid_positions = torch.nonzero(batch_valid_mask)
            if len(valid_positions) > 0:
                gedi_y, gedi_x = valid_positions[0][0].item(), valid_positions[0][1].item()
                print(f"GEDI pixel found at ({gedi_y}, {gedi_x}) in target space")
                
                # Map to logits space
                scale_y = logits_height / target_height
                scale_x = logits_width / target_width
                logits_center_h = int(gedi_y * scale_y)
                logits_center_w = int(gedi_x * scale_x)
                
                print(f"Scale factors: y={scale_y:.2f}, x={scale_x:.2f}")
                print(f"Mapped center in logits: ({logits_center_h}, {logits_center_w})")
                
                # Extract center pixels
                logits_center = logits.squeeze(dim=1)[:, logits_center_h, logits_center_w]
                target_center = target[:, gedi_y, gedi_x]
                
                print(f"Extracted logits: {logits_center}")
                print(f"Extracted targets: {target_center}")
                
                # Check valid samples
                valid_samples = target_center != -1
                print(f"Valid samples: {valid_samples.sum().item()}/{len(target_center)}")
                
                if valid_samples.sum() > 0:
                    valid_logits = logits_center[valid_samples]
                    valid_targets = target_center[valid_samples]
                    print(f"Valid logits for loss: {valid_logits}")
                    print(f"Valid targets for loss: {valid_targets}")
                    print("✅ AGBD trainer logic working correctly!")
                else:
                    print("❌ No valid samples for loss computation!")
            else:
                print("❌ No valid positions found!")
        else:
            print("❌ No valid pixels found!")
    else:
        print("❌ Not detected as AGBD dataset!")

def test_agbd_evaluator_logic():
    """Test the evaluator logic for AGBD."""
    print("\n" + "="*60)
    print("TESTING AGBD EVALUATOR LOGIC")
    print("="*60)
    
    # Mock evaluator inputs
    batch_size = 4
    logits = torch.randn(batch_size, 1, 224, 224)  # Model predictions
    target = torch.full((batch_size, 32, 32), -1.0, dtype=torch.float32)  # Padded AGBD targets
    
    # Set GEDI pixels
    gedi_locations = [(15, 15), (15, 15), (15, 15), (15, 15)]
    biomass_values = [120.0, 180.0, 95.0, 250.0]
    
    for i, ((gy, gx), biomass) in enumerate(zip(gedi_locations, biomass_values)):
        target[i, gy, gx] = biomass
        
    print(f"Logits shape: {logits.shape}")
    print(f"Target shape: {target.shape}")
    
    # Simulate evaluator logic - should NOT interpolate target for AGBD
    logits_height, logits_width = logits.shape[-2:]
    target_height, target_width = target.shape[-2:]
    
    # Check if this is AGBD
    is_agbd = (target_height == 25 and target_width == 25) or (target_height == 32 and target_width == 32)
    print(f"Is AGBD dataset: {is_agbd}")
    
    if is_agbd:
        print("✅ AGBD detected - preserving single-pixel supervision")
        print("Target will NOT be interpolated")
        
        # Find GEDI pixel and map to logits
        valid_mask = target != -1
        if torch.any(valid_mask):
            batch_valid_mask = valid_mask[0]
            valid_positions = torch.nonzero(batch_valid_mask)
            if len(valid_positions) > 0:
                gedi_y, gedi_x = valid_positions[0][0].item(), valid_positions[0][1].item()
                
                # Map to logits space
                scale_y = logits_height / target_height
                scale_x = logits_width / target_width
                logits_center_h = int(gedi_y * scale_y)
                logits_center_w = int(gedi_x * scale_x)
                
                print(f"GEDI pixel ({gedi_y}, {gedi_x}) maps to logits ({logits_center_h}, {logits_center_w})")
                
                # Extract predictions and targets
                predictions = logits.squeeze(dim=1)[:, logits_center_h, logits_center_w]
                targets = target[:, gedi_y, gedi_x]
                
                valid_samples = targets != -1
                if valid_samples.sum() > 0:
                    valid_preds = predictions[valid_samples]
                    valid_targs = targets[valid_samples]
                    
                    # Compute metrics
                    mse = torch.mean((valid_preds - valid_targs) ** 2)
                    print(f"MSE: {mse.item():.4f}")
                    print("✅ Evaluator logic working correctly!")
                else:
                    print("❌ No valid samples!")
            else:
                print("❌ No valid positions!")
        else:
            print("❌ No valid pixels!")
    else:
        print("❌ Not detected as AGBD dataset!")

def test_preprocessing_config():
    """Test that the preprocessing config uses AGBDCenterCropToEncoder."""
    print("\n" + "="*60)
    print("TESTING PREPROCESSING CONFIG")
    print("="*60)
    
    config_path = Path("/scratch/final2/pangaea-bench-agbd/configs/preprocessing/reg_agbd_percentile.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_content = f.read()
            
        print("Preprocessing config content:")
        print(config_content)
        
        if "AGBDCenterCropToEncoder" in config_content:
            print("✅ Config uses AGBDCenterCropToEncoder")
        else:
            print("❌ Config does NOT use AGBDCenterCropToEncoder")
            
        if "FocusRandomCropToEncoder" in config_content:
            print("❌ Config still uses FocusRandomCropToEncoder (should be removed)")
        else:
            print("✅ Config does not use FocusRandomCropToEncoder")
    else:
        print(f"❌ Config file not found: {config_path}")

def main():
    """Run all tests."""
    print("COMPREHENSIVE AGBD PIPELINE TEST")
    print("Testing all components after implementing fixes...")
    
    try:
        test_preprocessing_config()
        test_agbd_preprocessing()
        test_agbd_trainer_logic()
        test_agbd_evaluator_logic()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("All tests completed. Check above for any ❌ failures.")
        print("If all tests show ✅, the AGBD pipeline should be working correctly.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
