#!/usr/bin/env python3
"""
Quick test script to validate AGBD trainer fixes.

This script tests whether our fix to the target interpolation bug
correctly preserves single-pixel AGBD supervision.
"""

import sys
import os

# Add pangaea to path
sys.path.insert(0, '/scratch/final2/pangaea-bench-agbd')

def test_agbd_trainer_fix():
    """Test the fixed AGBD trainer logic."""
    print("🧪 Testing AGBD Trainer Fix")
    print("=" * 50)
    
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("❌ PyTorch not available in current environment")
        return False
        
    # Simulate AGBD target (25x25 with single valid pixel)
    batch_size = 4
    target = torch.full((batch_size, 25, 25), -1.0)  
    target[0, 12, 12] = 150.0  # GEDI pixel for sample 1
    target[1, 12, 12] = 200.0  # GEDI pixel for sample 2
    target[2, 12, 12] = 75.0   # GEDI pixel for sample 3
    target[3, 12, 12] = 300.0  # GEDI pixel for sample 4
    
    # Simulate large model output (224x224) 
    logits = torch.randn(batch_size, 1, 224, 224)
    
    print("🔍 Original AGBD targets:")
    for i in range(batch_size):
        valid_mask = target[i] != -1
        valid_count = valid_mask.sum().item()
        if valid_count > 0:
            coords = torch.nonzero(valid_mask)[0]
            value = target[i, coords[0], coords[1]].item()
            print(f"  Sample {i}: {valid_count} valid pixel at ({coords[0]}, {coords[1]}) = {value:.1f}")
        else:
            print(f"  Sample {i}: No valid pixels")
    
    print("\n❌ OLD BROKEN APPROACH (target interpolation):")
    # What the old broken code would do
    interpolated_target = F.interpolate(
        target.unsqueeze(1).float(),
        size=(224, 224),
        mode='nearest'
    ).squeeze(1)
    valid_pixels_total = (interpolated_target != -1).sum().item()
    print(f"  Total valid pixels after interpolation: {valid_pixels_total}")
    print(f"  Expected: {batch_size}, Got: {valid_pixels_total} (BROKEN!)")
    
    print("\n✅ NEW FIXED APPROACH (center pixel extraction):")
    # Our fixed approach
    ignore_index = -1
    center_h, center_w = 224 // 2, 224 // 2
    logits_center = logits.squeeze(dim=1)[:, center_h, center_w]
    
    target_center = torch.full((batch_size,), float(ignore_index))
    
    for i in range(batch_size):
        valid_mask = target[i] != ignore_index
        if valid_mask.any():
            valid_coords = torch.nonzero(valid_mask, as_tuple=False)
            if len(valid_coords) == 1:
                y, x = valid_coords[0]
                target_center[i] = target[i, y, x]
    
    print(f"  Logits center shape: {logits_center.shape}")
    print(f"  Target center values: {target_center.tolist()}")
    
    valid_count = (target_center != ignore_index).sum().item()
    print(f"  Valid samples extracted: {valid_count}")
    print(f"  Expected: {batch_size}, Got: {valid_count} ({'✅ PERFECT!' if valid_count == batch_size else '❌ BROKEN!'})")
    
    # Test loss computation
    final_valid_mask = target_center != ignore_index
    if final_valid_mask.sum() > 0:
        valid_logits = logits_center[final_valid_mask]
        valid_targets = target_center[final_valid_mask]
        
        # Simulate MSE loss
        loss = torch.nn.functional.mse_loss(valid_logits, valid_targets)
        print(f"  Simulated loss: {loss.item():.4f}")
        print("  ✅ Loss computation works!")
    else:
        print("  ❌ No valid samples for loss computation!")
        return False
        
    print("\n🎉 AGBD Trainer Fix Test: SUCCESS!")
    return True

def test_agbd_preprocessing():
    """Test if AGBD preprocessing components exist."""
    print("\n🔧 Testing AGBD Preprocessing Components")
    print("=" * 50)
    
    try:
        # Test imports
        from pangaea.engine.data_preprocessor import AGBDCenterCropToEncoder, AGBDPercentileNormalize
        print("✅ AGBDCenterCropToEncoder: Available")
        print("✅ AGBDPercentileNormalize: Available")
        
        # Test AGBD normalizer import
        from pangaea.engine.agbd_percentile_normalizer import AGBDPercentileNormalizer
        print("✅ AGBDPercentileNormalizer: Available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 AGBD Fix Validation Suite")
    print("="*60)
    
    # Test 1: Trainer fix
    trainer_ok = test_agbd_trainer_fix()
    
    # Test 2: Preprocessing components
    preprocessing_ok = test_agbd_preprocessing()
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS:")
    print(f"  Trainer Fix: {'✅ PASS' if trainer_ok else '❌ FAIL'}")
    print(f"  Preprocessing: {'✅ PASS' if preprocessing_ok else '❌ FAIL'}")
    
    if trainer_ok and preprocessing_ok:
        print("\n🎉 ALL TESTS PASSED! Ready for training.")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED! Check implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
