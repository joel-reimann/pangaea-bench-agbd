"""
Debug script to investigate tensor shapes and loss computation in AGBD training.

This script will help identify why the model is still producing zero predictions
despite all the normalization and center pixel fixes.
"""

import torch
import sys
import os
sys.path.append('/scratch/final2/pangaea-bench-agbd')

# Import AGBD dataset and check tensor shapes
from pangaea.datasets.agbd import AGBD
from pangaea.engine.data_preprocessor import Preprocessor, FocusRandomCropToEncoder, BandFilter, AGBDPercentileNormalize, BandPadding

def debug_decoder_output_size():
    """Check what spatial size the UperNet decoder actually outputs."""
    print("🏗️ Decoder Output Size Investigation")
    print("=" * 50)
    
    try:
        # Import decoder
        import sys
        sys.path.append('/scratch/final2/pangaea-bench-agbd')
        from pangaea.decoders.upernet import RegUPerNet
        
        # Create decoder (matching config)
        decoder = RegUPerNet(channels=512, finetune=False)
        print(f"✅ RegUPerNet decoder created successfully")
        
        # Test with different input sizes (from SatMAE encoder outputs)
        # SatMAE has patch_size=8, so for 96x96 input -> 12x12 features
        test_sizes = [
            (1, 2304, 12, 12),   # SatMAE output: 96/8 = 12 patches per dim
        ]
        
        for input_shape in test_sizes:
            print(f"\n  Input shape: {input_shape}")
            fake_input = torch.randn(input_shape)
            
            try:
                with torch.no_grad():
                    output = decoder(fake_input)
                print(f"  Output shape: {output.shape}")
                
                # Check if output spatial size matches expected target size
                if output.shape[-2:] == (25, 25):
                    print("    ✅ Matches 25x25 AGBD patch size")
                elif output.shape[-2:] == (32, 32):
                    print("    ✅ Matches 32x32 padded size")
                elif output.shape[-2:] == (96, 96):
                    print("    📏 Matches 96x96 encoder input size")
                else:
                    print(f"    ⚠️ Unusual size {output.shape[-2:]} - expected 25x25, 32x32, or 96x96")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
                
    except ImportError as e:
        print(f"❌ Could not import decoder: {e}")

def debug_center_pixel_calculation():
    """Debug center pixel calculation for different tensor sizes."""
    print("\n🎯 Center Pixel Calculation Debug")
    print("=" * 50)
    
    sizes = [(25, 25), (32, 32), (96, 96)]
    
    for height, width in sizes:
        print(f"\n  Size: {height}x{width}")
        
        # Our current logic from trainer.py
        if height == 25 and width == 25:
            center_h = center_w = 12
        elif height != width:
            center_h = height // 2
            center_w = width // 2
        else:
            center_h = height // 2
            center_w = width // 2
            
        print(f"    Center pixel: ({center_h}, {center_w})")
        
        # Check if center is valid
        if center_h < height and center_w < width:
            print(f"    ✅ Valid center pixel")
        else:
            print(f"    ❌ Invalid center pixel - out of bounds!")

def debug_loss_computation_issue():
    """Investigate potential issues with loss computation."""
    print("\n🔍 Loss Computation Analysis")
    print("=" * 50)
    
    # Common issue: target and prediction size mismatch
    print("  Scenario 1: Size mismatch")
    target_shape = (1, 25, 25)  # AGBD target
    pred_shape = (1, 1, 96, 96)  # Possible decoder output
    
    print(f"    Target shape: {target_shape}")
    print(f"    Prediction shape: {pred_shape}")
    
    if target_shape[-2:] != pred_shape[-2:]:
        print("    ❌ SPATIAL SIZE MISMATCH! This would cause loss computation errors.")
        print("    🔧 Solution: Ensure decoder outputs same spatial size as targets")
    
    print("\n  Scenario 2: Center pixel extraction")
    # Simulate correct sizes
    batch_size = 2
    
    # If decoder outputs 96x96 but target is 25x25, we have a problem
    fake_pred = torch.randn(batch_size, 1, 96, 96)
    fake_target = torch.randn(batch_size, 25, 25)
    
    print(f"    Pred shape: {fake_pred.shape}")
    print(f"    Target shape: {fake_target.shape}")
    
    # Center calculation for 96x96
    center_96 = 96 // 2  # = 48
    # Center calculation for 25x25
    center_25 = 12
    
    print(f"    Pred center (96x96): ({center_96}, {center_96})")
    print(f"    Target center (25x25): ({center_25}, {center_25})")
    
    # This would extract completely different spatial locations!
    pred_center_val = fake_pred[0, 0, center_96, center_96] 
    target_center_val = fake_target[0, center_25, center_25]
    
    print(f"    ❌ Extracting from different spatial locations!")
    print(f"    This explains why model learns nothing useful.")

if __name__ == "__main__":
    import torch
    
    debug_decoder_output_size()
    debug_center_pixel_calculation() 
    debug_loss_computation_issue()
    
    print("\n🎯 CRITICAL FINDINGS:")
    print("1. ❓ What spatial size does decoder actually output?")
    print("2. ❓ Does decoder output size match target size?") 
    print("3. ❗ If sizes don't match, center pixels are from different locations!")
    print("4. 🔧 Solution: Ensure consistent spatial dimensions in loss computation")
    """Check what spatial size the UperNet decoder actually outputs."""
    print("\n🏗️ Decoder Output Size Investigation")
    print("=" * 50)
    
    try:
        # Import decoder
        from pangaea.decoders.upernet import RegUPerNet
        
        # Create decoder (matching config)
        decoder = RegUPerNet(channels=512, finetune=False)
        
        # Test with different input sizes (from SatMAE encoder outputs)
        test_sizes = [
            (1, 2304, 12, 12),   # SatMAE output: 96/8 = 12 patches per dim
            (1, 2304, 24, 24),   # If input was 192
            (1, 2304, 32, 32),   # If input was 256
        ]
        
        for input_shape in test_sizes:
            print(f"\n  Input shape: {input_shape}")
            fake_input = torch.randn(input_shape)
            
            try:
                with torch.no_grad():
                    output = decoder(fake_input)
                print(f"  Output shape: {output.shape}")
                
                # Check if output spatial size matches expected target size
                if output.shape[-2:] == (25, 25):
                    print("    ✅ Matches 25x25 AGBD patch size")
                elif output.shape[-2:] == (32, 32):
                    print("    ✅ Matches 32x32 padded size")
                else:
                    print(f"    ⚠️ Unusual size - expected 25x25 or 32x32")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
                
    except ImportError as e:
        print(f"❌ Could not import decoder: {e}")

if __name__ == "__main__":
    debug_agbd_tensor_shapes()
    debug_decoder_output_size()
    
    print("\n🎯 Summary:")
    print("1. Check if decoder output spatial size matches target size")
    print("2. Verify center pixel locations are consistent")
    print("3. Ensure loss computation is using correct tensor dimensions")
    print("4. Consider if 16 training samples are sufficient for learning")
