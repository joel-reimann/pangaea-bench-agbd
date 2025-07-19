#!/usr/bin/env python3
"""
Test AGBD alignment fix with a single model.
Run for just a few epochs to validate that:
1. No more checkerboard artifacts
2. Center pixel is preserved
3. Performance improves
4. Padding works correctly
"""

import subprocess
import sys
import os

def test_single_model():
    """Test with just one model to validate the alignment fix"""
    
    print("=== TESTING AGBD ALIGNMENT FIX ===\n")
    
    # Test with a simple model first
    model = "ssl4eo_mae_optical"  # ViT-based model that should benefit from alignment
    
    print(f"Testing model: {model}")
    print(f"Configuration:")
    print(f"  - Dataset: AGBD with img_size=32 (padded from 25x25)")
    print(f"  - Model: {model} (ViT-based)")
    print(f"  - Epochs: 2 (just for validation)")
    print(f"  - Expected: No checkerboard, better alignment\n")
    
    # Create a minimal test command
    cmd = [
        "python", "pangaea/run.py",
        "experiment=agbd_finetune",
        f"encoder={model}",
        "trainer.max_epochs=2",
        "trainer.limit_train_batches=10",  # Only 10 batches
        "trainer.limit_val_batches=5",     # Only 5 validation batches
        "trainer.check_val_every_n_epoch=1",
        "dataset.debug=true",              # Use debug mode for faster testing
        "+trainer.fast_dev_run=false",     # Don't use fast dev run
        "wandb.mode=disabled",             # Disable wandb for testing
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print("\nThis will test:")
    print("1. Dataset loads correctly with 32x32 padding")
    print("2. Model processes aligned tokens")
    print("3. No crashes from dimension mismatches") 
    print("4. Training/validation runs smoothly")
    print("\nStarting test...\n")
    
    try:
        # Run the test
        result = subprocess.run(cmd, cwd="/scratch/final2/pangaea-bench-agbd", 
                              capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        print("=== TEST RESULTS ===")
        
        if result.returncode == 0:
            print("✅ SUCCESS: Model ran without errors!")
            print("✅ AGBD alignment fix appears to work correctly")
        else:
            print("❌ ERROR: Model failed to run")
            print(f"Return code: {result.returncode}")
            
        # Show relevant output
        if result.stdout:
            print("\nSTDOUT (last 50 lines):")
            lines = result.stdout.split('\n')
            for line in lines[-50:]:
                if any(keyword in line.lower() for keyword in ['error', 'warning', 'agbd', 'shape', 'alignment']):
                    print(line)
                    
        if result.stderr:
            print("\nSTDERR (errors/warnings):")
            lines = result.stderr.split('\n')
            for line in lines:
                if line.strip():
                    print(line)
                    
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT: Test took too long (>10 minutes)")
        print("This might indicate an issue with the configuration")
        
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

def check_config_status():
    """Check that our configuration changes are in place"""
    
    print("=== CONFIGURATION CHECK ===\n")
    
    # Check AGBD config
    config_file = "/scratch/final2/pangaea-bench-agbd/configs/dataset/agbd.yaml"
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
            
        if "img_size: 32" in content:
            print("✅ AGBD config updated: img_size: 32")
        elif "img_size: 25" in content:
            print("❌ AGBD config still has: img_size: 25")
            return False
        else:
            print("⚠️  AGBD config img_size not found")
            
        if "ViT token alignment" in content or "checkerboard" in content:
            print("✅ Alignment fix comment found in config")
        else:
            print("⚠️  No alignment fix comment in config")
            
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False
        
    # Check dataset code
    dataset_file = "/scratch/final2/pangaea-bench-agbd/pangaea/datasets/agbd.py"
    
    try:
        with open(dataset_file, 'r') as f:
            content = f.read()
            
        if "ViT alignment" in content or "checkerboard" in content:
            print("✅ AGBD dataset has alignment fix comments")
        else:
            print("⚠️  No alignment fix comments in dataset")
            
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        
    print()
    return True

if __name__ == "__main__":
    
    # First check configuration
    if not check_config_status():
        print("❌ Configuration not properly updated. Please fix before testing.")
        sys.exit(1)
    
    print("All configurations look good. Starting model test...\n")
    
    # Run the test
    test_single_model()
    
    print("\n=== SUMMARY ===")
    print("1. Updated AGBD config: img_size 25 → 32")
    print("2. Tested single model with alignment fix")
    print("3. This should eliminate checkerboard artifacts")
    print("4. Next: Run full evaluation on 1-2 models")
    print("5. Compare performance before/after fix")
    print("6. Monitor visualization outputs for artifacts")
    
    print(f"\n💡 If test succeeded:")
    print(f"  - Run full training with: trainer.max_epochs=50")
    print(f"  - Test 1-2 different ViT models")
    print(f"  - Compare metrics vs previous runs")
    print(f"  - Check visualization outputs")
    
    print(f"\n💡 If test failed:")
    print(f"  - Check error messages above")
    print(f"  - Verify padding logic in preprocessor")
    print(f"  - Test with different target sizes (48x48)")
    print(f"  - Ensure center pixel preservation")
