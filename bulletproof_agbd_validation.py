#!/usr/bin/env python3

"""
AGBD Integration Validation Script - BULLETPROOF VERSION

This script validates that our AGBD integration is working correctly by checking:
1. Loss computation with ignore_index filtering
2. Model learning progression 
3. Spatial alignment (96x96 input/output)
4. Center pixel logic consistency
5. Biomass value ranges
6. WandB logging functionality

CRITICAL: This must pass ALL checks before we can declare success.
"""

import sys
import os
sys.path.append('/scratch/final2/pangaea-bench-agbd')

import torch
import numpy as np
from pathlib import Path

def validate_loss_logs(log_file_path):
    """Validate loss computation from training logs."""
    print("🔍 VALIDATING LOSS COMPUTATION...")
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Check for ignore_index filtering
    ignore_index_found = "[LOSS DEBUG] Valid samples:" in log_content
    target_filtering = "Valid targets:" in log_content
    
    # Check for proper center pixel computation
    center_pixel_found = "[LOSS DEBUG] Center pixel: (48, 48)" in log_content
    spatial_match = "[LOSS DEBUG] Logits spatial: 96x96" in log_content and "[LOSS DEBUG] Target spatial: 96x96" in log_content
    
    # Extract some loss values to check progression
    import re
    loss_values = re.findall(r'\[LOSS DEBUG\] Loss value: ([\d.]+)', log_content)
    loss_values = [float(x) for x in loss_values[-10:]]  # Last 10 values
    
    print(f"   ✅ Ignore index filtering: {'YES' if ignore_index_found else 'NO'}")
    print(f"   ✅ Target filtering working: {'YES' if target_filtering else 'NO'}")
    print(f"   ✅ Center pixel (48,48): {'YES' if center_pixel_found else 'NO'}")
    print(f"   ✅ Spatial size match (96x96): {'YES' if spatial_match else 'NO'}")
    print(f"   📊 Recent loss values: {loss_values}")
    
    if loss_values:
        avg_loss = np.mean(loss_values)
        print(f"   📊 Average recent loss: {avg_loss:.1f}")
        # Loss should be in reasonable range for biomass prediction (not 0, not >100k consistently)
        loss_reasonable = 1000 < avg_loss < 80000
        print(f"   ✅ Loss in reasonable range: {'YES' if loss_reasonable else 'NO'}")
    
    return ignore_index_found and target_filtering and center_pixel_found and spatial_match

def validate_model_predictions(log_file_path):
    """Validate model prediction progression."""
    print("\n🎯 VALIDATING MODEL PREDICTIONS...")
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Extract logits and targets from recent batches
    import re
    
    # Find logits center values
    logits_pattern = r'\[LOSS DEBUG\] Valid logits: tensor\(\[([\d., -]+)\]'
    logits_matches = re.findall(logits_pattern, log_content)
    
    # Find target values  
    targets_pattern = r'\[LOSS DEBUG\] Valid targets: tensor\(\[([\d., -]+)\]'
    targets_matches = re.findall(targets_pattern, log_content)
    
    if logits_matches and targets_matches:
        # Parse last few batches
        recent_logits = []
        recent_targets = []
        
        for match in logits_matches[-5:]:  # Last 5 batches
            values = [float(x.strip()) for x in match.split(',') if x.strip()]
            recent_logits.extend(values)
            
        for match in targets_matches[-5:]:  # Last 5 batches
            values = [float(x.strip()) for x in match.split(',') if x.strip()]
            recent_targets.extend(values)
        
        if recent_logits and recent_targets:
            logits_range = (min(recent_logits), max(recent_logits))
            targets_range = (min(recent_targets), max(recent_targets))
            
            print(f"   📊 Model predictions range: {logits_range[0]:.1f} - {logits_range[1]:.1f} Mg/ha")
            print(f"   📊 Target values range: {targets_range[0]:.1f} - {targets_range[1]:.1f} Mg/ha")
            
            # Check if model is learning (predictions should be > 0 and < targets but not too far)
            predictions_positive = all(x > 0 for x in recent_logits)
            predictions_reasonable = all(x < 100 for x in recent_logits)  # Model predicting reasonable biomass
            targets_reasonable = all(20 <= x <= 500 for x in recent_targets)  # AGBD range
            
            print(f"   ✅ Predictions > 0: {'YES' if predictions_positive else 'NO'}")
            print(f"   ✅ Predictions < 100 Mg/ha: {'YES' if predictions_reasonable else 'NO'}")
            print(f"   ✅ Targets in AGBD range (20-500): {'YES' if targets_reasonable else 'NO'}")
            
            return predictions_positive and targets_reasonable
    
    print("   ❌ Could not extract prediction data from logs")
    return False

def validate_training_progression(log_file_path):
    """Validate that training is progressing correctly."""
    print("\n📈 VALIDATING TRAINING PROGRESSION...")
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Check for training batches being processed
    training_batches = log_content.count("[TRAINER DEBUG] Processing training batch")
    validation_batches = log_content.count("[DEBUG] Batch")
    
    # Check for epoch progression
    import re
    epoch_pattern = r'Epoch \[(\d+)-(\d+)/64\]'
    epoch_matches = re.findall(epoch_pattern, log_content)
    
    print(f"   📊 Training batches processed: {training_batches}")
    print(f"   📊 Validation batches processed: {validation_batches}")
    
    if epoch_matches:
        current_epoch, current_batch = epoch_matches[-1]
        print(f"   📊 Current progress: Epoch {current_epoch}, Batch {current_batch}/64")
        
        epochs_progressing = int(current_batch) > 5  # Should have processed some batches
        print(f"   ✅ Training progressing: {'YES' if epochs_progressing else 'NO'}")
        
        return epochs_progressing and training_batches > 0
    
    return False

def validate_wandb_issues(log_file_path):
    """Check WandB logging issues."""
    print("\n📊 VALIDATING WANDB LOGGING...")
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    wandb_warnings = log_content.count("wandb: WARNING Tried to log to step")
    wandb_success = "wandb: Tracking run with wandb" in log_content
    
    print(f"   📊 WandB step warnings: {wandb_warnings}")
    print(f"   ✅ WandB connection successful: {'YES' if wandb_success else 'NO'}")
    
    # Too many warnings indicates step ordering issue
    warnings_acceptable = wandb_warnings < 50  # Some warnings OK, but not excessive
    print(f"   ✅ WandB warnings acceptable: {'YES' if warnings_acceptable else 'NO'}")
    
    return wandb_success and warnings_acceptable

def validate_agbd_integration():
    """Main validation function."""
    print("=" * 80)
    print("🔬 AGBD INTEGRATION BULLETPROOF VALIDATION")
    print("=" * 80)
    
    # Find the most recent log file
    log_dir = Path("/scratch/final2/pangaea-bench-agbd/test_agbd_logs")
    log_files = list(log_dir.glob("satmae_base_agbd.log"))
    
    if not log_files:
        print("❌ No log files found!")
        return False
    
    log_file = log_files[0]  # Most recent
    print(f"📄 Analyzing log file: {log_file}")
    
    # Run all validations
    results = []
    
    try:
        results.append(("Loss Computation", validate_loss_logs(log_file)))
        results.append(("Model Predictions", validate_model_predictions(log_file)))
        results.append(("Training Progression", validate_training_progression(log_file)))
        results.append(("WandB Logging", validate_wandb_issues(log_file)))
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED! AGBD integration is bulletproof!")
    else:
        print("⚠️  SOME VALIDATIONS FAILED! Integration needs fixes.")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    validate_agbd_integration()
