#!/usr/bin/env python3
"""
Investigate center pixel alignment precision in the AGBD pipeline.
Check if there are systematic offsets in coordinate mapping.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_center_pixel_mapping():
    """Analyze coordinate transformations through the pipeline."""
    
    print("="*60)
    print("AGBD CENTER PIXEL ALIGNMENT ANALYSIS")
    print("="*60)
    
    # Step 1: Original AGBD patch (25x25)
    original_center = (12, 12)
    print(f"1. Original AGBD center: {original_center} in 25×25 patch")
    
    # Step 2: After padding to 32x32 (as logged: 25→423 suggests some interpolation happens)
    # But we saw "Target shape: (423, 423)" which means something else is happening
    # Let's trace what we observed in logs:
    
    # From logs: "[AGBD CENTERCROP] Target shape: (423, 423)"
    # This suggests the 25x25 gets padded to some intermediate size first
    
    # Calculate padding to get from 25 to 423
    padding_per_side = (423 - 25) / 2
    print(f"2. Padding calculation: 25 → 423 requires {padding_per_side} pixels per side")
    
    # GEDI pixel location after padding
    gedi_in_423 = (12 + padding_per_side, 12 + padding_per_side)
    print(f"3. GEDI pixel in 423×423 space: {gedi_in_423}")
    
    # From logs: "[AGBD CENTERCROP] GEDI pixel found at: (211, 211)"
    logged_gedi = (211, 211)
    print(f"4. Logged GEDI pixel location: {logged_gedi}")
    print(f"   Difference from calculated: ({logged_gedi[0] - gedi_in_423[0]}, {logged_gedi[1] - gedi_in_423[1]})")
    
    # Step 3: After cropping 423x423 → 224x224
    # From logs: "[AGBD CENTERCROP] Crop window: i=99, j=99 (top-left corner)"
    crop_start = (99, 99)
    crop_size = (224, 224)
    
    print(f"5. Crop window: start={crop_start}, size={crop_size}")
    
    # GEDI pixel in cropped space
    gedi_in_crop = (logged_gedi[0] - crop_start[0], logged_gedi[1] - crop_start[1])
    print(f"6. GEDI pixel in 224×224 crop: {gedi_in_crop}")
    
    # Expected center of 224x224
    expected_center = (224//2, 224//2)
    print(f"7. Expected center of 224×224: {expected_center}")
    
    # Calculate offset
    offset = (gedi_in_crop[0] - expected_center[0], gedi_in_crop[1] - expected_center[1])
    print(f"8. Offset from exact center: {offset} pixels")
    
    if abs(offset[0]) <= 1 and abs(offset[1]) <= 1:
        print("✅ GEDI pixel is very well centered (≤1 pixel offset)")
    elif abs(offset[0]) <= 2 and abs(offset[1]) <= 2:
        print("✅ GEDI pixel is reasonably centered (≤2 pixel offset)")
    else:
        print("❌ GEDI pixel has significant offset from center")
    
    print("\n" + "="*60)
    print("ANALYSIS: MODEL PREDICTION PATTERNS")
    print("="*60)
    
    # Analyze the prediction patterns we observed
    predictions = [84.68, 149.73, 100.12, 188.29, 115.67, 154.12, 161.52, 152.20, 178.93, 190.12]
    ground_truth = [289.43, 239.67, 138.60, 111.22, 481.71, 226.46, 173.70, 224.60, 101.35, 82.75]
    
    print(f"Prediction range: {min(predictions):.1f} - {max(predictions):.1f} Mg/ha (span: {max(predictions)-min(predictions):.1f})")
    print(f"Ground truth range: {min(ground_truth):.1f} - {max(ground_truth):.1f} Mg/ha (span: {max(ground_truth)-min(ground_truth):.1f})")
    print(f"Prediction mean: {np.mean(predictions):.1f} Mg/ha")
    print(f"Ground truth mean: {np.mean(ground_truth):.1f} Mg/ha")
    
    # Calculate correlation
    correlation = np.corrcoef(predictions, ground_truth)[0,1]
    print(f"Correlation between pred and GT: {correlation:.3f}")
    
    # Analyze bias
    bias = np.mean(predictions) - np.mean(ground_truth)
    print(f"Mean bias (pred - gt): {bias:.1f} Mg/ha")
    
    # Range compression analysis
    pred_range = max(predictions) - min(predictions)
    gt_range = max(ground_truth) - min(ground_truth)
    compression_ratio = pred_range / gt_range
    print(f"Range compression ratio: {compression_ratio:.3f} (1.0 = no compression)")
    
    if compression_ratio < 0.5:
        print("❌ Strong range compression - model is conservative")
    elif compression_ratio < 0.8:
        print("⚠️ Moderate range compression - typical for regression")
    else:
        print("✅ Good range preservation")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if abs(offset[0]) <= 1 and abs(offset[1]) <= 1:
        print("✅ Center pixel alignment is excellent - no further action needed")
    else:
        print("🔧 Consider investigating coordinate mapping precision")
    
    if correlation > 0.5:
        print("✅ Model shows good correlation with ground truth")
    else:
        print("⚠️ Model correlation could be improved")
    
    if compression_ratio < 0.5:
        print("🔧 Consider techniques to improve prediction range:")
        print("   - Increase model capacity")
        print("   - Adjust loss function")
        print("   - Check for gradient clipping")
        print("   - Try different activation functions")
    
    return {
        'center_offset': offset,
        'correlation': correlation,
        'compression_ratio': compression_ratio,
        'bias': bias
    }

def create_prediction_analysis_plot():
    """Create visualization of prediction vs ground truth patterns."""
    
    predictions = [84.68, 149.73, 100.12, 188.29, 115.67, 154.12, 161.52, 152.20, 178.93, 190.12]
    ground_truth = [289.43, 239.67, 138.60, 111.22, 481.71, 226.46, 173.70, 224.60, 101.35, 82.75]
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(ground_truth, predictions, alpha=0.7, s=50)
    plt.plot([0, 500], [0, 500], 'r--', alpha=0.7, label='Perfect prediction')
    plt.xlabel('Ground Truth (Mg/ha)')
    plt.ylabel('Predictions (Mg/ha)')
    plt.title('Predictions vs Ground Truth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(2, 2, 2)
    residuals = np.array(predictions) - np.array(ground_truth)
    plt.scatter(ground_truth, residuals, alpha=0.7, s=50)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Ground Truth (Mg/ha)')
    plt.ylabel('Residuals (Pred - GT)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Histogram of predictions
    plt.subplot(2, 2, 3)
    plt.hist(predictions, bins=8, alpha=0.7, label='Predictions', color='blue')
    plt.hist(ground_truth, bins=8, alpha=0.7, label='Ground Truth', color='orange')
    plt.xlabel('Biomass (Mg/ha)')
    plt.ylabel('Frequency')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Time series (sample order)
    plt.subplot(2, 2, 4)
    sample_indices = range(len(predictions))
    plt.plot(sample_indices, predictions, 'b-o', label='Predictions', markersize=4)
    plt.plot(sample_indices, ground_truth, 'r-s', label='Ground Truth', markersize=4)
    plt.xlabel('Sample Index')
    plt.ylabel('Biomass (Mg/ha)')
    plt.title('Predictions vs GT by Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = '/scratch/final2/pangaea-bench-agbd/agbd_prediction_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Analysis plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    results = analyze_center_pixel_mapping()
    create_prediction_analysis_plot()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Center offset: {results['center_offset']} pixels")
    print(f"   Correlation: {results['correlation']:.3f}")
    print(f"   Range compression: {results['compression_ratio']:.3f}")
    print(f"   Mean bias: {results['bias']:.1f} Mg/ha")
