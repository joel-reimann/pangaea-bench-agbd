#!/usr/bin/env python3
"""
Test the critical normalization fix for AGBD dataset.
This should solve the ~0 Mg/ha prediction problem.
"""

import sys
sys.path.insert(0, '/scratch/final2/pangaea-bench-agbd')

import torch
import numpy as np
from pangaea.datasets.agbd import normalize_data

def test_normalization_fix():
    """Test that percentile normalization gives reasonable ranges"""
    
    print("=== TESTING AGBD NORMALIZATION FIX ===\n")
    
    # Mock normalization values for percentile strategy
    # These would come from the original AGBD preprocessing
    mock_norm_values = {
        'p1': 0.01,    # 1st percentile
        'p99': 0.85,   # 99th percentile
    }
    
    # Test surface reflectance values (typical range from your logs)
    test_surface_reflectance = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    
    print("Testing percentile normalization ('pct'):")
    print(f"Input surface reflectance: {test_surface_reflectance}")
    
    # Apply percentile normalization
    normalized_pct = normalize_data(test_surface_reflectance, mock_norm_values, 'pct')
    print(f"After 'pct' normalization: {normalized_pct}")
    print(f"Range: [{normalized_pct.min():.3f}, {normalized_pct.max():.3f}]")
    
    # Compare with mean_std (old buggy way)
    mock_meanstd_values = {
        'mean': 0.3,
        'std': 0.1,
    }
    
    print(f"\nComparing with mean_std normalization (old buggy way):")
    normalized_meanstd = normalize_data(test_surface_reflectance, mock_meanstd_values, 'mean_std')
    print(f"After 'mean_std' normalization: {normalized_meanstd}")
    print(f"Range: [{normalized_meanstd.min():.3f}, {normalized_meanstd.max():.3f}]")
    
    print(f"\n🎯 KEY INSIGHT:")
    print(f"- Percentile normalization: {normalized_pct.min():.3f} to {normalized_pct.max():.3f}")
    print(f"- Mean_std normalization: {normalized_meanstd.min():.3f} to {normalized_meanstd.max():.3f}")
    print(f"- The ranges are COMPLETELY different!")
    print(f"- This explains why models trained with 'pct' fail with 'mean_std' inputs")
    
    # Test biomass values
    print(f"\n=== TESTING BIOMASS TARGET NORMALIZATION ===")
    
    # Typical biomass values from your test logs
    test_biomass = np.array([82.14, 189.34, 213.36, 291.18, 494.39])
    
    print(f"Input biomass values (Mg/ha): {test_biomass}")
    
    # Mock biomass normalization values
    biomass_norm_values = {
        'p1': 10.0,    # 1st percentile biomass
        'p99': 450.0,  # 99th percentile biomass
    }
    
    normalized_biomass = normalize_data(test_biomass, biomass_norm_values, 'pct')
    print(f"After 'pct' normalization: {normalized_biomass}")
    print(f"Range: [{normalized_biomass.min():.3f}, {normalized_biomass.max():.3f}]")
    
    print(f"\n✅ EXPECTED IMPROVEMENT:")
    print(f"- Models should now see input ranges similar to original AGBD training")
    print(f"- Predictions should be in reasonable biomass range (not ~0 Mg/ha)")
    print(f"- This should dramatically improve model performance")

if __name__ == "__main__":
    test_normalization_fix()
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Run your test script again with the fixed normalization")
    print(f"2. Model predictions should now be > 0 Mg/ha")
    print(f"3. Check that predictions are in reasonable range (10-400 Mg/ha)")
    print(f"4. Verify improvement in RMSE/MAE metrics")
    print(f"5. Test with multiple models to confirm fix works across architectures")
