#!/usr/bin/env python3
"""
Quick test to verify AGBDPercentileNormalize works correctly.
Tests that the percentile normalizer loads properly and processes data.
"""

import sys
sys.path.append('/scratch/final2/pangaea-bench-agbd')

import torch
from pangaea.engine.data_preprocessor import AGBDPercentileNormalize

def test_agbd_percentile_normalizer():
    """Test that our AGBD percentile normalizer works."""
    print("Testing AGBDPercentileNormalize...")
    
    try:
        # Test initialization
        normalizer = AGBDPercentileNormalize()
        print("✅ Normalizer initialized successfully")
        
        # Test update_meta method
        dummy_meta = {"test": "data"}
        result_meta = normalizer.update_meta(dummy_meta)
        print("✅ update_meta method works")
        
        # Test with dummy data structure (mimicking Pangaea data structure)
        class DummyData:
            def __init__(self):
                # Create dummy optical and SAR data
                self.optical = torch.randn(12, 1, 25, 25) * 0.5 + 0.3  # Realistic S2 range
                self.sar = torch.randn(2, 1, 25, 25) * 10 - 15        # Realistic SAR range
        
        dummy_data = DummyData()
        
        print(f"Input optical range: [{dummy_data.optical.min():.3f}, {dummy_data.optical.max():.3f}]")
        print(f"Input SAR range: [{dummy_data.sar.min():.3f}, {dummy_data.sar.max():.3f}]")
        
        # Apply normalization
        result = normalizer(dummy_data)
        
        print(f"Output optical range: [{result.optical.min():.3f}, {result.optical.max():.3f}]")
        print(f"Output SAR range: [{result.sar.min():.3f}, {result.sar.max():.3f}]")
        
        # Check that output is in [0, 1] range (clipped)
        assert result.optical.min() >= 0, "Optical data should be >= 0"
        assert result.optical.max() <= 1, "Optical data should be <= 1"
        assert result.sar.min() >= 0, "SAR data should be >= 0"
        assert result.sar.max() <= 1, "SAR data should be <= 1"
        
        print("✅ All tests passed! Percentile normalization works correctly")
        print("✅ Ready to test with full pipeline")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_agbd_percentile_normalizer()
    exit(0 if success else 1)
