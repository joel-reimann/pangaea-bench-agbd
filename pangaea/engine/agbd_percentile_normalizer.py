"""
AGBD-specific percentile normalization to match original AGBD paper implementation.

Original AGBD uses percentile normalization (1st and 99th percentiles) while 
Pangaea-Bench uses min/max normalization. This causes significant scale differences
that prevent the model from learning properly.

Based on:
- Original AGBD implementation: norm_strat = 'pct' 
- Statistics from: /scratch/reimannj/pangaea_agbd_integration_final/data/agbd/statistics_subset_2019-2020-v4_new.pkl
"""

import torch
import numpy as np
import pickle
from typing import Dict, Any


class AGBDPercentileNormalizer:
    """
    Percentile-based normalization for AGBD dataset to match original implementation.
    Uses 1st and 99th percentiles instead of min/max, with clipping to [0, 1].
    """
    
    def __init__(self, stats_path: str):
        """Load percentile statistics from AGBD statistics file."""
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
            
        # Extract percentile ranges for each modality
        self.s2_percentiles = {}
        for band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']:
            band_stats = self.stats['S2_bands'][band]
            self.s2_percentiles[band] = {
                'p1': band_stats['p1'],
                'p99': band_stats['p99']
            }
            
        # SAR bands are under ALOS_bands but mapped to HH/HV
        self.sar_percentiles = {}
        for band in ['HH', 'HV']:
            band_stats = self.stats['ALOS_bands'][band]
            self.sar_percentiles[band] = {
                'p1': band_stats['p1'],
                'p99': band_stats['p99']
            }
    
    def normalize_optical(self, optical_data: torch.Tensor) -> torch.Tensor:
        """
        Apply percentile normalization to optical bands.
        
        Args:
            optical_data: Tensor of shape [B, C, H, W] or [C, H, W]
            
        Returns:
            Normalized tensor with values clipped to [0, 1]
        """
        # Band order from AGBD config
        band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        
        normalized = optical_data.clone()
        
        for i, band_name in enumerate(band_names):
            if i >= optical_data.shape[-3]:  # Check if band exists
                break
                
            # Convert to statistics naming convention (B1 -> B01, etc.)
            stats_band_name = f"B{band_name[1:].zfill(2)}" if band_name.startswith('B') else band_name
            
            if stats_band_name in self.s2_percentiles:
                p1 = self.s2_percentiles[stats_band_name]['p1']
                p99 = self.s2_percentiles[stats_band_name]['p99']
                
                # Apply percentile normalization: (x - p1) / (p99 - p1)
                if len(normalized.shape) == 4:  # Batch dimension
                    normalized[:, i] = (normalized[:, i] - p1) / (p99 - p1)
                else:  # No batch dimension
                    normalized[i] = (normalized[i] - p1) / (p99 - p1)
        
        # Clip to [0, 1] as in original AGBD
        return torch.clamp(normalized, 0, 1)
    
    def normalize_sar(self, sar_data: torch.Tensor) -> torch.Tensor:
        """
        Apply percentile normalization to SAR bands.
        
        Args:
            sar_data: Tensor of shape [B, C, H, W] or [C, H, W] with HH, HV bands
            
        Returns:
            Normalized tensor with values clipped to [0, 1]
        """
        band_names = ['HH', 'HV']
        
        normalized = sar_data.clone()
        
        for i, band_name in enumerate(band_names):
            if i >= sar_data.shape[-3]:  # Check if band exists
                break
                
            if band_name in self.sar_percentiles:
                p1 = self.sar_percentiles[band_name]['p1']
                p99 = self.sar_percentiles[band_name]['p99']
                
                # Apply percentile normalization: (x - p1) / (p99 - p1)
                if len(normalized.shape) == 4:  # Batch dimension
                    normalized[:, i] = (normalized[:, i] - p1) / (p99 - p1)
                else:  # No batch dimension
                    normalized[i] = (normalized[i] - p1) / (p99 - p1)
        
        # Clip to [0, 1] as in original AGBD
        return torch.clamp(normalized, 0, 1)


def apply_agbd_percentile_normalization(data: Dict[str, torch.Tensor], stats_path: str) -> Dict[str, torch.Tensor]:
    """
    Apply AGBD percentile normalization to multi-modal data.
    
    Args:
        data: Dictionary with 'optical' and/or 'sar' keys containing tensor data
        stats_path: Path to AGBD statistics pickle file
        
    Returns:
        Dictionary with normalized data
    """
    normalizer = AGBDPercentileNormalizer(stats_path)
    normalized_data = {}
    
    if 'optical' in data:
        normalized_data['optical'] = normalizer.normalize_optical(data['optical'])
    
    if 'sar' in data:
        normalized_data['sar'] = normalizer.normalize_sar(data['sar'])
        
    # Copy any other modalities unchanged
    for key, value in data.items():
        if key not in ['optical', 'sar']:
            normalized_data[key] = value
    
    return normalized_data
