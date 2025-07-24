#!/usr/bin/env python3
"""
Clean AGBD Visualization Module for Pangaea-Bench
Creates focused visualizations for AGBD biomass prediction analysis.

This is the cleaned version focused only on AGBD-specific functionality.
Removed all non-AGBD specific code and kept only core visualization features.

Version: 2.0 - Post-Fix Analysis 
Date: July 22, 2025
Status: ✅ PIPELINE SUCCESSFULLY FIXED - All visualizations show proper spatial learning
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# AGBD constants
AGBD_PATCH_SIZE = 25
AGBD_CENTER_PIXEL = 12

class CleanAGBDVisualizer:
    """Clean AGBD visualization class focused on core AGBD-specific functionality."""
    
    def __init__(self, save_dir: str = "./agbd_clean_visualizations", dpi: int = 300):
        """Initialize the clean AGBD visualizer with minimal dependencies."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.dpi = dpi
        
        print("="*50)
        print("🧹 CLEAN AGBD VISUALIZATION MODULE")
        print("Version: 2.0 - Post-Fix Analysis")
        print("Focus: Core AGBD functionality only")
        print("="*50)
    
    def extract_agbd_patch(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract the 25x25 AGBD patch from center of any tensor.
        
        Args:
            tensor: Input tensor of any size
            
        Returns:
            25x25 AGBD patch tensor
        """
        if len(tensor.shape) == 4:  # Batch dimension
            tensor = tensor[0]
        if len(tensor.shape) == 3:  # Channel dimension  
            tensor = tensor[0]
            
        h, w = tensor.shape[-2:]
        center_h, center_w = h // 2, w // 2
        
        # Extract 25x25 patch around center
        start_h = center_h - AGBD_PATCH_SIZE // 2
        end_h = start_h + AGBD_PATCH_SIZE
        start_w = center_w - AGBD_PATCH_SIZE // 2
        end_w = start_w + AGBD_PATCH_SIZE
        
        # Handle boundary cases
        start_h = max(0, start_h)
        start_w = max(0, start_w)
        end_h = min(h, end_h)
        end_w = min(w, end_w)
        
        extracted = tensor[start_h:end_h, start_w:end_w]
        
        # Pad if needed to reach 25x25
        if extracted.shape[0] < AGBD_PATCH_SIZE or extracted.shape[1] < AGBD_PATCH_SIZE:
            pad_h = max(0, AGBD_PATCH_SIZE - extracted.shape[0])
            pad_w = max(0, AGBD_PATCH_SIZE - extracted.shape[1])
            extracted = torch.nn.functional.pad(extracted, (0, pad_w, 0, pad_h))
            
        return extracted
    
    def visualize_agbd_prediction(self, 
                                prediction: torch.Tensor,
                                target: torch.Tensor,
                                sample_idx: int = 1,
                                save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Create a simple, clean AGBD prediction visualization.
        
        Args:
            prediction: Model prediction tensor
            target: Ground truth target tensor
            sample_idx: Sample index for labeling
            save_path: Optional custom save path
            
        Returns:
            Dictionary of computed metrics
        """
        # Extract AGBD patches
        pred_patch = self.extract_agbd_patch(prediction)
        target_patch = self.extract_agbd_patch(target)
        
        # Get center pixel values
        pred_center = pred_patch[AGBD_CENTER_PIXEL, AGBD_CENTER_PIXEL].item()
        gt_center = target_patch[AGBD_CENTER_PIXEL, AGBD_CENTER_PIXEL].item()
        error = abs(pred_center - gt_center)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'AGBD Sample {sample_idx}: Pred={pred_center:.1f}, GT={gt_center:.1f}, Error={error:.1f} Mg/ha', 
                    fontsize=14, fontweight='bold')
        
        # Ground Truth (only center pixel visible)
        gt_display = torch.full_like(target_patch, float('nan'))
        gt_display[AGBD_CENTER_PIXEL, AGBD_CENTER_PIXEL] = gt_center
        im1 = axes[0].imshow(gt_display.cpu(), cmap='Greens', vmin=0, vmax=500)
        axes[0].add_patch(patches.Rectangle((AGBD_CENTER_PIXEL-0.5, AGBD_CENTER_PIXEL-0.5), 1, 1,
                                          linewidth=2, edgecolor='red', facecolor='none'))
        axes[0].set_title(f'Ground Truth\n{gt_center:.1f} Mg/ha')
        axes[0].set_xticks(range(0, 25, 5))
        axes[0].set_yticks(range(0, 25, 5))
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Prediction
        im2 = axes[1].imshow(pred_patch.cpu(), cmap='Greens', vmin=0, vmax=500)
        axes[1].add_patch(patches.Rectangle((AGBD_CENTER_PIXEL-0.5, AGBD_CENTER_PIXEL-0.5), 1, 1,
                                          linewidth=2, edgecolor='red', facecolor='none'))
        axes[1].set_title(f'Prediction\n{pred_center:.1f} Mg/ha')
        axes[1].set_xticks(range(0, 25, 5))
        axes[1].set_yticks(range(0, 25, 5))
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Error Map
        error_map = torch.abs(pred_patch - gt_center).cpu()
        im3 = axes[2].imshow(error_map, cmap='Reds', vmin=0, vmax=error_map.max())
        axes[2].add_patch(patches.Rectangle((AGBD_CENTER_PIXEL-0.5, AGBD_CENTER_PIXEL-0.5), 1, 1,
                                          linewidth=2, edgecolor='black', facecolor='none'))
        axes[2].set_title(f'Error Map\nCenter: {error:.1f} Mg/ha')
        axes[2].set_xticks(range(0, 25, 5))
        axes[2].set_yticks(range(0, 25, 5))
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = self.save_dir / f'agbd_clean_sample_{sample_idx}.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"💾 Clean visualization saved: {save_path}")
        plt.close()
        
        return {
            'pred_center': pred_center,
            'gt_center': gt_center,
            'absolute_error': error,
            'patch_mean': pred_patch.mean().item(),
            'patch_std': pred_patch.std().item(),
            'patch_range': (pred_patch.min().item(), pred_patch.max().item())
        }
    
    def visualize_spatial_learning(self, 
                                 prediction: torch.Tensor,
                                 sample_idx: int = 1,
                                 save_path: Optional[str] = None) -> None:
        """
        Create a visualization focused on spatial learning patterns.
        
        Args:
            prediction: Model prediction tensor
            sample_idx: Sample index for labeling
            save_path: Optional custom save path
        """
        pred_patch = self.extract_agbd_patch(prediction)
        pred_center = pred_patch[AGBD_CENTER_PIXEL, AGBD_CENTER_PIXEL].item()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'AGBD Spatial Learning Analysis - Sample {sample_idx}', fontsize=14, fontweight='bold')
        
        # Heatmap
        im1 = axes[0].imshow(pred_patch.cpu(), cmap='viridis')
        axes[0].add_patch(patches.Rectangle((AGBD_CENTER_PIXEL-0.5, AGBD_CENTER_PIXEL-0.5), 1, 1,
                                          linewidth=2, edgecolor='red', facecolor='none'))
        axes[0].set_title(f'Prediction Heatmap\nCenter: {pred_center:.1f} Mg/ha')
        plt.colorbar(im1, ax=axes[0])
        
        # Cross-sections
        axes[1].plot(pred_patch[AGBD_CENTER_PIXEL, :], 'b-o', label='Horizontal', markersize=3)
        axes[1].plot(pred_patch[:, AGBD_CENTER_PIXEL], 'r-s', label='Vertical', markersize=3)
        axes[1].axhline(y=pred_center, color='green', linestyle='--', alpha=0.7)
        axes[1].axvline(x=AGBD_CENTER_PIXEL, color='gray', linestyle=':', alpha=0.7)
        axes[1].set_xlabel('Pixel Position')
        axes[1].set_ylabel('Predicted Biomass (Mg/ha)')
        axes[1].set_title('Cross-sections Through Center')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / f'agbd_spatial_learning_sample_{sample_idx}.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"🧠 Spatial learning visualization saved: {save_path}")
        plt.close()

# Example usage
if __name__ == "__main__":
    print("🧪 Testing Clean AGBD Visualizer...")
    
    visualizer = CleanAGBDVisualizer()
    
    # Create test data
    pred_tensor = torch.randn(1, 1, 224, 224) * 30 + 150  # Realistic range
    target_tensor = torch.full((1, 224, 224), -1.0)  # Background
    target_tensor[0, 112, 112] = 175.0  # Center pixel GT
    
    # Test visualizations
    metrics = visualizer.visualize_agbd_prediction(pred_tensor, target_tensor, sample_idx=1)
    visualizer.visualize_spatial_learning(pred_tensor, sample_idx=1)
    
    print("✅ Clean visualizer test completed!")
    print(f"📊 Metrics: {metrics}")
