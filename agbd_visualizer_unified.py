"""
AGBD Unified Visualizer - Final Working Version

Combines the best features from all previous visualizers:
- Proper tensor device handling (.cpu() calls)
- Correct 25x25 AGBD patch extraction
- Multi-modal support (optical RGB + SAR)
- Professional multi-panel layout
- Comprehensive metrics and statistics
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional, Tuple, Any
import os
import warnings

# AGBD Constants
AGBD_PATCH_SIZE = 25
AGBD_CENTER_PIXEL = 12
AGBD_BIOMASS_RANGE = (0, 500)

class AGBDVisualizerUnified:
    """Unified AGBD visualizer with comprehensive analysis capabilities."""
    
    def __init__(self):
        """Initialize the unified AGBD visualizer."""
        # Create custom biomass colormap
        self.biomass_cmap = LinearSegmentedColormap.from_list(
            'biomass', 
            ['#ffffff', '#90EE90', '#228B22', '#006400'], 
            N=256
        )
        
        print("=" * 60)
        print("🌿 AGBD UNIFIED VISUALIZER LOADED")
        print("Features: Multi-modal analysis, spatial learning, comprehensive metrics")
        print("Version: 3.0 - Unified Final Implementation")
        print("=" * 60)
    
    def extract_agbd_patch(self, tensor: torch.Tensor, target_size: Tuple[int, int] = (25, 25)) -> torch.Tensor:
        """
        Extract the 25x25 AGBD patch from encoder output.
        
        Args:
            tensor: Input tensor of shape (H, W) or (C, H, W) or (1, C, H, W)
            target_size: Target patch size (default: 25x25)
            
        Returns:
            Extracted AGBD patch
        """
        # Handle different tensor shapes
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Remove batch dimension
        if len(tensor.shape) == 3:
            if tensor.shape[0] == 1:
                tensor = tensor[0]  # Remove single channel dimension
            
        # Get tensor dimensions
        if len(tensor.shape) == 3:
            # Multi-channel (C, H, W)
            C, H, W = tensor.shape
            center_h, center_w = H // 2, W // 2
        else:
            # Single channel (H, W)
            H, W = tensor.shape
            center_h, center_w = H // 2, W // 2
            
        # Extract centered patch
        th, tw = target_size
        half_h, half_w = th // 2, tw // 2
        
        start_h = max(0, center_h - half_h)
        end_h = min(H, start_h + th)
        start_w = max(0, center_w - half_w)
        end_w = min(W, start_w + tw)
        
        # Adjust if we hit boundaries
        if end_h - start_h < th:
            start_h = max(0, end_h - th)
        if end_w - start_w < tw:
            start_w = max(0, end_w - tw)
            
        if len(tensor.shape) == 3:
            extracted = tensor[:, start_h:start_h+th, start_w:start_w+tw]
        else:
            extracted = tensor[start_h:start_h+th, start_w:start_w+tw]
        
        # Pad if necessary
        if len(tensor.shape) == 3:
            C_out, H_out, W_out = extracted.shape
            if H_out < th or W_out < tw:
                pad_h = max(0, th - H_out)
                pad_w = max(0, tw - W_out)
                extracted = F.pad(extracted, (0, pad_w, 0, pad_h))
        else:
            H_out, W_out = extracted.shape
            if H_out < th or W_out < tw:
                pad_h = max(0, th - H_out)
                pad_w = max(0, tw - W_out)
                extracted = F.pad(extracted, (0, pad_w, 0, pad_h))
                
        return extracted
    
    def create_rgb_visualization(self, optical_data: torch.Tensor) -> np.ndarray:
        """
        Create RGB visualization from optical data.
        
        Args:
            optical_data: Optical tensor of shape (C, H, W) or (C, T, H, W)
            
        Returns:
            RGB array (H, W, 3)
        """
        # Handle temporal dimension
        if len(optical_data.shape) == 4:
            optical_data = optical_data[:, 0]  # Take first time step
            
        # Move to CPU and convert to numpy
        optical_np = optical_data.cpu().numpy()
        
        if optical_np.shape[0] < 3:
            # Fallback: create grayscale RGB
            if optical_np.shape[0] > 0:
                gray = optical_np[0]
            else:
                gray = np.zeros((optical_np.shape[1], optical_np.shape[2]))
            return np.stack([gray, gray, gray], axis=2)
        
        # For Sentinel-2, use bands [3, 2, 1] (B04, B03, B02) for RGB
        # This corresponds to Red, Green, Blue bands
        if optical_np.shape[0] >= 4:
            # Use B04 (Red), B03 (Green), B02 (Blue)
            rgb_bands = optical_np[[3, 2, 1]]  # 0-indexed: [B04, B03, B02]
        else:
            # Use first 3 bands
            rgb_bands = optical_np[:3]
        
        # Transpose to (H, W, C)
        rgb = np.transpose(rgb_bands, (1, 2, 0))
        
        # Robust normalization with percentile clipping
        for i in range(3):
            band = rgb[:, :, i]
            if band.max() > band.min():
                p2, p98 = np.percentile(band, [2, 98])
                band_norm = np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                band_norm = np.zeros_like(band)
            rgb[:, :, i] = band_norm
        
        # Enhance contrast
        rgb = np.clip(rgb * 1.2, 0, 1)
        
        return rgb
    
    def create_sar_visualization(self, sar_data: torch.Tensor) -> np.ndarray:
        """
        Create SAR visualization.
        
        Args:
            sar_data: SAR tensor of shape (C, H, W) or (C, T, H, W)
            
        Returns:
            Grayscale array (H, W)
        """
        # Handle temporal dimension
        if len(sar_data.shape) == 4:
            sar_data = sar_data[:, 0]  # Take first time step
            
        # Move to CPU and get first band (usually HH or VV)
        sar_np = sar_data[0].cpu().numpy()
        
        # SAR data is often in dB, handle carefully
        if sar_np.min() < 0:  # Likely dB values
            # Normalize dB values
            p1, p99 = np.percentile(sar_np, [1, 99])
            sar_norm = np.clip((sar_np - p1) / (p99 - p1), 0, 1)
        else:  # Linear values
            # Convert to dB for better visualization
            sar_db = 10 * np.log10(np.maximum(sar_np, 1e-10))
            p1, p99 = np.percentile(sar_db, [1, 99])
            sar_norm = np.clip((sar_db - p1) / (p99 - p1), 0, 1)
        
        return sar_norm
    
    def create_comprehensive_visualization(self, 
                                        prediction: torch.Tensor,
                                        target: torch.Tensor, 
                                        optical_data: Optional[torch.Tensor] = None,
                                        sar_data: Optional[torch.Tensor] = None,
                                        sample_idx: int = 1,
                                        save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Create comprehensive AGBD visualization.
        
        Args:
            prediction: Model prediction tensor
            target: Ground truth target tensor
            optical_data: Optional optical data
            sar_data: Optional SAR data
            sample_idx: Sample index for labeling
            save_path: Optional custom save path
            
        Returns:
            Dictionary of computed metrics
        """
        # Extract AGBD patches and move to CPU
        pred_patch = self.extract_agbd_patch(prediction).cpu()
        target_patch = self.extract_agbd_patch(target).cpu()
        
        # Calculate center pixel values
        center_idx = (12, 12)  # AGBD center pixel
        pred_center = pred_patch[center_idx].item()
        gt_center = target_patch[center_idx].item()
        error = abs(pred_center - gt_center)
        rel_error = (error / gt_center * 100) if gt_center != 0 else 0
        
        # Create figure with comprehensive layout
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle(f'AGBD Biomass Prediction Analysis - Sample {sample_idx}\n'
                    f'Predicted: {pred_center:.1f} Mg/ha, Ground Truth: {gt_center:.1f} Mg/ha, '
                    f'Error: {error:.1f} Mg/ha ({rel_error:.1f}%)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Optical RGB
        ax1 = fig.add_subplot(gs[0, 0])
        if optical_data is not None:
            try:
                optical_patch = self.extract_agbd_patch(optical_data)
                rgb = self.create_rgb_visualization(optical_patch)
                ax1.imshow(rgb)
                ax1.set_title('Optical RGB\n(25×25 AGBD patch)', fontweight='bold')
            except Exception as e:
                ax1.text(0.5, 0.5, f'RGB Error:\n{str(e)[:50]}', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=10)
                ax1.set_title('Optical Data (Error)', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No optical\ndata provided', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Optical Data (N/A)', fontweight='bold')
        ax1.axis('off')
        
        # 2. SAR Data
        ax2 = fig.add_subplot(gs[0, 1])
        if sar_data is not None:
            try:
                sar_patch = self.extract_agbd_patch(sar_data)
                sar_vis = self.create_sar_visualization(sar_patch)
                im2 = ax2.imshow(sar_vis, cmap='gray', vmin=0, vmax=1)
                ax2.set_title('SAR Data\n(HH/VV band)', fontweight='bold')
                plt.colorbar(im2, ax=ax2, shrink=0.6, label='Normalized')
            except Exception as e:
                ax2.text(0.5, 0.5, f'SAR Error:\n{str(e)[:50]}', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=10)
                ax2.set_title('SAR Data (Error)', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No SAR\ndata provided', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('SAR Data (N/A)', fontweight='bold')
        ax2.axis('off')
        
        # 3. Ground Truth
        ax3 = fig.add_subplot(gs[0, 2])
        gt_display = torch.full_like(target_patch, float('nan'))
        gt_display[center_idx] = gt_center
        im3 = ax3.imshow(gt_display, cmap=self.biomass_cmap, vmin=0, vmax=500)
        ax3.add_patch(patches.Rectangle((center_idx[1]-0.5, center_idx[0]-0.5), 1, 1, 
                                       linewidth=3, edgecolor='red', facecolor='none'))
        ax3.set_title(f'Ground Truth\nCenter: {gt_center:.1f} Mg/ha', fontweight='bold')
        ax3.set_xticks(range(0, 25, 5))
        ax3.set_yticks(range(0, 25, 5))
        plt.colorbar(im3, ax=ax3, shrink=0.6, label='Biomass (Mg/ha)')
        
        # 4. Prediction Heatmap
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(pred_patch, cmap=self.biomass_cmap, vmin=0, vmax=500)
        ax4.add_patch(patches.Rectangle((center_idx[1]-0.5, center_idx[0]-0.5), 1, 1,
                                       linewidth=3, edgecolor='red', facecolor='none'))
        ax4.set_title(f'Prediction\nCenter: {pred_center:.1f} Mg/ha', fontweight='bold')
        ax4.set_xticks(range(0, 25, 5))
        ax4.set_yticks(range(0, 25, 5))
        plt.colorbar(im4, ax=ax4, shrink=0.6, label='Biomass (Mg/ha)')
        
        # 5. Error Map
        ax5 = fig.add_subplot(gs[1, 0])
        error_map = torch.abs(pred_patch - gt_center)
        im5 = ax5.imshow(error_map, cmap='Reds', vmin=0, vmax=error_map.max())
        ax5.add_patch(patches.Rectangle((center_idx[1]-0.5, center_idx[0]-0.5), 1, 1,
                                       linewidth=3, edgecolor='black', facecolor='none'))
        ax5.set_title(f'Absolute Error\nCenter: {error:.1f} Mg/ha', fontweight='bold')
        ax5.set_xticks(range(0, 25, 5))
        ax5.set_yticks(range(0, 25, 5))
        plt.colorbar(im5, ax=ax5, shrink=0.6, label='Error (Mg/ha)')
        
        # 6. Spatial Profiles
        ax6 = fig.add_subplot(gs[1, 1:3])
        center_row = pred_patch[center_idx[0], :].cpu()
        center_col = pred_patch[:, center_idx[1]].cpu()
        x_positions = np.arange(25)
        
        ax6.plot(x_positions, center_row, 'b-o', label='Horizontal profile', linewidth=2, markersize=4)
        ax6.plot(x_positions, center_col, 'r-s', label='Vertical profile', linewidth=2, markersize=4)
        ax6.axhline(y=gt_center, color='green', linestyle='--', linewidth=2, label=f'Ground truth ({gt_center:.1f})')
        ax6.axvline(x=center_idx[1], color='gray', linestyle=':', alpha=0.7, label='Center pixel')
        ax6.set_xlabel('Pixel Position')
        ax6.set_ylabel('Predicted Biomass (Mg/ha)')
        ax6.set_title('Spatial Profiles Through Center Pixel', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Statistics Panel
        ax7 = fig.add_subplot(gs[1, 3])
        ax7.axis('off')
        
        # Calculate statistics
        patch_mean = pred_patch.mean().item()
        patch_std = pred_patch.std().item()
        patch_min = pred_patch.min().item()
        patch_max = pred_patch.max().item()
        
        stats_text = f"""AGBD 25x25 Patch Statistics:

Center Pixel Metrics:
  * Predicted: {pred_center:.2f} Mg/ha
  * Ground Truth: {gt_center:.2f} Mg/ha
  * Absolute Error: {error:.2f} Mg/ha
  * Relative Error: {rel_error:.1f}%

Patch Statistics:
  * Pred Mean: {patch_mean:.2f} Mg/ha
  * Pred Std: {patch_std:.2f} Mg/ha
  * Pred Range: [{patch_min:.1f}, {patch_max:.1f}]

Spatial Info:
  * Patch Size: 25x25 pixels
  * Resolution: 10m/pixel
  * Coverage: 250m x 250m
  * Valid Pixels: 1/625"""
        
        ax7.text(0.02, 0.98, stats_text, transform=ax7.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 8. Performance Metrics
        ax8 = fig.add_subplot(gs[2, :])
        
        # Calculate performance metrics
        spatial_consistency = min(1.0, 1.0 - (pred_patch.std() / pred_patch.mean()).item() * 0.5) if pred_patch.mean() > 0 else 0
        center_accuracy = max(0, 1.0 - (error / max(gt_center, 1)) * 2)
        range_preservation = min(1.0, pred_patch.max().item() / 300)
        spatial_learning = min(1.0, (pred_patch.max() - pred_patch.min()).item() / 100)
        
        metrics = ['Spatial\nConsistency', 'Center\nAccuracy', 'Range\nPreservation', 'Spatial\nLearning']
        values = [spatial_consistency, center_accuracy, range_preservation, spatial_learning]
        colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in values]
        
        bars = ax8.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax8.set_ylim(0, 1)
        ax8.set_ylabel('Performance Score (0-1)')
        ax8.set_title('Model Performance Assessment', fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Performance interpretation
        avg_performance = np.mean(values)
        if avg_performance > 0.7:
            perf_text = "Excellent (>0.7)"
        elif avg_performance > 0.5:
            perf_text = "Good (0.5-0.7)"
        else:
            perf_text = "Needs Improvement (<0.5)"
        
        ax8.text(0.02, 0.95, f'Overall Performance: {perf_text}', transform=ax8.transAxes,
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Save visualization
        if save_path is None:
            os.makedirs('agbd_visualizations', exist_ok=True)
            save_path = f'agbd_visualizations/agbd_unified_sample_{sample_idx}.png'
        else:
            # Ensure directory exists for provided save_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Unified AGBD visualization saved: {save_path}")
        
        # Return comprehensive metrics
        return {
            'pred_center': pred_center,
            'gt_center': gt_center,
            'absolute_error': error,
            'relative_error': rel_error,
            'spatial_consistency': spatial_consistency,
            'center_accuracy': center_accuracy,
            'range_preservation': range_preservation,
            'spatial_learning': spatial_learning,
            'patch_mean': patch_mean,
            'patch_std': patch_std,
            'patch_min': patch_min,
            'patch_max': patch_max
        }

# Module initialization
print("🌿 AGBD Unified Visualizer Module Loaded - Ready for comprehensive biomass analysis!")
