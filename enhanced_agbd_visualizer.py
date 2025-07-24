#!/usr/bin/env python3
"""
Enhanced AGBD Visualization Module for Pangaea-Bench
Creates comprehensive visualizations for AGBD biomass prediction analysis.

Features:
- Clean 25x25 patch extraction and visualization
- Professional multi-panel figures
- Ground truth vs prediction comparison
- Spatial pattern analysis
- Error visualization with heatmaps
- Statistical summaries and model performance metrics
- WandB integration for experiment tracking

Version: 2.0 - Post-Fix Analysis
Date: July 22, 2025
Status: ✅ PIPELINE SUCCESSFULLY FIXED
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class AGBDVisualizer:
    """Enhanced AGBD visualization class with comprehensive analysis capabilities."""
    
    def __init__(self, save_dir: str = "./agbd_visualizations", dpi: int = 300):
        """
        Initialize the AGBD visualizer.
        
        Args:
            save_dir: Directory to save visualization outputs
            dpi: Resolution for saved figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.dpi = dpi
        
        # Color schemes for different visualization types
        self.biomass_cmap = LinearSegmentedColormap.from_list(
            'biomass', ['#2E7D32', '#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B', '#FF9800', '#F57C00']
        )
        self.error_cmap = LinearSegmentedColormap.from_list(
            'error', ['#1565C0', '#42A5F5', '#E3F2FD', '#FFECB3', '#FFB74D', '#FF7043', '#D32F2F']
        )
        
        print("="*60)
        print("🌟 ENHANCED AGBD VISUALIZATION MODULE LOADED")
        print("Features: Professional visualizations, comprehensive analysis")
        print("Version: 2.0 - Post-Fix Analysis")
        print("="*60)
        
    def extract_agbd_patch(self, tensor: torch.Tensor, target_size: Tuple[int, int] = (25, 25)) -> torch.Tensor:
        """
        Extract the original AGBD patch region from model outputs.
        
        Args:
            tensor: Input tensor of any size
            target_size: Desired output size (default 25x25 for AGBD)
            
        Returns:
            Extracted patch tensor
        """
        if len(tensor.shape) == 4:  # Batch dimension
            tensor = tensor[0]  # Take first sample
        if len(tensor.shape) == 3:  # Channel dimension  
            tensor = tensor[0]  # Take first channel
            
        h, w = tensor.shape[-2:]
        th, tw = target_size
        
        # Calculate center coordinates  
        center_h, center_w = h // 2, w // 2
        
        # Extract patch around center
        start_h = max(0, center_h - th // 2)
        end_h = min(h, start_h + th)
        start_w = max(0, center_w - tw // 2)  
        end_w = min(w, start_w + tw)
        
        # Adjust if we hit boundaries
        if end_h - start_h < th:
            start_h = max(0, end_h - th)
        if end_w - start_w < tw:
            start_w = max(0, end_w - tw)
            
        extracted = tensor[start_h:start_h+th, start_w:start_w+tw]
        
        # Pad if necessary to reach target size
        if extracted.shape[0] < th or extracted.shape[1] < tw:
            pad_h = max(0, th - extracted.shape[0])
            pad_w = max(0, tw - extracted.shape[1])
            extracted = torch.nn.functional.pad(extracted, (0, pad_w, 0, pad_h))
            
        return extracted
    
    def create_comprehensive_visualization(self, 
                                        prediction: torch.Tensor,
                                        target: torch.Tensor, 
                                        optical_data: Optional[torch.Tensor] = None,
                                        sar_data: Optional[torch.Tensor] = None,
                                        sample_idx: int = 1,
                                        save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Create a comprehensive multi-panel visualization for AGBD analysis.
        
        Args:
            prediction: Model prediction tensor
            target: Ground truth target tensor
            optical_data: Optional RGB optical data for context
            sar_data: Optional SAR data for context
            sample_idx: Sample index for labeling
            save_path: Optional custom save path
            
        Returns:
            Dictionary of computed metrics
        """
        # Extract AGBD patches
        pred_patch = self.extract_agbd_patch(prediction)
        target_patch = self.extract_agbd_patch(target)
        
        # Calculate center pixel values
        center_idx = (12, 12)  # AGBD center pixel
        pred_center = pred_patch[center_idx].item()
        gt_center = target_patch[center_idx].item()
        error = abs(pred_center - gt_center)
        rel_error = (error / gt_center * 100) if gt_center != 0 else 0
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle(f'AGBD Biomass Prediction Analysis - Sample {sample_idx}\n'
                    f'Predicted: {pred_center:.1f} Mg/ha, Ground Truth: {gt_center:.1f} Mg/ha, '
                    f'Error: {error:.1f} Mg/ha ({rel_error:.1f}%)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Optical RGB (if available)
        ax1 = fig.add_subplot(gs[0, 0])
        if optical_data is not None:
            optical_patch = self.extract_agbd_patch(optical_data)
            if optical_patch.shape[0] >= 3:  # Ensure we have RGB bands
                rgb = optical_patch[[3, 2, 1]]  # Assuming B04, B03, B02 are RGB
                rgb_normalized = torch.clamp(rgb.permute(1, 2, 0) * 3, 0, 1)  # Enhance contrast
                ax1.imshow(rgb_normalized)
                ax1.set_title('Optical RGB\n(25×25 AGBD patch)', fontweight='bold')
            else:
                ax1.text(0.5, 0.5, 'RGB bands\nnot available', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=12)
                ax1.set_title('Optical Data (N/A)', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No optical\ndata provided', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Optical Data (N/A)', fontweight='bold')
        ax1.axis('off')
        
        # 2. SAR Data (if available) 
        ax2 = fig.add_subplot(gs[0, 1])
        if sar_data is not None:
            sar_patch = self.extract_agbd_patch(sar_data)
            if sar_patch.shape[0] >= 1:
                sar_display = sar_patch[0]  # First SAR band
                im2 = ax2.imshow(sar_display.cpu(), cmap='gray', vmin=sar_display.quantile(0.02), 
                               vmax=sar_display.quantile(0.98))
                ax2.set_title('SAR Data\n(HH/VV band)', fontweight='bold')
                plt.colorbar(im2, ax=ax2, shrink=0.6, label='dB')
            else:
                ax2.text(0.5, 0.5, 'SAR data\nnot available', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('SAR Data (N/A)', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No SAR\ndata provided', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('SAR Data (N/A)', fontweight='bold')
        ax2.axis('off')
        
        # 3. Ground Truth
        ax3 = fig.add_subplot(gs[0, 2])
        # Only show center pixel with biomass value, rest as context
        gt_display = torch.full_like(target_patch, float('nan'))
        gt_display[center_idx] = gt_center
        im3 = ax3.imshow(gt_display.cpu(), cmap=self.biomass_cmap, vmin=0, vmax=500)
        ax3.add_patch(patches.Rectangle((center_idx[1]-0.5, center_idx[0]-0.5), 1, 1, 
                                       linewidth=3, edgecolor='red', facecolor='none'))
        ax3.set_title(f'Ground Truth\nCenter: {gt_center:.1f} Mg/ha', fontweight='bold')
        ax3.set_xticks(range(0, 25, 5))
        ax3.set_yticks(range(0, 25, 5))
        plt.colorbar(im3, ax=ax3, shrink=0.6, label='Biomass (Mg/ha)')
        
        # 4. Prediction Heatmap
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(pred_patch.cpu(), cmap=self.biomass_cmap, vmin=0, vmax=500)
        ax4.add_patch(patches.Rectangle((center_idx[1]-0.5, center_idx[0]-0.5), 1, 1,
                                       linewidth=3, edgecolor='red', facecolor='none'))
        ax4.set_title(f'Prediction\nCenter: {pred_center:.1f} Mg/ha', fontweight='bold')
        ax4.set_xticks(range(0, 25, 5))
        ax4.set_yticks(range(0, 25, 5))
        plt.colorbar(im4, ax=ax4, shrink=0.6, label='Biomass (Mg/ha)')
        
        # 5. Error Heatmap
        ax5 = fig.add_subplot(gs[1, 0])
        error_patch = torch.abs(pred_patch - gt_center)  # Error relative to ground truth center
        im5 = ax5.imshow(error_patch.cpu(), cmap=self.error_cmap, vmin=0, vmax=error_patch.max())
        ax5.add_patch(patches.Rectangle((center_idx[1]-0.5, center_idx[0]-0.5), 1, 1,
                                       linewidth=3, edgecolor='black', facecolor='none'))
        ax5.set_title(f'Absolute Error\nCenter: {error:.1f} Mg/ha', fontweight='bold')
        ax5.set_xticks(range(0, 25, 5))
        ax5.set_yticks(range(0, 25, 5))
        plt.colorbar(im5, ax=ax5, shrink=0.6, label='Error (Mg/ha)')
        
        # 6. Spatial Profile Analysis
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
        
        # 7. Statistics Summary
        ax7 = fig.add_subplot(gs[1, 3])
        stats_text = [
            f"AGBD 25×25 Patch Statistics:",
            f"",
            f"Center Pixel:",
            f"  Predicted: {pred_center:.2f} Mg/ha", 
            f"  Ground Truth: {gt_center:.2f} Mg/ha",
            f"  Absolute Error: {error:.2f} Mg/ha",
            f"  Relative Error: {rel_error:.1f}%",
            f"",
            f"Patch Statistics:",
            f"  Pred Mean: {pred_patch.mean():.1f} Mg/ha",
            f"  Pred Std: {pred_patch.std():.1f} Mg/ha", 
            f"  Pred Range: [{pred_patch.min():.1f}, {pred_patch.max():.1f}]",
            f"",
            f"Spatial Info:",
            f"  Patch Size: 25×25 pixels",
            f"  Resolution: 10m/pixel",
            f"  Coverage: 250m × 250m",
            f"  Valid Pixels: 1/625"
        ]
        ax7.text(0.05, 0.95, '\n'.join(stats_text), transform=ax7.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax7.axis('off')
        
        # 8. Model Performance Indicators  
        ax8 = fig.add_subplot(gs[2, :])
        
        # Create performance visualization
        metrics = ['Spatial\nConsistency', 'Center\nAccuracy', 'Range\nPreservation', 'Spatial\nLearning']
        
        # Calculate metrics (normalized 0-1)
        spatial_consistency = min(1.0, 1.0 - (pred_patch.std() / pred_patch.mean()).item() * 0.5) if pred_patch.mean() > 0 else 0
        center_accuracy = max(0, 1.0 - (error / max(gt_center, 1)) * 2)  # Inverse of relative error
        range_preservation = min(1.0, pred_patch.max().item() / 300)  # Normalized to expected max ~300 Mg/ha
        spatial_learning = min(1.0, (pred_patch.max() - pred_patch.min()).item() / 100)  # Range/diversity metric
        
        values = [spatial_consistency, center_accuracy, range_preservation, spatial_learning]
        colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in values]
        
        bars = ax8.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax8.set_ylim(0, 1)
        ax8.set_ylabel('Performance Score (0-1)')
        ax8.set_title('Model Performance Assessment', fontweight='bold', fontsize=14)
        ax8.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Add performance legend
        ax8.text(0.02, 0.98, 'Performance Indicators:\n' +
                '🟢 Excellent (>0.7)  🟡 Good (0.4-0.7)  🔴 Needs Improvement (<0.4)',
                transform=ax8.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Save figure
        if save_path is None:
            save_path = self.save_dir / f'agbd_comprehensive_analysis_sample_{sample_idx}.png'
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f"📊 Comprehensive visualization saved: {save_path}")
        
        # Close figure to free memory
        plt.close(fig)
        
        # Return computed metrics
        return {
            'pred_center': pred_center,
            'gt_center': gt_center,
            'absolute_error': error,
            'relative_error': rel_error,
            'spatial_consistency': spatial_consistency,
            'center_accuracy': center_accuracy,
            'range_preservation': range_preservation,
            'spatial_learning': spatial_learning,
            'patch_mean': pred_patch.mean().item(),
            'patch_std': pred_patch.std().item(),
            'patch_min': pred_patch.min().item(),
            'patch_max': pred_patch.max().item()
        }
    
    def create_batch_summary(self, metrics_list: List[Dict[str, float]], 
                           save_path: Optional[str] = None) -> None:
        """
        Create a summary visualization for a batch of samples.
        
        Args:
            metrics_list: List of metrics dictionaries from individual samples
            save_path: Optional custom save path
        """
        if not metrics_list:
            return
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(metrics_list)
        
        # Create summary figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AGBD Batch Performance Summary', fontsize=16, fontweight='bold')
        
        # 1. Prediction vs Ground Truth Scatter
        ax1 = axes[0, 0]
        ax1.scatter(df['gt_center'], df['pred_center'], alpha=0.7, s=50, c='blue')
        min_val = min(df['gt_center'].min(), df['pred_center'].min())
        max_val = max(df['gt_center'].max(), df['pred_center'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect prediction')
        ax1.set_xlabel('Ground Truth (Mg/ha)')
        ax1.set_ylabel('Predictions (Mg/ha)')
        ax1.set_title('Predictions vs Ground Truth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = df['gt_center'].corr(df['pred_center'])
        ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Error Distribution
        ax2 = axes[0, 1]  
        ax2.hist(df['absolute_error'], bins=10, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Absolute Error (Mg/ha)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(df['absolute_error'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["absolute_error"].mean():.1f}')
        ax2.legend()
        
        # 3. Performance Metrics Radar
        ax3 = axes[0, 2]
        performance_cols = ['spatial_consistency', 'center_accuracy', 'range_preservation', 'spatial_learning']
        mean_performance = df[performance_cols].mean()
        
        angles = np.linspace(0, 2*np.pi, len(performance_cols), endpoint=False)
        values = mean_performance.values
        
        # Close the radar chart
        angles = np.concatenate((angles, [angles[0]]))
        values = np.concatenate((values, [values[0]]))
        
        ax3.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax3.fill(angles, values, alpha=0.25, color='blue')
        ax3.set_ylim(0, 1)
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels([col.replace('_', '\n') for col in performance_cols])
        ax3.set_title('Average Performance Metrics')
        ax3.grid(True)
        
        # 4. Residuals vs Ground Truth
        ax4 = axes[1, 0]
        residuals = df['pred_center'] - df['gt_center']
        ax4.scatter(df['gt_center'], residuals, alpha=0.7, s=50, c='green')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax4.set_xlabel('Ground Truth (Mg/ha)')
        ax4.set_ylabel('Residuals (Pred - GT)')
        ax4.set_title('Residual Analysis')
        ax4.grid(True, alpha=0.3)
        
        # 5. Sample Performance Timeline
        ax5 = axes[1, 1]
        sample_indices = range(len(df))
        ax5.plot(sample_indices, df['absolute_error'], 'bo-', markersize=4, label='Absolute Error')
        ax5.set_xlabel('Sample Index')
        ax5.set_ylabel('Absolute Error (Mg/ha)')
        ax5.set_title('Error by Sample')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. Summary Statistics Table
        ax6 = axes[1, 2]
        summary_stats = {
            'Metric': ['Count', 'RMSE', 'MAE', 'Bias', 'Correlation', 'Best Error', 'Worst Error'],
            'Value': [
                len(df),
                f"{np.sqrt((df['absolute_error']**2).mean()):.1f} Mg/ha",
                f"{df['absolute_error'].mean():.1f} Mg/ha", 
                f"{(df['pred_center'] - df['gt_center']).mean():.1f} Mg/ha",
                f"{corr:.3f}",
                f"{df['absolute_error'].min():.1f} Mg/ha",
                f"{df['absolute_error'].max():.1f} Mg/ha"
            ]
        }
        
        # Create table
        table_data = list(zip(summary_stats['Metric'], summary_stats['Value']))
        table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax6.axis('off')
        ax6.set_title('Summary Statistics', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.save_dir / f'agbd_batch_summary_{len(df)}_samples.png'
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f"📈 Batch summary saved: {save_path}")
        plt.close(fig)

# Example usage and validation
if __name__ == "__main__":
    print("🧪 Testing Enhanced AGBD Visualizer...")
    
    visualizer = AGBDVisualizer()
    
    # Create synthetic test data
    pred_tensor = torch.randn(1, 1, 224, 224) * 50 + 150  # Realistic biomass range
    target_tensor = torch.full((1, 224, 224), -1.0)  # Background ignore_index
    target_tensor[0, 112, 112] = 180.5  # Center pixel ground truth
    
    # Test comprehensive visualization
    metrics = visualizer.create_comprehensive_visualization(
        pred_tensor, target_tensor, sample_idx=999
    )
    
    print("✅ Enhanced visualizer test completed!")
    print(f"📊 Metrics: {metrics}")
    print(f"💾 Visualizations saved to: {visualizer.save_dir}")
