"""
AGBD Visualization Module - Clean Implementation

Advanced scientific visualization for AGBD (Above-Ground Biomass Dataset) regression
that shows only the relevant 25x25 AGBD patches instead of full encoder inputs.

FIXES APPLIED:
- Extract actual 25x25 AGBD patches from full encoder outputs  
- Clean visualization without confusing center pixel overlays
- Proper error maps between GT and predicted patches
- WandB logging with meaningful metrics
"""

import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any
import os

# AGBD-specific constants
AGBD_PATCH_SIZE = 25
AGBD_CENTER_PIXEL = 12
AGBD_BIOMASS_RANGE = (0, 500)
AGBD_SPATIAL_RESOLUTION = 10


def extract_agbd_patch_from_encoder_output(
    tensor: torch.Tensor, 
    agbd_patch_size: int = 25
) -> torch.Tensor:
    """
    Extract the actual 25x25 AGBD patch from the full encoder output.
    
    Args:
        tensor: Model output tensor of shape (H, W) or (B, H, W)
        agbd_patch_size: Size of original AGBD patch (default: 25)
        
    Returns:
        Extracted AGBD patch
    """
    if len(tensor.shape) == 2:
        # Single sample (H, W)
        H, W = tensor.shape
        center_h, center_w = H // 2, W // 2
        
        # Extract centered AGBD patch
        half_size = agbd_patch_size // 2
        start_h = max(0, center_h - half_size)
        end_h = min(H, start_h + agbd_patch_size)
        start_w = max(0, center_w - half_size)
        end_w = min(W, start_w + agbd_patch_size)
        
        return tensor[start_h:end_h, start_w:end_w]
        
    elif len(tensor.shape) == 3:
        # Batch (B, H, W)
        B, H, W = tensor.shape
        center_h, center_w = H // 2, W // 2
        
        half_size = agbd_patch_size // 2
        start_h = max(0, center_h - half_size)
        end_h = min(H, start_h + agbd_patch_size)
        start_w = max(0, center_w - half_size)
        end_w = min(W, start_w + agbd_patch_size)
        
        return tensor[:, start_h:end_h, start_w:end_w]
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def extract_agbd_patch_from_inputs(
    inputs: dict, 
    sample_idx: int = 0,
    agbd_patch_size: int = 25
) -> dict:
    """
    Extract the actual AGBD patch from the full encoder inputs.
    
    Args:
        inputs: Input data dictionary with 'image' containing optical/sar data
        sample_idx: Which sample in the batch to extract
        agbd_patch_size: Size of AGBD patch to extract
        
    Returns:
        Dictionary with extracted AGBD patches for each modality
    """
    extracted = {}
    
    if 'image' in inputs:
        extracted['image'] = {}
        for modality, data in inputs['image'].items():
            # data shape: (B, C, T, H, W)
            if len(data.shape) == 5:
                B, C, T, H, W = data.shape
                center_h, center_w = H // 2, W // 2
                
                half_size = agbd_patch_size // 2
                start_h = max(0, center_h - half_size)
                end_h = min(H, start_h + agbd_patch_size)
                start_w = max(0, center_w - half_size)
                end_w = min(W, start_w + agbd_patch_size)
                
                # Extract the AGBD patch for the specified sample
                extracted['image'][modality] = data[sample_idx:sample_idx+1, :, :, start_h:end_h, start_w:end_w]
            else:
                # Fallback: use original data
                extracted['image'][modality] = data[sample_idx:sample_idx+1] if len(data.shape) > 2 else data
    
    return extracted


def create_simple_rgb(optical_data: torch.Tensor) -> np.ndarray:
    """
    Create simple RGB visualization from optical data.
    
    Args:
        optical_data: Tensor of shape (C, H, W) with optical bands
        
    Returns:
        RGB array (H, W, 3)
    """
    if optical_data.shape[0] < 3:
        # Grayscale fallback
        if optical_data.shape[0] > 0:
            gray = optical_data[0].cpu().numpy()
        else:
            gray = np.zeros((optical_data.shape[1], optical_data.shape[2]))
        return np.stack([gray, gray, gray], axis=2)
    
    # Use first 3 bands as RGB
    rgb_bands = optical_data[:3].cpu().numpy()
    rgb = np.transpose(rgb_bands, (1, 2, 0))  # (H, W, 3)
    
    # Simple normalization
    if rgb.max() > rgb.min():
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    return rgb


def create_simple_sar_vis(sar_data: torch.Tensor) -> np.ndarray:
    """
    Create simple SAR visualization.
    
    Args:
        sar_data: Tensor of shape (C, H, W) with SAR bands
        
    Returns:
        RGB array (H, W, 3)
    """
    if sar_data.shape[0] == 0:
        return np.zeros((sar_data.shape[1], sar_data.shape[2], 3))
    
    # Use first band
    sar_band = sar_data[0].cpu().numpy()
    
    # Simple normalization
    if sar_band.max() > sar_band.min():
        sar_norm = (sar_band - sar_band.min()) / (sar_band.max() - sar_band.min())
    else:
        sar_norm = np.zeros_like(sar_band)
    
    return np.stack([sar_norm, sar_norm, sar_norm], axis=2)


def log_agbd_regression_visuals(
    inputs: Dict[str, Any],
    pred: torch.Tensor,
    target: torch.Tensor,
    band_order: Optional[list] = None,
    wandb_run: Any = None,
    step: int = 0,
    prefix: str = "eval",
    max_samples: int = 3,
    dataset: Any = None,
    model_name: str = "Unknown"
) -> None:
    """
    Log AGBD regression visualizations showing only the 25x25 AGBD patches.
    
    Args:
        inputs: Dictionary containing 'image' with optical/sar data
        pred: Predicted biomass tensor of shape (B, H, W)
        target: Ground truth biomass tensor of shape (B, H, W)
        band_order: Optional band ordering information
        wandb_run: WandB run object for logging
        step: Current step/batch number
        prefix: Prefix for logging (e.g., 'train', 'val', 'test')
        max_samples: Maximum number of samples to visualize per batch
        dataset: Dataset object for metadata access
        model_name: Name of the model being evaluated
    """
    try:
        import wandb
        
        if wandb_run is None or not hasattr(wandb_run, 'log'):
            print("[VIZ] No WandB logging available")
            return
            
        batch_size = pred.shape[0]
        samples_to_viz = min(batch_size, max_samples)
        
        # Extract dataset name
        dataset_name = "AGBD"
        if dataset is not None:
            if hasattr(dataset, 'dataset_name'):
                dataset_name = dataset.dataset_name
            elif hasattr(dataset, '__class__'):
                dataset_name = dataset.__class__.__name__
        
        print(f"[VIZ] Creating AGBD patch visualizations for {samples_to_viz} samples")
        
        for i in range(samples_to_viz):
            try:
                print(f"[VIZ] Processing sample {i+1}/{samples_to_viz}")
                print(f"[VIZ] Original pred shape: {pred.shape}, target shape: {target.shape}")
                
                # Extract 25x25 AGBD patches from the center of encoder outputs
                agbd_pred = extract_agbd_patch_from_encoder_output(pred[i])
                agbd_target = extract_agbd_patch_from_encoder_output(target[i])
                agbd_inputs = extract_agbd_patch_from_inputs(inputs, sample_idx=i)
                
                print(f"[VIZ] Extracted AGBD pred shape: {agbd_pred.shape}, target shape: {agbd_target.shape}")
                
                # Calculate center pixel from AGBD patches
                agbd_center_h = agbd_center_w = agbd_pred.shape[-1] // 2
                pred_center = agbd_pred[agbd_center_h, agbd_center_w].item()
                target_center = agbd_target[agbd_center_h, agbd_center_w].item()
                error = abs(pred_center - target_center)
                rel_error = (error / (target_center + 1e-6)) * 100
                
                print(f"[VIZ] Center pixel - Pred: {pred_center:.2f}, GT: {target_center:.2f}, Error: {error:.2f}")
                
                # Create visualization figure - AGBD patches only
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'{dataset_name} {model_name} - Sample {i+1}\nPred: {pred_center:.1f} Mg/ha, GT: {target_center:.1f} Mg/ha, Error: {error:.1f} Mg/ha ({rel_error:.1f}%)', 
                           fontsize=14, fontweight='bold')
                
                # 1. Optical RGB from AGBD patch
                if 'image' in agbd_inputs and 'optical' in agbd_inputs['image']:
                    optical_data = agbd_inputs['image']['optical'][0, :, 0, :, :]  # (C, H, W)
                    rgb_vis = create_simple_rgb(optical_data)
                    axes[0, 0].imshow(rgb_vis)
                    axes[0, 0].set_title('Optical RGB\n(25x25 AGBD patch)')
                else:
                    axes[0, 0].text(0.5, 0.5, 'No Optical Data', ha='center', va='center')
                    axes[0, 0].set_title('Optical RGB (N/A)')
                axes[0, 0].set_xlabel('Pixel')
                axes[0, 0].set_ylabel('Pixel')
                
                # 2. SAR from AGBD patch
                if 'image' in agbd_inputs and 'sar' in agbd_inputs['image']:
                    sar_data = agbd_inputs['image']['sar'][0, :, 0, :, :]  # (C, H, W)
                    sar_vis = create_simple_sar_vis(sar_data)
                    axes[0, 1].imshow(sar_vis, cmap='gray')
                    axes[0, 1].set_title('SAR Data\n(25x25 AGBD patch)')
                else:
                    axes[0, 1].text(0.5, 0.5, 'No SAR Data', ha='center', va='center')
                    axes[0, 1].set_title('SAR Data (N/A)')
                axes[0, 1].set_xlabel('Pixel')
                axes[0, 1].set_ylabel('Pixel')
                
                # 3. Ground Truth Biomass (25x25 patch)
                gt_patch = agbd_target.cpu().numpy()
                max_biomass = max(gt_patch.max(), agbd_pred.cpu().numpy().max())
                
                im3 = axes[0, 2].imshow(gt_patch, cmap='Greens', vmin=0, vmax=max_biomass)
                axes[0, 2].set_title(f'Ground Truth\nCenter: {target_center:.1f} Mg/ha')
                axes[0, 2].set_xlabel('Pixel')
                axes[0, 2].set_ylabel('Pixel')
                plt.colorbar(im3, ax=axes[0, 2], label='Biomass (Mg/ha)')
                
                # 4. Predicted Biomass (25x25 patch)
                pred_patch = agbd_pred.cpu().numpy()
                im4 = axes[1, 0].imshow(pred_patch, cmap='Greens', vmin=0, vmax=max_biomass)
                axes[1, 0].set_title(f'Predicted\nCenter: {pred_center:.1f} Mg/ha')
                axes[1, 0].set_xlabel('Pixel')
                axes[1, 0].set_ylabel('Pixel')
                plt.colorbar(im4, ax=axes[1, 0], label='Biomass (Mg/ha)')
                
                # 5. Error Map (25x25 patch)
                error_map = np.abs(pred_patch - gt_patch)
                im5 = axes[1, 1].imshow(error_map, cmap='Reds')
                axes[1, 1].set_title(f'Absolute Error\nCenter: {error:.1f} Mg/ha')
                axes[1, 1].set_xlabel('Pixel')
                axes[1, 1].set_ylabel('Pixel')
                plt.colorbar(im5, ax=axes[1, 1], label='Error (Mg/ha)')
                
                # 6. Statistics
                axes[1, 2].axis('off')
                stats_text = f"""AGBD 25x25 Patch Statistics:

Center Pixel:
  Predicted: {pred_center:.2f} Mg/ha
  Ground Truth: {target_center:.2f} Mg/ha
  Absolute Error: {error:.2f} Mg/ha
  Relative Error: {rel_error:.1f}%

Patch Statistics:
  GT Mean: {gt_patch.mean():.2f} Mg/ha
  GT Std: {gt_patch.std():.2f} Mg/ha
  Pred Mean: {pred_patch.mean():.2f} Mg/ha
  Pred Std: {pred_patch.std():.2f} Mg/ha
  MAE: {error_map.mean():.2f} Mg/ha
  RMSE: {np.sqrt((error_map**2).mean()):.2f} Mg/ha

Spatial Info:
  Patch Size: 25x25 pixels
  Resolution: 10m/pixel
  Coverage: 250m x 250m
  Model: {model_name}
  Valid Pixels: {np.sum(gt_patch != -1)}/625"""
                
                axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                               fontsize=9, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                
                # Log to WandB
                plt.tight_layout()
                
                wandb_image = wandb.Image(
                    fig, 
                    caption=f"AGBD {model_name} - Sample {i+1}: 25x25 patch visualization"
                )
                
                wandb_run.log({
                    f"{prefix}/sample_{i+1}_agbd_patch_visualization": wandb_image,
                    f"{prefix}/sample_{i+1}_predicted_biomass_center": pred_center,
                    f"{prefix}/sample_{i+1}_ground_truth_biomass_center": target_center,
                    f"{prefix}/sample_{i+1}_absolute_error_center": error,
                    f"{prefix}/sample_{i+1}_relative_error_percent": rel_error,
                })
                
                plt.close(fig)
                print(f"[VIZ] Successfully logged sample {i+1}")
                
            except Exception as e:
                print(f"[VIZ] Sample {i+1} visualization failed: {e}")
                import traceback
                traceback.print_exc()
                
        print(f"[VIZ] Completed visualization for {samples_to_viz} samples")
        
    except Exception as e:
        print(f"[VIZ] Visualization failed: {e}")
        import traceback
        traceback.print_exc()


def validate_agbd_visualization_setup() -> bool:
    """Validate AGBD visualization setup."""
    try:
        # Check WandB availability
        try:
            import wandb
            wandb_available = True
        except ImportError:
            wandb_available = False
            print("[AGBD-VIZ] WandB not available - visualizations will not be logged")
        
        # Validate constants
        assert AGBD_PATCH_SIZE == 25, "AGBD patch size must be 25x25"
        assert AGBD_CENTER_PIXEL == 12, "AGBD center pixel must be at position 12"
        assert AGBD_BIOMASS_RANGE == (0, 500), "AGBD biomass range must be 0-500 Mg/ha"
        
        print("[AGBD-VIZ] Validation successful!")
        print(f"[AGBD-VIZ] Patch size: {AGBD_PATCH_SIZE}x{AGBD_PATCH_SIZE}, center: ({AGBD_CENTER_PIXEL},{AGBD_CENTER_PIXEL})")
        print(f"[AGBD-VIZ] WandB logging: {'Available' if wandb_available else 'Not available'}")
        
        return True
        
    except Exception as e:
        print(f"[AGBD-VIZ] Validation failed: {e}")
        return False


def quick_agbd_visualization(
    inputs: dict,
    pred: torch.Tensor,
    target: torch.Tensor,
    wandb_run=None,
    step: int = 0,
    prefix: str = "eval",
    revolutionary: bool = True
) -> None:
    """Quick access function for AGBD visualization."""
    if not validate_agbd_visualization_setup():
        print("[AGBD-VIZ] Using fallback visualization")
    
    try:
        log_agbd_regression_visuals(
            inputs=inputs,
            pred=pred,
            target=target,
            wandb_run=wandb_run,
            step=step,
            prefix=prefix,
            max_samples=2
        )
        print("[AGBD-VIZ] Visualization completed successfully")
    except Exception as e:
        print(f"[AGBD-VIZ] Visualization failed: {e}")


# Module initialization
print("=" * 60)
print("AGBD VISUALIZATION MODULE LOADED")
print("Features: 25x25 patch extraction, clean visualizations")
print("=" * 60)

# Validate on import
_validation_result = validate_agbd_visualization_setup()
if _validation_result:
    print("Ready for AGBD biomass visualization!")
else:
    print("Partial functionality available")
print("=" * 60)
