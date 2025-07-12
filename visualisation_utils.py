import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import wandb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def get_rgb(optical, band_order):
    """
    Compose a visually meaningful RGB image from optical bands using percentile stretch.
    Enhanced for AGBD: Better handling of Sentinel-2 bands with improved contrast.
    Args:
        optical: np.ndarray, shape (C, H, W) or (H, W, C)
        band_order: list of band names, e.g., ['B01', 'B02', 'B03', 'B04', ...]
    Returns:
        rgb: np.ndarray, shape (H, W, 3), values in [0,1]
    """
    # Accept torch.Tensor or np.ndarray
    if isinstance(optical, torch.Tensor):
        optical = optical.detach().cpu().numpy()
    # Remove batch dim if present
    if optical.ndim == 4:
        optical = optical[0]
    # Remove time dim if present (AGBD has T=1)
    if optical.ndim == 4:
        optical = optical[:, 0]  # (C, T, H, W) -> (C, H, W)
    
    # Accept (C, H, W) or (H, W, C)
    if optical.shape[0] == len(band_order):
        # (C, H, W)
        idx_r = band_order.index('B04')  # Red
        idx_g = band_order.index('B03')  # Green
        idx_b = band_order.index('B02')  # Blue
        rgb = np.stack([
            optical[idx_r],
            optical[idx_g],
            optical[idx_b]
        ], axis=-1)
    elif optical.shape[-1] == len(band_order):
        # (H, W, C)
        idx_r = band_order.index('B04')
        idx_g = band_order.index('B03')
        idx_b = band_order.index('B02')
        rgb = np.stack([
            optical[..., idx_r],
            optical[..., idx_g],
            optical[..., idx_b]
        ], axis=-1)
    else:
        raise ValueError(f"Unexpected optical shape: {optical.shape}")
    
    # Enhanced percentile stretch for better contrast
    vmin = np.percentile(rgb, 1)
    vmax = np.percentile(rgb, 99)
    rgb = (rgb - vmin) / (vmax - vmin + 1e-8)
    rgb = np.clip(rgb, 0, 1)
    
    # Gamma correction for better visual appearance
    rgb = np.power(rgb, 0.8)
    
    return rgb


def get_false_color_composite(optical, band_order, composite_type='nir'):
    """
    Create false color composites for vegetation analysis.
    Enhanced for AGBD: Multiple composite types for biomass interpretation.
    Args:
        optical: np.ndarray, shape (C, H, W) or (H, W, C)
        band_order: list of band names
        composite_type: str, type of composite ('nir', 'swir', 'agriculture', 'ndvi')
    Returns:
        composite: np.ndarray, shape (H, W, 3), values in [0,1]
    """
    if isinstance(optical, torch.Tensor):
        optical = optical.detach().cpu().numpy()
    if optical.ndim == 4:
        optical = optical[0]
    if optical.ndim == 4:
        optical = optical[:, 0]  # Remove time dim
    
    if optical.shape[0] == len(band_order):
        # (C, H, W) format
        if composite_type == 'nir':
            # NIR-Red-Green (vegetation emphasis)
            idx_r = band_order.index('B08')  # NIR
            idx_g = band_order.index('B04')  # Red
            idx_b = band_order.index('B03')  # Green
        elif composite_type == 'swir':
            # SWIR-NIR-Red (biomass/moisture emphasis)
            idx_r = band_order.index('B11')  # SWIR1
            idx_g = band_order.index('B08')  # NIR
            idx_b = band_order.index('B04')  # Red
        elif composite_type == 'agriculture':
            # Agriculture-specific bands
            idx_r = band_order.index('B11')  # SWIR1
            idx_g = band_order.index('B08')  # NIR
            idx_b = band_order.index('B02')  # Blue
        elif composite_type == 'ndvi':
            # NDVI-based visualization
            nir = optical[band_order.index('B08')]
            red = optical[band_order.index('B04')]
            ndvi = (nir - red) / (nir + red + 1e-8)
            # Create RGB from NDVI
            ndvi_normalized = (ndvi + 1) / 2  # Scale from [-1,1] to [0,1]
            composite = np.stack([ndvi_normalized, ndvi_normalized, np.zeros_like(ndvi)], axis=-1)
            return np.clip(composite, 0, 1)
        else:
            raise ValueError(f"Unknown composite type: {composite_type}")
        
        composite = np.stack([
            optical[idx_r],
            optical[idx_g],
            optical[idx_b]
        ], axis=-1)
    else:
        raise ValueError(f"Unexpected optical shape: {optical.shape}")
    
    # Enhanced contrast stretch
    vmin = np.percentile(composite, 2)
    vmax = np.percentile(composite, 98)
    composite = (composite - vmin) / (vmax - vmin + 1e-8)
    composite = np.clip(composite, 0, 1)
    
    return composite


def get_sar_visualization(sar, band_order=None, visualization_type='dual_pol'):
    """
    Create SAR visualizations optimized for ALOS PALSAR data.
    Enhanced for AGBD: Multiple SAR visualization modes for biomass analysis.
    Args:
        sar: np.ndarray, shape (C, H, W) - SAR data in dB scale
        band_order: list, ['HH', 'HV'] for ALOS
        visualization_type: str, type of visualization
    Returns:
        sar_vis: np.ndarray, shape (H, W, 3) or (H, W), values in [0,1]
    """
    if isinstance(sar, torch.Tensor):
        sar = sar.detach().cpu().numpy()
    if sar.ndim == 4:
        sar = sar[0]  # Remove batch dim
    if sar.ndim == 4:
        sar = sar[:, 0]  # Remove time dim
    
    if band_order is None:
        band_order = ['HH', 'HV']
    
    if visualization_type == 'dual_pol':
        # Dual-polarization RGB composite
        hh = sar[0] if sar.shape[0] > 0 else np.zeros_like(sar[0])
        hv = sar[1] if sar.shape[0] > 1 else np.zeros_like(sar[0])
        
        # HH-HV-HH/HV ratio composite
        ratio = (hh - hv)  # Polarization ratio
        
        # Normalize each channel
        hh_norm = (hh - np.percentile(hh, 2)) / (np.percentile(hh, 98) - np.percentile(hh, 2) + 1e-8)
        hv_norm = (hv - np.percentile(hv, 2)) / (np.percentile(hv, 98) - np.percentile(hv, 2) + 1e-8)
        ratio_norm = (ratio - np.percentile(ratio, 2)) / (np.percentile(ratio, 98) - np.percentile(ratio, 2) + 1e-8)
        
        sar_vis = np.stack([
            np.clip(hh_norm, 0, 1),
            np.clip(hv_norm, 0, 1),
            np.clip(ratio_norm, 0, 1)
        ], axis=-1)
        
    elif visualization_type == 'hh_only':
        # HH polarization only (grayscale)
        hh = sar[0]
        hh_norm = (hh - np.percentile(hh, 2)) / (np.percentile(hh, 98) - np.percentile(hh, 2) + 1e-8)
        sar_vis = np.clip(hh_norm, 0, 1)
        
    elif visualization_type == 'hv_only':
        # HV polarization only (grayscale)
        hv = sar[1] if sar.shape[0] > 1 else sar[0]
        hv_norm = (hv - np.percentile(hv, 2)) / (np.percentile(hv, 98) - np.percentile(hv, 2) + 1e-8)
        sar_vis = np.clip(hv_norm, 0, 1)
        
    elif visualization_type == 'biomass_optimized':
        # Biomass-optimized SAR visualization
        hh = sar[0]
        hv = sar[1] if sar.shape[0] > 1 else sar[0]
        
        # Volume scattering component (related to biomass)
        volume = hv
        # Surface scattering component
        surface = hh - hv
        
        vol_norm = (volume - np.percentile(volume, 5)) / (np.percentile(volume, 95) - np.percentile(volume, 5) + 1e-8)
        surf_norm = (surface - np.percentile(surface, 5)) / (np.percentile(surface, 95) - np.percentile(surface, 5) + 1e-8)
        hh_norm = (hh - np.percentile(hh, 5)) / (np.percentile(hh, 95) - np.percentile(hh, 5) + 1e-8)
        
        sar_vis = np.stack([
            np.clip(vol_norm, 0, 1),     # Red: Volume scattering (biomass)
            np.clip(surf_norm, 0, 1),    # Green: Surface scattering
            np.clip(hh_norm, 0, 1)       # Blue: Total power
        ], axis=-1)
        
    else:
        raise ValueError(f"Unknown SAR visualization type: {visualization_type}")
    
    return sar_vis


def compute_vegetation_indices(optical, band_order):
    """
    Compute vegetation indices relevant for biomass estimation.
    Enhanced for AGBD: Multiple indices for comprehensive vegetation analysis.
    """
    if isinstance(optical, torch.Tensor):
        optical = optical.detach().cpu().numpy()
    if optical.ndim == 4:
        optical = optical[0]
    if optical.ndim == 4:
        optical = optical[:, 0]  # Remove time dim
    
    # Extract bands
    nir = optical[band_order.index('B08')]    # NIR
    red = optical[band_order.index('B04')]    # Red
    blue = optical[band_order.index('B02')]   # Blue
    green = optical[band_order.index('B03')]  # Green
    swir1 = optical[band_order.index('B11')]  # SWIR1
    swir2 = optical[band_order.index('B12')]  # SWIR2
    
    indices = {}
    
    # NDVI - Normalized Difference Vegetation Index
    indices['NDVI'] = (nir - red) / (nir + red + 1e-8)
    
    # EVI - Enhanced Vegetation Index
    indices['EVI'] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    
    # SAVI - Soil Adjusted Vegetation Index
    L = 0.5  # Soil brightness correction factor
    indices['SAVI'] = ((nir - red) / (nir + red + L)) * (1 + L)
    
    # NDMI - Normalized Difference Moisture Index
    indices['NDMI'] = (nir - swir1) / (nir + swir1 + 1e-8)
    
    # NBR - Normalized Burn Ratio
    indices['NBR'] = (nir - swir2) / (nir + swir2 + 1e-8)
    
    # Green NDVI
    indices['GNDVI'] = (nir - green) / (nir + green + 1e-8)
    
    return indices


def create_biomass_interpretation_text(agbd_value, metadata=None):
    """
    Create interpretive text for AGBD values with ecological context.
    Enhanced for AGBD: Comprehensive biomass interpretation.
    """
    # Convert normalized AGBD back to Mg/ha (approximate)
    # Using normalization: (agbd - 66.97) / 98.67
    agbd_mgha = agbd_value * 98.67 + 66.97
    
    interpretation = []
    interpretation.append(f"AGBD: {agbd_mgha:.1f} Mg/ha")
    
    # Biomass categories based on ecological standards
    if agbd_mgha < 10:
        category = "Very Low"
        description = "Sparse vegetation, grassland, or recently disturbed areas"
        color = "#D32F2F"  # Red
    elif agbd_mgha < 50:
        category = "Low"
        description = "Shrubland, young forest, or agricultural areas"
        color = "#FF9800"  # Orange
    elif agbd_mgha < 100:
        category = "Moderate"
        description = "Secondary forest or managed forest areas"
        color = "#FFC107"  # Amber
    elif agbd_mgha < 200:
        category = "High"
        description = "Mature forest with substantial biomass"
        color = "#4CAF50"  # Green
    elif agbd_mgha < 350:
        category = "Very High"
        description = "Dense mature forest, high carbon storage"
        color = "#2E7D32"  # Dark Green
    else:
        category = "Extremely High"
        description = "Primary/old-growth forest, maximum carbon storage"
        color = "#1B5E20"  # Very Dark Green
    
    interpretation.append(f"Category: {category}")
    interpretation.append(f"Context: {description}")
    
    # Add carbon content estimation (biomass * 0.47 for carbon fraction)
    carbon_content = agbd_mgha * 0.47
    interpretation.append(f"Est. Carbon: {carbon_content:.1f} Mg C/ha")
    
    # Add metadata if available
    if metadata:
        if 'lat' in metadata and 'lon' in metadata:
            interpretation.append(f"Location: {metadata['lat']:.3f}°N, {metadata['lon']:.3f}°E")
        if 'tile_name' in metadata:
            interpretation.append(f"Tile: {metadata['tile_name']}")
    
    return "\n".join(interpretation), color


def log_regression_images_wandb(image_dict, target, pred, band_order, wandb_run, step=None, prefix="val", max_samples=3):
    print(f"[DEBUG] log_regression_images_wandb called with prefix={prefix}, step={step}")
    # =================== DATA PREPARATION ===================
    def to_np(x): 
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
    
    optical = to_np(image_dict.get('optical'))  # (B, C, T, H, W)
    sar = to_np(image_dict.get('sar'))          # (B, C, T, H, W) 
    target_np = to_np(target)                   # (B, H, W) or (B,)
    pred_np = to_np(pred)                       # (B, H, W) or (B,)
    
    # Handle different tensor formats
    if optical.ndim == 5:  # (B, C, T, H, W)
        optical = optical[:, :, 0, :, :]  # Remove time dimension -> (B, C, H, W)
    if sar.size > 0 and sar.ndim == 5:
        sar = sar[:, :, 0, :, :]  # Remove time dimension -> (B, C, H, W)
    elif sar.size == 0:
        # Create placeholder for empty SAR data
        sar = np.array([])  # Keep as empty array
    
    B = target_np.shape[0] if target_np.ndim > 0 else 1
    
    # Handle center pixel extraction for patch-based targets
    if target_np.ndim == 3:  # (B, H, W) - patch targets
        H, W = target_np.shape[1], target_np.shape[2]
        cy, cx = H//2, W//2
        gt_centers = target_np[:, cy, cx]
        pred_centers = pred_np[:, cy, cx]
        is_patch_regression = True
    else:  # (B,) - scalar targets
        gt_centers = target_np.flatten()
        pred_centers = pred_np.flatten()
        is_patch_regression = False
    
    # =================== BATCH METRICS ===================
    mae_batch = mean_absolute_error(gt_centers, pred_centers)
    mse_batch = mean_squared_error(gt_centers, pred_centers)
    rmse_batch = np.sqrt(mse_batch)
    r2_batch = r2_score(gt_centers, pred_centers)
    
    # AGBD-specific metrics (convert back to Mg/ha for interpretation)
    gt_mgha = gt_centers * 98.67 + 66.97
    pred_mgha = pred_centers * 98.67 + 66.97
    
    mean_error_mgha = np.mean(pred_mgha - gt_mgha)
    rmse_mgha = np.sqrt(mean_squared_error(gt_mgha, pred_mgha))
    mape = np.mean(np.abs((gt_mgha - pred_mgha) / (gt_mgha + 1e-8))) * 100
    
    # Log comprehensive metrics
    metrics = {
        f'{prefix}/MAE_normalized': mae_batch,
        f'{prefix}/RMSE_normalized': rmse_batch,
        f'{prefix}/R2': r2_batch,
        f'{prefix}/MAE_Mg_ha': np.mean(np.abs(pred_mgha - gt_mgha)),
        f'{prefix}/RMSE_Mg_ha': rmse_mgha,
        f'{prefix}/Mean_Error_Mg_ha': mean_error_mgha,
        f'{prefix}/MAPE_percent': mape,
        f'{prefix}/Mean_GT_Mg_ha': np.mean(gt_mgha),
        f'{prefix}/Mean_Pred_Mg_ha': np.mean(pred_mgha)
    }
    print(f"[DEBUG] Logging metrics to wandb: {metrics}")
    try:
        if hasattr(wandb_run, 'log'):
            wandb_run.log(metrics, step=step)
        else:
            wandb.log(metrics, step=step)
    except Exception as e:
        print(f"[ERROR] wandb.log(metrics) failed: {e}")
    # =================== SAMPLE VISUALIZATIONS ===================
    # Skip visualization if optical data is invalid
    if optical.size == 0 or optical.ndim < 3 or B == 0:
        print(f"Warning: Skipping visualization due to invalid optical data shape: {optical.shape}")
        return
    samples = random.sample(range(B), min(max_samples, B))
    panels = []
    sample_rows = []
    
    # SAR band order for ALOS
    sar_band_order = ['HH', 'HV']
    
    for idx in samples:
        try:
            # Validate optical data for this sample
            if idx >= optical.shape[0] or optical[idx].size == 0:
                print(f"Warning: Skipping sample {idx} due to invalid optical data")
                continue
                
            # Create comprehensive figure with multiple subplots
            fig = plt.figure(figsize=(20, 16))
            gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.3)
            
            # ============= TOP ROW: INPUT VISUALIZATIONS =============
            
            # 1. RGB True Color
            ax1 = fig.add_subplot(gs[0, 0])
            try:
                rgb_img = get_rgb(optical[idx], band_order)
                ax1.imshow(rgb_img)
                ax1.set_title('Sentinel-2 RGB\n(True Color)', fontsize=10, fontweight='bold')
            except Exception as e:
                print(f"Warning: Failed to create RGB image for sample {idx}: {e}")
                ax1.text(0.5, 0.5, 'RGB\nUnavailable', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Sentinel-2 RGB\n(Error)', fontsize=10, fontweight='bold')
            ax1.axis('off')
            
            # 2. False Color NIR
            ax2 = fig.add_subplot(gs[0, 1])
            try:
                nir_img = get_false_color_composite(optical[idx], band_order, 'nir')
                ax2.imshow(nir_img)
                ax2.set_title('NIR False Color\n(Vegetation)', fontsize=10, fontweight='bold')
            except Exception as e:
                print(f"Warning: Failed to create NIR image for sample {idx}: {e}")
                ax2.text(0.5, 0.5, 'NIR\nUnavailable', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('NIR False Color\n(Error)', fontsize=10, fontweight='bold')
            ax2.axis('off')
            
            # 3. SWIR Composite
            ax3 = fig.add_subplot(gs[0, 2])
            try:
                swir_img = get_false_color_composite(optical[idx], band_order, 'swir')
                ax3.imshow(swir_img)
                ax3.set_title('SWIR Composite\n(Biomass/Moisture)', fontsize=10, fontweight='bold')
            except Exception as e:
                print(f"Warning: Failed to create SWIR image for sample {idx}: {e}")
                ax3.text(0.5, 0.5, 'SWIR\nUnavailable', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('SWIR Composite\n(Error)', fontsize=10, fontweight='bold')
            ax3.axis('off')
            
            # 4. SAR Dual-Pol (only if SAR data is available)
            ax4 = fig.add_subplot(gs[0, 3])
            if sar.size > 0 and sar.ndim >= 3 and len(sar.shape) >= 4 and sar.shape[0] > idx:
                try:
                    sar_img = get_sar_visualization(sar[idx], sar_band_order, 'biomass_optimized')
                    ax4.imshow(sar_img)
                    ax4.set_title('ALOS SAR\n(Biomass Optimized)', fontsize=10, fontweight='bold')
                except Exception as e:
                    # Fallback if SAR visualization fails
                    ax4.text(0.5, 0.5, f'SAR Error:\n{str(e)[:50]}...', 
                            ha='center', va='center', transform=ax4.transAxes,
                            fontsize=10, fontweight='bold', color='red')
                    ax4.set_title('ALOS SAR\n(Error)', fontsize=10, fontweight='bold', color='red')
            else:
                # Show placeholder when SAR data is not available
                ax4.text(0.5, 0.5, 'SAR Data\nNot Available', 
                        ha='center', va='center', transform=ax4.transAxes,
                        fontsize=12, fontweight='bold', color='gray')
                ax4.set_title('ALOS SAR\n(Not Available)', fontsize=10, fontweight='bold', color='gray')
            ax4.axis('off')
            
            # 5. NDVI
            ax5 = fig.add_subplot(gs[0, 4])
            ndvi_img = get_false_color_composite(optical[idx], band_order, 'ndvi')
            im5 = ax5.imshow(ndvi_img[:,:,0], cmap='RdYlGn', vmin=0, vmax=1)
            ax5.set_title('NDVI\n(Vegetation Index)', fontsize=10, fontweight='bold')
            ax5.axis('off')
            plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
            
            # ============= SECOND ROW: REGRESSION ANALYSIS =============
            
            # 6. Target vs Prediction (patch view or center pixel)
            ax6 = fig.add_subplot(gs[1, 0])
            if is_patch_regression:
                # Show target patch with center pixel highlighted
                im6 = ax6.imshow(target_np[idx], cmap='viridis')
                # Highlight center pixel
                cy, cx = target_np.shape[1]//2, target_np.shape[2]//2
                rect = Rectangle((cx-0.5, cy-0.5), 1, 1, linewidth=2, 
                               edgecolor='red', facecolor='none')
                ax6.add_patch(rect)
                ax6.set_title(f'GT Patch\nCenter: {gt_centers[idx]:.3f}', 
                             fontsize=10, fontweight='bold')
                plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
            else:
                # Scalar regression - show as bar
                ax6.bar(['Ground Truth'], [gt_centers[idx]], color='blue', alpha=0.7)
                ax6.set_title('Ground Truth\n(Scalar)', fontsize=10, fontweight='bold')
                ax6.set_ylabel('AGBD (normalized)')
            ax6.set_xticks([])
        
            # 7. Prediction
            ax7 = fig.add_subplot(gs[1, 1])
            if is_patch_regression:
                im7 = ax7.imshow(pred_np[idx], cmap='viridis')
                rect = Rectangle((cx-0.5, cy-0.5), 1, 1, linewidth=2, 
                            edgecolor='red', facecolor='none')
                ax7.add_patch(rect)
                ax7.set_title(f'Prediction\nCenter: {pred_centers[idx]:.3f}', 
                            fontsize=10, fontweight='bold')
                plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
            else:
                ax7.bar(['Prediction'], [pred_centers[idx]], color='green', alpha=0.7)
                ax7.set_title('Prediction\n(Scalar)', fontsize=10, fontweight='bold')
                ax7.set_ylabel('AGBD (normalized)')
            ax7.set_xticks([])
            
            # 8. Error Analysis
            ax8 = fig.add_subplot(gs[1, 2])
            abs_error = abs(pred_centers[idx] - gt_centers[idx])
            if is_patch_regression:
                error_map = np.abs(pred_np[idx] - target_np[idx])
                im8 = ax8.imshow(error_map, cmap='Reds')
                ax8.set_title(f'Error Map\nCenter Error: {abs_error:.3f}', 
                            fontsize=10, fontweight='bold')
                plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
            else:
                ax8.bar(['Abs Error'], [abs_error], color='red', alpha=0.7)
                ax8.set_title(f'Absolute Error\n{abs_error:.3f}', 
                            fontsize=10, fontweight='bold')
            ax8.set_xticks([])
            
            # 9. Biomass Interpretation
            ax9 = fig.add_subplot(gs[1, 3])
            ax9.axis('off')
            
            # Get metadata if available
            metadata = image_dict.get('metadata', [{}])
            sample_metadata = metadata[idx] if len(metadata) > idx else {}
            
            # Create interpretation for both GT and prediction
            gt_text, gt_color = create_biomass_interpretation_text(gt_centers[idx], sample_metadata)
            pred_text, pred_color = create_biomass_interpretation_text(pred_centers[idx], sample_metadata)
            
            interpretation_text = f"GROUND TRUTH:\n{gt_text}\n\nPREDICTION:\n{pred_text}"
            ax9.text(0.05, 0.95, interpretation_text, transform=ax9.transAxes, 
                    fontsize=9, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax9.set_title('Biomass Interpretation', fontsize=10, fontweight='bold')
            
            # 10. Vegetation Indices
            ax10 = fig.add_subplot(gs[1, 4])
            vi = compute_vegetation_indices(optical[idx], band_order)
            vi_names = list(vi.keys())[:4]  # Show top 4 indices
            vi_values = [np.mean(vi[name]) for name in vi_names]
            
            bars = ax10.barh(vi_names, vi_values, color=['green', 'darkgreen', 'olive', 'forestgreen'])
            ax10.set_title('Vegetation Indices\n(Mean Values)', fontsize=10, fontweight='bold')
            ax10.set_xlabel('Index Value')
            
            # Add values as text
            for i, (name, value) in enumerate(zip(vi_names, vi_values)):
                ax10.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=8)
            
            # ============= THIRD ROW: DETAILED SAR ANALYSIS =============
            
            # 11. SAR HH
            ax11 = fig.add_subplot(gs[2, 0])
            if sar.size > 0 and sar.ndim >= 3 and len(sar.shape) >= 4 and sar.shape[0] > idx:
                try:
                    sar_hh_vis = get_sar_visualization(sar[idx], sar_band_order, 'hh_only')
                    im11 = ax11.imshow(sar_hh_vis, cmap='gray')
                    ax11.set_title('SAR HH\n(Co-polarized)', fontsize=10, fontweight='bold')
                    plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04)
                except Exception as e:
                    ax11.text(0.5, 0.5, 'SAR HH\nNot Available', 
                            ha='center', va='center', transform=ax11.transAxes,
                            fontsize=10, fontweight='bold', color='gray')
                    ax11.set_title('SAR HH\n(Not Available)', fontsize=10, fontweight='bold', color='gray')
            else:
                ax11.text(0.5, 0.5, 'SAR HH\nNot Available', 
                        ha='center', va='center', transform=ax11.transAxes,
                        fontsize=10, fontweight='bold', color='gray')
                ax11.set_title('SAR HH\n(Not Available)', fontsize=10, fontweight='bold', color='gray')
            ax11.axis('off')
            
            # 12. SAR HV  
            ax12 = fig.add_subplot(gs[2, 1])
            if sar.size > 0 and sar.ndim >= 3 and len(sar.shape) >= 4 and sar.shape[0] > idx:
                try:
                    sar_hv_vis = get_sar_visualization(sar[idx], sar_band_order, 'hv_only')
                    im12 = ax12.imshow(sar_hv_vis, cmap='gray')
                    ax12.set_title('SAR HV\n(Cross-polarized)', fontsize=10, fontweight='bold')
                    plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)
                except Exception as e:
                    ax12.text(0.5, 0.5, 'SAR HV\nNot Available', 
                            ha='center', va='center', transform=ax12.transAxes,
                            fontsize=10, fontweight='bold', color='gray')
                    ax12.set_title('SAR HV\n(Not Available)', fontsize=10, fontweight='bold', color='gray')
            else:
                ax12.text(0.5, 0.5, 'SAR HV\nNot Available', 
                        ha='center', va='center', transform=ax12.transAxes,
                        fontsize=10, fontweight='bold', color='gray')
                ax12.set_title('SAR HV\n(Not Available)', fontsize=10, fontweight='bold', color='gray')
            ax12.axis('off')
            
            # 13. Band Statistics
            ax13 = fig.add_subplot(gs[2, 2])
            ax13.axis('off')
            
            # Calculate comprehensive statistics
            opt_stats = f"OPTICAL STATS (S2):\n"
            opt_stats += f"Shape: {optical[idx].shape}\n"
            opt_stats += f"Min: {optical[idx].min():.3f}\n"
            opt_stats += f"Max: {optical[idx].max():.3f}\n"
            opt_stats += f"Mean: {optical[idx].mean():.3f}\n"
            opt_stats += f"Std: {optical[idx].std():.3f}\n\n"
            
            sar_stats = f"SAR STATS (ALOS):\n"
            if sar.size > 0 and sar.ndim >= 3 and len(sar.shape) >= 4 and sar.shape[0] > idx:
                try:
                    sar_stats += f"Shape: {sar[idx].shape}\n"
                    sar_stats += f"HH Mean: {sar[idx][0].mean():.3f}\n"
                    sar_stats += f"HV Mean: {sar[idx][1].mean():.3f}\n"
                    sar_stats += f"HH Std: {sar[idx][0].std():.3f}\n"
                    sar_stats += f"HV Std: {sar[idx][1].std():.3f}\n"
                except Exception as e:
                    sar_stats += f"Error: {str(e)[:30]}...\n"
            else:
                sar_stats += f"Data: Not Available\n"
                sar_stats += f"Note: Encoder uses optical only\n"
            
            stats_text = opt_stats + sar_stats
            ax13.text(0.05, 0.95, stats_text, transform=ax13.transAxes, 
                    fontsize=8, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
            ax13.set_title('Data Statistics', fontsize=10, fontweight='bold')
            
            # 14. Error Distribution (if patch regression)
            ax14 = fig.add_subplot(gs[2, 3])
            if is_patch_regression:
                error_map = np.abs(pred_np[idx] - target_np[idx])
                error_flat = error_map.flatten()
                
                # Remove invalid values and check if we have valid data for histogram
                error_flat = error_flat[np.isfinite(error_flat)]
                
                if len(error_flat) > 0 and np.max(error_flat) > np.min(error_flat):
                    # Adaptive bin count based on data range
                    data_range = np.max(error_flat) - np.min(error_flat)
                    if data_range > 1e-6:  # Only create histogram if range is meaningful
                        bins = min(20, max(5, len(np.unique(error_flat))))
                        ax14.hist(error_flat, bins=bins, color='purple', alpha=0.7, edgecolor='black')
                        ax14.set_xlabel('Absolute Error')
                        ax14.set_ylabel('Pixel Count')
                    else:
                        # All values are essentially the same
                        ax14.text(0.5, 0.5, f'Uniform Error\n{np.mean(error_flat):.6f}', 
                                transform=ax14.transAxes, ha='center', va='center')
                else:
                    # No valid data
                    ax14.text(0.5, 0.5, 'No Valid\nError Data', 
                            transform=ax14.transAxes, ha='center', va='center')
                
                ax14.set_title('Patch Error\nDistribution', fontsize=10, fontweight='bold')
            else:
                # For scalar, show comparison
                ax14.bar(['GT', 'Pred'], [gt_centers[idx], pred_centers[idx]], 
                        color=['blue', 'green'], alpha=0.7)
                ax14.set_title('GT vs Prediction\nComparison', fontsize=10, fontweight='bold')
                ax14.set_ylabel('AGBD (normalized)')
            
            # 15. Spectral Profile
            ax15 = fig.add_subplot(gs[2, 4])
            # Center pixel spectral profile
            if optical[idx].ndim == 3:  # (C, H, W)
                center_spectrum = optical[idx][:, optical[idx].shape[1]//2, optical[idx].shape[2]//2]
                ax15.plot(range(len(band_order)), center_spectrum, 'o-', linewidth=2, markersize=4)
                ax15.set_xticks(range(len(band_order)))
                ax15.set_xticklabels(band_order, rotation=45, fontsize=8)
                ax15.set_title('Center Pixel\nSpectral Profile', fontsize=10, fontweight='bold')
                ax15.set_ylabel('Reflectance (normalized)')
                ax15.grid(True, alpha=0.3)
            
            # ============= BOTTOM ROW: METADATA AND CONTEXT =============
            
            # 16. Geographic Context
            ax16 = fig.add_subplot(gs[3, :2])
            ax16.axis('off')
            
            metadata_text = "GEOGRAPHIC CONTEXT:\n"
            if sample_metadata:
                if 'lat' in sample_metadata and 'lon' in sample_metadata:
                    metadata_text += f"Coordinates: {sample_metadata['lat']:.4f}°N, {sample_metadata['lon']:.4f}°E\n"
                if 'tile_name' in sample_metadata:
                    metadata_text += f"Tile: {sample_metadata['tile_name']}\n"
                if 'patch_index' in sample_metadata:
                    metadata_text += f"Patch Index: {sample_metadata['patch_index']}\n"
            
            # Add AGBD context
            gt_mgha_sample = gt_centers[idx] * 98.67 + 66.97
            pred_mgha_sample = pred_centers[idx] * 98.67 + 66.97
            metadata_text += f"\nBIOMASS ANALYSIS:\n"
            metadata_text += f"Ground Truth: {gt_mgha_sample:.1f} Mg/ha\n"
            metadata_text += f"Prediction: {pred_mgha_sample:.1f} Mg/ha\n"
            metadata_text += f"Error: {abs(pred_mgha_sample - gt_mgha_sample):.1f} Mg/ha\n"
            metadata_text += f"Carbon Storage (GT): {gt_mgha_sample * 0.47:.1f} Mg C/ha\n"
            
            ax16.text(0.05, 0.95, metadata_text, transform=ax16.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax16.set_title('Sample Metadata & Context', fontsize=12, fontweight='bold')
            
            # 17. Quality Assessment
            ax17 = fig.add_subplot(gs[3, 2:])
            ax17.axis('off')
            
            # Quality metrics for this sample
            sample_error = abs(pred_centers[idx] - gt_centers[idx])
            relative_error = sample_error / (abs(gt_centers[idx]) + 1e-8) * 100
            
            quality_text = "PREDICTION QUALITY ASSESSMENT:\n\n"
            
            if sample_error < 0.1:
                quality = "EXCELLENT"
                quality_color = "green"
            elif sample_error < 0.2:
                quality = "GOOD"
                quality_color = "orange"
            elif sample_error < 0.3:
                quality = "FAIR"
                quality_color = "darkorange"
            else:
                quality = "POOR"
                quality_color = "red"
            
            quality_text += f"Overall Quality: {quality}\n"
            quality_text += f"Absolute Error: {sample_error:.3f} (normalized)\n"
            quality_text += f"Relative Error: {relative_error:.1f}%\n"
            quality_text += f"Error in Mg/ha: {abs(pred_mgha_sample - gt_mgha_sample):.1f}\n\n"
            
            # Add interpretation guidance
            quality_text += "INTERPRETATION GUIDE:\n"
            quality_text += "• EXCELLENT: <10 Mg/ha error\n"
            quality_text += "• GOOD: 10-20 Mg/ha error\n"
            quality_text += "• FAIR: 20-30 Mg/ha error\n"
            quality_text += "• POOR: >30 Mg/ha error\n"
            
            ax17.text(0.05, 0.95, quality_text, transform=ax17.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=quality_color, alpha=0.2))
            ax17.set_title('Quality Assessment & Guide', fontsize=12, fontweight='bold')
            
            # Add sample info to main title
            plt.suptitle(f'AGBD Comprehensive Analysis - Sample {idx}\n'
                        f'Batch Size: {B}, Step: {step}, Split: {prefix.upper()}', 
                        fontsize=14, fontweight='bold', y=0.98)
        
        
            # Save figure - use constrained layout for better spacing
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
                try:
                    plt.tight_layout(pad=2.0)
                except Exception:
                    # If tight_layout fails, try subplots_adjust as fallback
                    try:
                        plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05, 
                                        wspace=0.3, hspace=0.4)
                    except Exception:
                        pass
            
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                panels.append(wandb.Image(Image.open(buf), 
                                        caption=f"Sample {idx}: GT={gt_mgha_sample:.1f} Mg/ha, "
                                            f"Pred={pred_mgha_sample:.1f} Mg/ha, "
                                            f"Error={abs(pred_mgha_sample - gt_mgha_sample):.1f} Mg/ha"))
                buf.close()
            except Exception as e:
                print(f"Warning: Failed to save/upload visualization for sample {idx}: {e}")
                # Create a simple text placeholder instead
                try:
                    panels.append(wandb.Image(np.zeros((100, 100, 3), dtype=np.uint8), 
                                            caption=f"Sample {idx}: Visualization failed - GT={gt_mgha_sample:.1f} Mg/ha, "
                                                f"Pred={pred_mgha_sample:.1f} Mg/ha"))
                except Exception:
                    pass  # Skip if even placeholder fails
            finally:
                plt.close(fig)
            
            # Add to sample metrics table
            sample_rows.append([
                idx, 
                float(gt_centers[idx]), 
                float(pred_centers[idx]), 
                float(abs(pred_centers[idx] - gt_centers[idx])),
                float(gt_mgha_sample),
                float(pred_mgha_sample),
                float(abs(pred_mgha_sample - gt_mgha_sample))
            ])
        
        except Exception as e:
            print(f"[ERROR] Exception in visualization for sample {idx}: {e}")
    print(f"[DEBUG] log_regression_images_wandb finished for step {step}")
    # Reminder for user about wandb media limit
    print("[INFO] If you want to log more than 8 images per step, set wandb.init(settings=wandb.Settings(max_media_per_step=25)) at the start of your run.")
    
    # =================== BATCH-LEVEL VISUALIZATIONS ===================
    
    try:
        # Comprehensive scatter plot
        scatter_fig, scatter_axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Normalized values scatter
        scatter_axes[0].scatter(gt_centers, pred_centers, c='blue', alpha=0.6, s=30)
        scatter_axes[0].plot([gt_centers.min(), gt_centers.max()], 
                            [gt_centers.min(), gt_centers.max()], 'r--', linewidth=2)
        scatter_axes[0].set_xlabel('Ground Truth (normalized)')
        scatter_axes[0].set_ylabel('Prediction (normalized)')
        scatter_axes[0].set_title(f'Predictions vs Ground Truth\nR² = {r2_batch:.3f}')
        scatter_axes[0].grid(True, alpha=0.3)
        
        # Mg/ha values scatter  
        scatter_axes[1].scatter(gt_mgha, pred_mgha, c='green', alpha=0.6, s=30)
        scatter_axes[1].plot([gt_mgha.min(), gt_mgha.max()], 
                            [gt_mgha.min(), gt_mgha.max()], 'r--', linewidth=2)
        scatter_axes[1].set_xlabel('Ground Truth (Mg/ha)')
        scatter_axes[1].set_ylabel('Prediction (Mg/ha)')
        scatter_axes[1].set_title(f'Biomass Predictions vs Ground Truth\nRMSE = {rmse_mgha:.1f} Mg/ha')
        scatter_axes[1].grid(True, alpha=0.3)
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
            try:
                plt.tight_layout()
            except Exception:
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
        
        try:
            scatter_buf = io.BytesIO()
            scatter_fig.savefig(scatter_buf, format='png', bbox_inches='tight', dpi=150)
            scatter_buf.seek(0)
            scatter_img = wandb.Image(Image.open(scatter_buf), 
                                    caption=f'Batch scatter plots (N={B})')
            scatter_buf.close()
        except Exception as e:
            print(f"Warning: Failed to save scatter plot: {e}")
            scatter_img = wandb.Image(np.zeros((100, 100, 3), dtype=np.uint8), 
                                    caption=f'Scatter plot failed (N={B})')
        finally:
            plt.close(scatter_fig)
    except Exception as e:
        print(f"Warning: Failed to create scatter plot: {e}")
        scatter_img = wandb.Image(np.zeros((100, 100, 3), dtype=np.uint8), 
                                caption=f'Scatter plot failed (N={B})')
    
    # Error distribution histogram
    hist_fig, hist_axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Normalized error distribution
    errors_norm = np.abs(pred_centers - gt_centers)
    hist_axes[0].hist(errors_norm, bins=20, color='orange', alpha=0.7, edgecolor='black')
    hist_axes[0].axvline(np.mean(errors_norm), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(errors_norm):.3f}')
    hist_axes[0].set_xlabel('Absolute Error (normalized)')
    hist_axes[0].set_ylabel('Count')
    hist_axes[0].set_title('Error Distribution (Normalized)')
    hist_axes[0].legend()
    hist_axes[0].grid(True, alpha=0.3)
    
    # Mg/ha error distribution
    errors_mgha = np.abs(pred_mgha - gt_mgha)
    hist_axes[1].hist(errors_mgha, bins=20, color='red', alpha=0.7, edgecolor='black')
    hist_axes[1].axvline(np.mean(errors_mgha), color='darkred', linestyle='--', 
                        label=f'Mean: {np.mean(errors_mgha):.1f} Mg/ha')
    hist_axes[1].set_xlabel('Absolute Error (Mg/ha)')
    hist_axes[1].set_ylabel('Count')
    hist_axes[1].set_title('Biomass Error Distribution')
    hist_axes[1].legend()
    hist_axes[1].grid(True, alpha=0.3)
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        try:
            plt.tight_layout()
        except Exception:
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
    hist_buf = io.BytesIO()
    hist_fig.savefig(hist_buf, format='png', bbox_inches='tight', dpi=150)
    hist_buf.seek(0)
    hist_img = wandb.Image(Image.open(hist_buf), 
                          caption=f'Error distributions (N={B})')
    plt.close(hist_fig)
    hist_buf.close()
    
    # Biomass distribution analysis
    dist_fig, dist_axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # GT vs Pred distributions
    dist_axes[0].hist(gt_mgha, bins=20, alpha=0.6, label='Ground Truth', color='blue')
    dist_axes[0].hist(pred_mgha, bins=20, alpha=0.6, label='Predictions', color='green')
    dist_axes[0].set_xlabel('Biomass (Mg/ha)')
    dist_axes[0].set_ylabel('Count')
    dist_axes[0].set_title('Biomass Value Distributions')
    dist_axes[0].legend()
    dist_axes[0].grid(True, alpha=0.3)
    
    # Biomass categories
    def categorize_biomass(agbd_mgha):
        if agbd_mgha < 10:
            return "Very Low"
        elif agbd_mgha < 50:
            return "Low"
        elif agbd_mgha < 100:
            return "Moderate"
        elif agbd_mgha < 200:
            return "High"
        elif agbd_mgha < 350:
            return "Very High"
        else:
            return "Extremely High"
    
    gt_categories = [categorize_biomass(x) for x in gt_mgha]
    pred_categories = [categorize_biomass(x) for x in pred_mgha]
    
    categories = ["Very Low", "Low", "Moderate", "High", "Very High", "Extremely High"]
    gt_counts = [gt_categories.count(cat) for cat in categories]
    pred_counts = [pred_categories.count(cat) for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    dist_axes[1].bar(x - width/2, gt_counts, width, label='Ground Truth', color='blue', alpha=0.7)
    dist_axes[1].bar(x + width/2, pred_counts, width, label='Predictions', color='green', alpha=0.7)
    dist_axes[1].set_xlabel('Biomass Category')
    dist_axes[1].set_ylabel('Count')
    dist_axes[1].set_title('Biomass Category Distribution')
    dist_axes[1].set_xticks(x)
    dist_axes[1].set_xticklabels(categories, rotation=45)
    dist_axes[1].legend()
    dist_axes[1].grid(True, alpha=0.3)
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        try:
            plt.tight_layout()
        except Exception:
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
    dist_buf = io.BytesIO()
    dist_fig.savefig(dist_buf, format='png', bbox_inches='tight', dpi=150)
    dist_buf.seek(0)
    dist_img = wandb.Image(Image.open(dist_buf), 
                          caption=f'Biomass distributions and categories (N={B})')
    plt.close(dist_fig)
    dist_buf.close()
    
    # =================== COMPREHENSIVE LOGGING ===================
    
    # Create comprehensive metrics table
    table_columns = ["Sample_ID", "GT_norm", "Pred_norm", "Error_norm", 
                    "GT_Mg_ha", "Pred_Mg_ha", "Error_Mg_ha"]
    table = wandb.Table(data=sample_rows, columns=table_columns)
    
    # Log everything to wandb with proper step handling
    log_dict = {
        f"{prefix}/samples_comprehensive": panels,
        f"{prefix}/scatter_analysis": scatter_img,
        f"{prefix}/error_distributions": hist_img,
        f"{prefix}/biomass_distributions": dist_img,
        f"{prefix}/detailed_metrics_table": table
    }
    
    # Use a default step if none provided to avoid wandb step warnings
    if step is None:
        try:
            import time
            step = int(time.time()) % 10000  # Use timestamp mod to avoid large numbers
        except:
            step = 0
    
    if hasattr(wandb_run, 'log'): 
        wandb_run.log(log_dict, step=step)
    else: 
        wandb.log(log_dict, step=step)
    
    print(f"AGBD Visualization Complete! Logged {len(panels)} comprehensive sample analyses")
    print(f"Batch Metrics - RMSE: {rmse_mgha:.1f} Mg/ha, R²: {r2_batch:.3f}, MAPE: {mape:.1f}%")