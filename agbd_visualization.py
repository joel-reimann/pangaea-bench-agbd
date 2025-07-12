import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import wandb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def auto_band_order(dataset):
    # Try to infer band order from dataset object
    if hasattr(dataset, 'band_order'):
        return dataset.band_order
    if hasattr(dataset, 'bands'):
        bands = dataset.bands
        if isinstance(bands, dict) and 'optical' in bands:
            return bands['optical']
        if isinstance(bands, list):
            return bands
    return None


def interpret_biomass(val):
    # Example: convert normalized AGBD to Mg/ha and assign category
    agbd_mgha = val * 98.67 + 66.97
    if agbd_mgha < 10:
        return f"{agbd_mgha:.1f} Mg/ha (Very Low)"
    elif agbd_mgha < 50:
        return f"{agbd_mgha:.1f} Mg/ha (Low)"
    elif agbd_mgha < 100:
        return f"{agbd_mgha:.1f} Mg/ha (Moderate)"
    elif agbd_mgha < 200:
        return f"{agbd_mgha:.1f} Mg/ha (High)"
    elif agbd_mgha < 350:
        return f"{agbd_mgha:.1f} Mg/ha (Very High)"
    else:
        return f"{agbd_mgha:.1f} Mg/ha (Extremely High)"


def _get_overlay_mask(inputs, idx):
    # Try to get a mask/outline for overlay if present in batch
    for k in ['mask', 'outline', 'valid_mask', 'cloud_mask', 'region_mask']:
        if k in inputs:
            mask = inputs[k][idx]
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            return mask
    return None


def _get_best_worst(pred_c, target_c, n=1):
    # Return indices of best (lowest error) and worst (highest error) samples
    err = np.abs(pred_c - target_c)
    best = np.argsort(err)[:n]
    worst = np.argsort(err)[-n:][::-1]
    return best, worst


def _panel_of_panels(figs, titles, layout=(1, 1), figsize=(24, 12)):
    # Combine multiple figures into a single panel
    fig, axs = plt.subplots(*layout, figsize=figsize)
    axs = np.array(axs).reshape(-1)
    for i, (f, t) in enumerate(zip(figs, titles)):
        buf = io.BytesIO()
        f.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        axs[i].imshow(img)
        axs[i].set_title(t)
        axs[i].axis('off')
        plt.close(f)
    plt.tight_layout()
    return fig


def log_agbd_regression_visuals(
    inputs,
    pred,
    target,
    band_order=None,
    wandb_run=None,
    step=None,
    prefix="val",
    max_samples=3,
    dataset=None,
    exp_dir=None  # for marker file
):
    """
    🚀 Boss-level, publication-ready AGBD regression visualizations for any model, any batch.
    """
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    if band_order is None and dataset is not None:
        band_order = auto_band_order(dataset)

    pred_np = to_np(pred)
    target_np = to_np(target)
    B = pred_np.shape[0]
    samples = np.random.choice(B, min(B, max_samples), replace=False)

    # Metrics
    if pred_np.ndim == 3:
        cy, cx = pred_np.shape[1] // 2, pred_np.shape[2] // 2
        pred_c = pred_np[:, cy, cx]
        target_c = target_np[:, cy, cx]
    else:
        pred_c = pred_np.flatten()
        target_c = target_np.flatten()
    mae = mean_absolute_error(target_c, pred_c)
    mse = mean_squared_error(target_c, pred_c)
    rmse = np.sqrt(mse)
    r2 = r2_score(target_c, pred_c)
    mape = np.mean(np.abs((target_c - pred_c) / (target_c + 1e-8))) * 100
    metrics = {
        f"{prefix}/MAE": mae,
        f"{prefix}/RMSE": rmse,
        f"{prefix}/R2": r2,
        f"{prefix}/MAPE": mape,
    }
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)

    # Best/worst samples
    best_idx, worst_idx = _get_best_worst(pred_c, target_c, n=1)
    brag_text = f"Best: GT={target_c[best_idx[0]]:.2f}, Pred={pred_c[best_idx[0]]:.2f}, Error={pred_c[best_idx[0]]-target_c[best_idx[0]]:.2f}\n" \
                f"Worst: GT={target_c[worst_idx[0]]:.2f}, Pred={pred_c[worst_idx[0]]:.2f}, Error={pred_c[worst_idx[0]]-target_c[worst_idx[0]]:.2f}\n" \
                f"MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}, MAPE={mape:.2f}"

    images = []
    figs = []
    titles = []
    for idx in samples:
        fig, axs = plt.subplots(2, 6, figsize=(28, 10))
        # True color or fallback
        if 'optical' in inputs and band_order is not None:
            optical = to_np(inputs['optical'][idx])
            try:
                idx_r = band_order.index('B04')
                idx_g = band_order.index('B03')
                idx_b = band_order.index('B02')
                rgb = np.stack([
                    optical[idx_r],
                    optical[idx_g],
                    optical[idx_b]
                ], axis=-1)
                vmin, vmax = np.percentile(rgb, 1), np.percentile(rgb, 99)
                rgb = np.clip((rgb - vmin) / (vmax - vmin + 1e-8), 0, 1)
                axs[0, 0].imshow(rgb)
                axs[0, 0].set_title('True Color (S2)')
            except Exception as e:
                mean_img = np.mean(optical, axis=0)
                axs[0, 0].imshow(mean_img, cmap='gray')
                axs[0, 0].set_title('Optical Mean')
        elif 'optical' in inputs:
            optical = to_np(inputs['optical'][idx])
            mean_img = np.mean(optical, axis=0)
            axs[0, 0].imshow(mean_img, cmap='gray')
            axs[0, 0].set_title('Optical Mean')
        else:
            axs[0, 0].text(0.5, 0.5, 'No Optical', ha='center', va='center')
        axs[0, 0].axis('off')
        # SAR or fallback
        if 'sar' in inputs:
            sar = to_np(inputs['sar'][idx])
            if sar.shape[0] >= 2:
                sar_img = np.stack([
                    np.clip((sar[0] - sar[0].min()) / (sar[0].ptp() + 1e-8), 0, 1),
                    np.clip((sar[1] - sar[1].min()) / (sar[1].ptp() + 1e-8), 0, 1),
                    np.zeros_like(sar[0])
                ], axis=-1)
                axs[0, 1].imshow(sar_img)
                axs[0, 1].set_title('SAR Dual-pol')
            else:
                axs[0, 1].imshow(sar[0], cmap='gray')
                axs[0, 1].set_title('SAR (single pol)')
        else:
            axs[0, 1].text(0.5, 0.5, 'No SAR', ha='center', va='center')
        axs[0, 1].axis('off')
        # NDVI or fallback
        if 'optical' in inputs and band_order is not None and 'B08' in band_order and 'B04' in band_order:
            optical = to_np(inputs['optical'][idx])
            try:
                nir = optical[band_order.index('B08')]
                red = optical[band_order.index('B04')]
                ndvi = (nir - red) / (nir + red + 1e-8)
                axs[0, 2].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                axs[0, 2].set_title('NDVI')
            except Exception:
                axs[0, 2].hist(optical.flatten(), bins=20, color='gray')
                axs[0, 2].set_title('Optical Histogram')
        elif 'optical' in inputs:
            optical = to_np(inputs['optical'][idx])
            axs[0, 2].hist(optical.flatten(), bins=20, color='gray')
            axs[0, 2].set_title('Optical Histogram')
        else:
            axs[0, 2].text(0.5, 0.5, 'No NDVI', ha='center', va='center')
        axs[0, 2].axis('off')
        # Overlay mask if present
        mask = _get_overlay_mask(inputs, idx)
        if mask is not None:
            axs[0, 3].imshow(mask, cmap='cool', alpha=0.5)
            axs[0, 3].set_title('Overlay Mask')
        else:
            axs[0, 3].text(0.5, 0.5, 'No Mask', ha='center', va='center')
        axs[0, 3].axis('off')
        # Target
        im = axs[1, 0].imshow(target_np[idx], cmap='viridis')
        axs[1, 0].set_title('Target')
        plt.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)
        axs[1, 0].axis('off')
        # Prediction
        im = axs[1, 1].imshow(pred_np[idx], cmap='viridis')
        axs[1, 1].set_title('Prediction')
        plt.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)
        axs[1, 1].axis('off')
        # Error map
        im = axs[1, 2].imshow(np.abs(pred_np[idx] - target_np[idx]), cmap='Reds')
        axs[1, 2].set_title('Abs Error')
        plt.colorbar(im, ax=axs[1, 2], fraction=0.046, pad=0.04)
        axs[1, 2].axis('off')
        # Violin/boxplot of errors
        err = pred_np[idx].flatten() - target_np[idx].flatten()
        if HAS_SEABORN:
            sns.violinplot(y=err, ax=axs[1, 3], color='lightblue')
            axs[1, 3].set_title('Error Violin')
        else:
            axs[1, 3].boxplot(err)
            axs[1, 3].set_title('Error Boxplot')
        axs[1, 3].axhline(np.mean(err), color='red', linestyle='--', label='Mean')
        axs[1, 3].legend()
        axs[1, 3].set_xticks([])
        # Interpretive text
        gt_val = float(target_c[idx])
        pred_val = float(pred_c[idx])
        axs[0, 4].text(0.05, 0.8, f"GT: {interpret_biomass(gt_val)}", fontsize=12, color='blue')
        axs[0, 4].text(0.05, 0.6, f"Pred: {interpret_biomass(pred_val)}", fontsize=12, color='green')
        axs[0, 4].text(0.05, 0.4, f"Error: {pred_val-gt_val:.2f}", fontsize=12, color='red')
        axs[0, 4].set_title('Interpretation')
        axs[0, 4].axis('off')
        # Brag panel
        axs[0, 5].text(0.05, 0.8, brag_text, fontsize=12, color='purple')
        axs[0, 5].set_title('Brag Stats')
        axs[0, 5].axis('off')
        # Overlay central pixel value
        axs[1, 0].text(0.05, 0.95, f"Center: {target_c[idx]:.2f}", color='white', fontsize=10, transform=axs[1, 0].transAxes)
        axs[1, 1].text(0.05, 0.95, f"Center: {pred_c[idx]:.2f}", color='white', fontsize=10, transform=axs[1, 1].transAxes)
        # Outlier highlight
        axs[1, 2].text(0.05, 0.95, f"MaxErr: {np.max(np.abs(err)):.2f}", color='black', fontsize=10, transform=axs[1, 2].transAxes)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        images.append(wandb.Image(Image.open(buf), caption=f"{prefix}_sample_{idx}"))
        figs.append(fig)
        titles.append(f"Sample {idx}")
        plt.close(fig)
    # Scatter plot with regression line
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(target_c, pred_c, alpha=0.6)
    if len(target_c) > 1:
        m, b = np.polyfit(target_c, pred_c, 1)
        ax.plot(target_c, m*target_c + b, color='orange', label='Fit')
        ax.legend()
    ax.plot([target_c.min(), target_c.max()], [target_c.min(), target_c.max()], 'k--')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Prediction')
    ax.set_title('GT vs Prediction')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    scatter_img = wandb.Image(Image.open(buf), caption=f"{prefix}_scatter")
    figs.append(fig)
    titles.append("Scatter")
    plt.close(fig)
    # Error histogram with mean/median
    fig, ax = plt.subplots(figsize=(5, 3))
    err = pred_c - target_c
    ax.hist(err, bins=20, color='red', alpha=0.7)
    ax.axvline(np.mean(err), color='blue', linestyle='--', label='Mean')
    ax.axvline(np.median(err), color='green', linestyle=':', label='Median')
    ax.set_title('Prediction Error Histogram')
    ax.set_xlabel('Pred - GT')
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    hist_img = wandb.Image(Image.open(buf), caption=f"{prefix}_error_hist")
    figs.append(fig)
    titles.append("Error Hist")
    plt.close(fig)
    # Heatmap (binned GT vs Pred)
    if HAS_SEABORN:
        fig, ax = plt.subplots(figsize=(5, 4))
        bins = np.linspace(min(target_c.min(), pred_c.min()), max(target_c.max(), pred_c.max()), 20)
        h, xedges, yedges = np.histogram2d(target_c, pred_c, bins=[bins, bins])
        sns.heatmap(h.T, ax=ax, cmap='Blues', cbar=True, xticklabels=False, yticklabels=False)
        ax.set_xlabel('GT bin')
        ax.set_ylabel('Pred bin')
        ax.set_title('GT vs Pred Heatmap')
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        heatmap_img = wandb.Image(Image.open(buf), caption=f"{prefix}_heatmap")
        figs.append(fig)
        titles.append("Heatmap")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, 'Seaborn not installed', ha='center', va='center')
        ax.set_title('GT vs Pred Heatmap')
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        heatmap_img = wandb.Image(Image.open(buf), caption=f"{prefix}_heatmap")
        figs.append(fig)
        titles.append("Heatmap")
        plt.close(fig)
    # Table
    table = wandb.Table(data=[[int(i), float(target_c[i]), float(pred_c[i]), float(pred_c[i]-target_c[i]), interpret_biomass(target_c[i]), interpret_biomass(pred_c[i])] for i in range(len(target_c))],
                        columns=["Sample", "GT", "Pred", "Error", "GT_Interp", "Pred_Interp"])
    # Batch summary table
    batch_table = wandb.Table(data=[[mae, rmse, r2, mape]], columns=["MAE", "RMSE", "R2", "MAPE"])
    # Panel of panels (all-in-one)
    panel_fig = _panel_of_panels(figs, titles, layout=(2, int(np.ceil(len(figs)/2))), figsize=(32, 16))
    buf = io.BytesIO()
    panel_fig.savefig(buf, format='png')
    buf.seek(0)
    panel_img = wandb.Image(Image.open(buf), caption=f"{prefix}_panel_of_panels")
    plt.close(panel_fig)
    # Log all
    log_dict = {
        f"{prefix}/samples": images,
        f"{prefix}/scatter": scatter_img,
        f"{prefix}/error_hist": hist_img,
        f"{prefix}/heatmap": heatmap_img,
        f"{prefix}/metrics_table": table,
        f"{prefix}/batch_summary": batch_table,
        f"{prefix}/panel_of_panels": panel_img,
        f"{prefix}/brag_text": wandb.Html(f"<pre>{brag_text}</pre>")
    }
    if wandb_run is not None:
        wandb_run.log(log_dict, step=step)
    else:
        wandb.log(log_dict, step=step)
    # Write marker file if exp_dir is given
    if exp_dir is not None:
        try:
            with open(f"{exp_dir}/visualization_done.txt", "a") as f:
                f.write(f"Visualization logged for step {step}\n")
        except Exception as e:
            print(f"[WARN] Could not write visualization marker: {e}")

"""
Quickstart integration (in your evaluator):

from agbd_visualization import log_agbd_regression_visuals

# ... inside your batch loop ...
log_agbd_regression_visuals(
    inputs=batch,         # dict with 'optical', 'sar', etc.
    pred=logits,          # model predictions
    target=target,        # ground truth
    band_order=band_order, # list of band names for optical (or pass dataset=your_dataset)
    wandb_run=wandb,      # wandb run object
    step=batch_idx,       # current step or batch index
    prefix=self.split,    # or "test"
    max_samples=3,        # number of samples to visualize
    dataset=self.val_loader.dataset # for auto band order
)
"""
