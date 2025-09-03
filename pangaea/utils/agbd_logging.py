from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import torch

_logged_once = False


def create_agbd_panel(
    image_dict: Dict[str, torch.Tensor],
    pred_logits: torch.Tensor,
    target_scalar: torch.Tensor | float,
    metadata: Dict[str, Any],
    raw_pack: Dict[str, torch.Tensor] | None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    import PIL.Image as Image
    import wandb

    def to_rgb_panel(
        t: torch.Tensor | None,
    ) -> Tuple[np.ndarray | None, Tuple[int, int]]:
        if t is None:
            return None, (0, 0)
        x = t[:, 0] if t.ndim == 4 else t
        if x.ndim != 3:
            return None, (0, 0)
        C, H, W = x.shape
        if C < 3:
            return None, (H, W)
        rgb = x[:3].detach().cpu().numpy().transpose(1, 2, 0)
        out = rgb.copy()
        for i in range(3):
            b = rgb[:, :, i]
            p1, p99 = np.percentile(b, 1), np.percentile(b, 99)
            out[:, :, i] = 0 if p99 <= p1 else np.clip((b - p1) / (p99 - p1), 0, 1)
        return out, (H, W)

    pred_map = pred_logits.squeeze()
    if pred_map.ndim != 2:
        raise RuntimeError(f"pred_logits must be (H,W), got {tuple(pred_logits.shape)}")
    H, W = pred_map.shape
    cy, cx = metadata["center_pixel_yx"]
    y_true = float(
        target_scalar
        if isinstance(target_scalar, (float, int))
        else target_scalar.item()
    )
    nz = pred_map[pred_map > 0]
    if isinstance(nz, torch.Tensor):
        nz = nz.detach().cpu().numpy()
    vmax = float(np.percentile(nz, 99)) if nz.size > 0 else float(pred_map.max().item())
    vmax = max(vmax, 50.0)
    vmin = 0.0

    global _logged_once
    if not _logged_once:
        print(
            f"[AGBD-LOG] pred_map={(H,W)} center={(cy,cx)} target={y_true:.3f} vmax={vmax:.1f}"
        )
        _logged_once = True

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    if raw_pack is not None and isinstance(raw_pack, dict) and "optical" in raw_pack:
        rgb1, size1 = to_rgb_panel(raw_pack["optical"])
        if rgb1 is not None:
            axes[0].imshow(rgb1)
            axes[0].set_title(
                f"1. Original AGBD RGB ({size1[0]}×{size1[1]})", fontsize=12
            )
        else:
            axes[0].text(
                0.5,
                0.5,
                "Original RGB unavailable",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_title("1. Original AGBD RGB", fontsize=12)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No raw snapshot",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title("1. Original AGBD RGB", fontsize=12)
    axes[0].axis("off")

    rgb2, size2 = to_rgb_panel(image_dict.get("optical", None))
    if rgb2 is not None:
        axes[1].imshow(rgb2)
        axes[1].set_title(f"2. Preprocessed RGB ({size2[0]}×{size2[1]})", fontsize=12)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Preprocessed RGB unavailable",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("2. Preprocessed RGB", fontsize=12)
    axes[1].axis("off")

    im = axes[2].imshow(
        pred_map.detach().cpu().numpy(), cmap="Greens", vmin=vmin, vmax=vmax
    )
    pred_center = float(pred_map[cy, cx].item())
    err = pred_center - y_true
    axes[2].scatter([cx], [cy], c="red", s=10)
    axes[2].set_title(
        f"3. Biomass Prediction ({H}×{W})\nPred: {pred_center:.1f} | GT: {y_true:.1f} | Error: {err:+.1f} Mg/ha",
        fontsize=12,
    )
    plt.colorbar(im, ax=axes[2], label="Biomass (Mg/ha)", fraction=0.046)
    axes[2].axis("off")

    fig.suptitle(
        "AGBD Pipeline: Original → Preprocessed → Prediction",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    panel = wandb.Image(Image.open(buf))
    plt.close(fig)
    return panel
