from __future__ import annotations

from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AGBDDownsampleMSE(nn.Module):
    def __init__(
        self,
        mode: str = "bicubic",
        window_radius: Optional[int] = 0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if mode not in {"bicubic", "bilinear", "avgpool"}:
            raise ValueError(f"mode must be one of bicubic|bilinear|avgpool, got {mode}")
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"reduction must be mean|sum, got {reduction}")
        self.mode = mode
        self.window_radius = window_radius
        self.reduction = reduction

    def _to_25(self, pred: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 4:
            if pred.shape[1] == 1:
                x = pred
            else:
                x = pred[:, :1]
        elif pred.ndim == 3:
            x = pred.unsqueeze(1)
        else:
            raise RuntimeError(f"pred must be (B,H,W) or (B,1,H,W); got {tuple(pred.shape)}")
        if self.mode == "avgpool":
            y = F.adaptive_avg_pool2d(x, (25, 25))
        else:
            y = F.interpolate(x, size=(25, 25), mode=self.mode, align_corners=False)
        return y.squeeze(1)

    @staticmethod
    def _map_center_to_25(cy: int, cx: int, H: int, W: int) -> tuple[int, int]:
        ny = int(round((cy + 0.5) * 25.0 / float(H) - 0.5))
        nx = int(round((cx + 0.5) * 25.0 / float(W) - 0.5))
        ny = max(0, min(24, ny))
        nx = max(0, min(24, nx))
        return ny, nx

    def forward(
        self,
        pred: torch.Tensor,
        target_scalar: torch.Tensor,
        metadata: List[Dict[str, Any]] | Dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if target_scalar.ndim == 0:
            target_scalar = target_scalar.view(1)
        if target_scalar.ndim != 1:
            raise RuntimeError(f"target_scalar must be (B,), got {tuple(target_scalar.shape)}")
        B = target_scalar.shape[0]
        if pred.ndim == 4:
            H, W = pred.shape[-2:]
        elif pred.ndim == 3:
            H, W = pred.shape[-2:]
        else:
            raise RuntimeError(f"pred must be (B,H,W) or (B,1,H,W); got {tuple(pred.shape)}")

        p25 = self._to_25(pred)

        if self.window_radius is None:
            tgt25 = target_scalar.view(B, 1, 1).expand(B, 25, 25)
            diff = p25 - tgt25
            loss_map = diff * diff
            loss = loss_map.mean() if self.reduction == "mean" else loss_map.sum()
            return loss

        if isinstance(metadata, dict):
            metadata_list = [metadata] * B
        else:
            metadata_list = metadata
        if metadata_list is None or len(metadata_list) != B:
            raise RuntimeError("metadata must be a list of length B or a dict")

        cycx_25 = []
        for i in range(B):
            cy, cx = metadata_list[i].get("center_pixel_yx", (H // 2, W // 2))
            cy25, cx25 = self._map_center_to_25(int(cy), int(cx), H, W)
            cycx_25.append((cy25, cx25))

        if self.window_radius == 0:
            vals = []
            for i, (cy25, cx25) in enumerate(cycx_25):
                vals.append(p25[i, cy25, cx25])
            pred_vec = torch.stack(vals, dim=0)
            diff = pred_vec - target_scalar
            loss_elem = diff * diff
            loss = loss_elem.mean() if self.reduction == "mean" else loss_elem.sum()
            return loss

        r = int(self.window_radius)
        vals = []
        for i, (cy25, cx25) in enumerate(cycx_25):
            y0 = max(0, cy25 - r)
            y1 = min(25, cy25 + r + 1)
            x0 = max(0, cx25 - r)
            x1 = min(25, cx25 + r + 1)
            vals.append(p25[i, y0:y1, x0:x1].mean())
        pred_vec = torch.stack(vals, dim=0)
        diff = pred_vec - target_scalar
        loss_elem = diff * diff
        loss = loss_elem.mean() if self.reduction == "mean" else loss_elem.sum()
        return loss
