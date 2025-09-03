from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterPixelMSE(nn.Module):
    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        self.debug = debug
        self._did_log = False

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, metadata: list[dict]
    ) -> torch.Tensor:
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        elif pred.ndim != 3:
            raise RuntimeError(
                f"pred must be (B,1,H,W) or (B,H,W); got {tuple(pred.shape)}"
            )

        if target.ndim != 1 or target.shape[0] != pred.shape[0]:
            raise RuntimeError(
                f"target must be (B,) scalar labels; got {tuple(target.shape)} for batch {pred.shape[0]}"
            )

        if not isinstance(metadata, list) or len(metadata) != pred.shape[0]:
            raise RuntimeError("metadata must be a list with one dict per sample")

        B, H, W = pred.shape
        pred_center = torch.empty(B, device=pred.device, dtype=pred.dtype)

        for i in range(B):
            cy, cx = metadata[i]["center_pixel_yx"]
            if not (0 <= cy < H and 0 <= cx < W):
                raise RuntimeError(
                    f"center_pixel_yx out of bounds for sample {i}: {(cy, cx)} not in [0,{H})x[0,{W})"
                )
            pred_center[i] = pred[i, cy, cx]

        if self.debug and not self._did_log:
            i = 0
            cy, cx = metadata[i]["center_pixel_yx"]
            pc = float(pred_center[i].detach().cpu())
            tc = float(target[i].detach().cpu())
            print(
                f"[CenterPixelMSE] pred={tuple(pred.shape)} target={tuple(target.shape)} "
                f"center@{(cy,cx)} pred_center={pc:.3f} target={tc:.3f}"
            )
            self._did_log = True

        return F.mse_loss(pred_center, target, reduction="mean")
