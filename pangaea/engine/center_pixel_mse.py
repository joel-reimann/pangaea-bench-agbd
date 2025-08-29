"""
Custom regression loss for the AGBD dataset.

This module defines a simple mean squared error (MSE) loss that only
considers the value predicted at the centre pixel of each patch.  The
AGBD dataset provides supervision only for the central pixel in each
25×25 input patch, while the decoders in PANGAEA typically predict a
dense map.  Using this loss ensures that all pixels outside the
centre are ignored during optimisation.  To use this criterion,
specify ``_target_: pangaea.engine.center_pixel_mse.CenterPixelMSE`` in
your training configuration.

The loss operates on tensors of shape ``(B, H, W)`` or
``(B, 1, H, W)`` and automatically extracts the centre pixel.  It
works with batched inputs on both CPU and GPU.  When used in a
distributed setting, averaging across processes should be done by the
trainer or evaluator (see ``AGBDTrainer`` and ``AGBDEvaluator``).

Example:

    >>> criterion = CenterPixelMSE()
    >>> pred = torch.randn(4, 1, 25, 25)
    >>> tgt = torch.randn(4, 25, 25)
    >>> loss = criterion(pred, tgt)
    >>> loss.backward()

"""

from __future__ import annotations

import torch
import torch.nn as nn


class CenterPixelMSE(nn.Module):
    """Mean squared error on the centre pixel.

    This loss expects predictions of shape ``(B, *, H, W)`` and targets of
    shape ``(B, H, W)``.  The additional channel dimension of length 1 is
    optional for predictions.  During the forward pass, it extracts
    the centre pixel from both predictions and targets and computes
    their mean squared error.  If either input does not have at least
    two spatial dimensions, a ``ValueError`` is raised.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, metadata=None) -> torch.Tensor:
        """Compute the centre‑pixel mean squared error using optional metadata.

        Args:
            pred: predicted regression map of shape ``(B, ..., H, W)``.
            target: target regression map of shape ``(B, H, W)``.
            metadata: optional list of dictionaries (one per sample) containing
                ``center_pixel_yx`` coordinates.  If provided, these
                coordinates are used instead of the geometric centre.

        Returns:
            A scalar tensor representing the mean squared error between
            the predicted and true centre pixel values across the batch.

        Raises:
            ValueError: if the spatial dimensions are missing or the batch
                dimensions are incompatible.
        """
        # Ensure we have at least batch and two spatial dims
        if pred.ndim < 3 or target.ndim < 3:
            raise ValueError(
                f"Expected pred and target to have at least 3 dims (B,H,W), got {pred.shape} and {target.shape}"
            )
        # Squeeze optional channel dimension
        if pred.ndim == 4:
            if pred.shape[1] == 1:
                pred = pred.squeeze(1)
            else:
                pred = pred[:, 0]
        # Check shapes
        if pred.shape[0] != target.shape[0] or pred.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape}, target {target.shape}"
            )
        b, h, w = pred.shape
        # Use metadata centre coordinates if available
        if metadata is not None:
            pred_centers = []
            target_centers = []
            for i in range(b):
                meta = metadata[i] if isinstance(metadata, list) else metadata
                cy, cx = meta.get('center_pixel_yx', (h // 2, w // 2))
                pred_centers.append(pred[i, cy, cx])
                target_centers.append(target[i, cy, cx])
            pred_center = torch.stack(pred_centers)
            target_center = torch.stack(target_centers)
        else:
            # Geometric centre
            cy = h // 2
            cx = w // 2
            pred_center = pred[:, cy, cx]
            target_center = target[:, cy, cx]
        return torch.mean((pred_center - target_center) ** 2)
