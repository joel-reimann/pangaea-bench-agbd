from __future__ import annotations
import torch
import torch.nn as nn


class GaussianWeightedMSE(nn.Module):
    def __init__(self, size: int, sigma: float, device: str = "cuda") -> None:
        super().__init__()
        self.size = size
        self.sigma = sigma

        x = torch.arange(0, size, dtype=torch.float32, device=device)
        y = torch.arange(0, size, dtype=torch.float32, device=device)
        y, x = torch.meshgrid(y, x, indexing="ij")

        center = size // 2
        gaussian = torch.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))

        self.register_buffer("weights", gaussian / gaussian.sum())

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, metadata: list[dict]
    ) -> torch.Tensor:

        if pred.ndim == 4:
            pred = pred.squeeze(1)

        target_map = target.view(-1, 1, 1).expand_as(pred)

        squared_error = (pred - target_map) ** 2

        weighted_squared_error = squared_error * self.weights

        return weighted_squared_error.sum(dim=(1, 2)).mean()
