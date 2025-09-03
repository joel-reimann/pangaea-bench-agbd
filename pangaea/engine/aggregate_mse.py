import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregatedMSE(nn.Module):
    def __init__(self, downsample_mode: str = "avg") -> None:
        super().__init__()
        self.downsample_mode = downsample_mode

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, metadata: list[dict]
    ) -> torch.Tensor:

        if pred.ndim == 4:
            pred = pred.squeeze(1)

        pred_aggregated = F.adaptive_avg_pool2d(pred.unsqueeze(1), (1, 1)).squeeze()

        if pred_aggregated.ndim == 0:
            pred_aggregated = pred_aggregated.unsqueeze(0)

        return F.mse_loss(pred_aggregated, target)
