"""
Custom trainer for the AGBD regression task with centre‑pixel supervision.

The standard ``RegTrainer`` in PANGAEA computes regression losses and
metrics over the entire output map.  For the AGBD dataset only the
central pixel of each patch carries supervision; the surrounding
context is provided to the network for spatial context but should not
contribute to the loss.  Furthermore, most encoders in PANGAEA
operate on fixed input sizes larger than 25×25, so patches must be
upsampled.  During upsampling the location of the centre pixel
changes; this trainer uses metadata provided by the custom
preprocessor ``ResizeToEncoderWithCenter`` to identify the correct
pixel and compute the mean squared error (MSE) accordingly.

This trainer also produces comprehensive visualisations of the
learning process.  At the beginning and end of each epoch, it logs
three panels to Weights and Biases (wandb) when enabled:

  1. **Original AGBD RGB:** a colour composite of the raw Sentinel‑2
     patch (before normalisation and resizing).  If no raw data is
     available, a placeholder is shown.
  2. **Preprocessed RGB:** the input actually fed to the model after
     band filtering, resizing and normalisation.
  3. **Prediction map:** the model’s output regression map, with the
     centre pixel prediction, ground truth and absolute error annotated.

Only rank 0 in a distributed setting performs logging to avoid
duplicate images.  Metrics (centre‑pixel MSE) are aggregated across
processes via the ``RunningAverageMeter`` provided by PANGAEA.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List
import logging

import numpy as np
import torch
import torch.nn.functional as F

from pangaea.engine.trainer import RegTrainer
from pangaea.utils.logger import RunningAverageMeter


class AGBDTrainer(RegTrainer):
    """Trainer for the AGBD regression task with centre‑pixel supervision and visualisation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Override training metrics: track MSE on centre pixel only
        self.training_metrics = {
            name: RunningAverageMeter(length=100) for name in ["MSE"]
        }
        # For regression lower is better
        self.best_metric = float("inf")
        # Use the Python less-than operator for comparing float metrics.
        # Do not use ``torch.lt`` here because it expects tensors; the
        # PANGAEA trainer compares floats when saving the best checkpoint.
        import operator
        self.best_metric_comp = operator.lt
        self.logger = logging.getLogger()

    def _extract_center_values(self, preds: torch.Tensor, target: torch.Tensor, metadata: List[Dict[str, Any]]):
        """Extract centre pixel predictions and targets given metadata.

        Args:
            preds: tensor of shape (B, H, W) containing model outputs.
            target: tensor of shape (B, H, W) containing ground truth maps.
            metadata: list of dictionaries for each sample with key
                ``center_pixel_yx`` storing (y, x) coordinates.

        Returns:
            tuple of two tensors ``(pred_center, target_center)`` each of shape (B,).
        """
        pred_centers = []
        target_centers = []
        for i, meta in enumerate(metadata):
            y, x = meta.get('center_pixel_yx', (preds.shape[-2] // 2, preds.shape[-1] // 2))
            pred_centers.append(preds[i, y, x])
            target_centers.append(target[i, y, x])
        pred_center = torch.stack(pred_centers)
        target_center = torch.stack(target_centers)
        return pred_center, target_center


    def train_one_epoch(self, epoch: int) -> None:
        """Train model for one epoch with centre‑pixel supervision and visualisation.

        Overrides the base implementation to compute loss on the centre
        pixel and log comprehensive panels at the start and end of
        each epoch.
        """
        self.model.train()
        # Reset timers
        end_time = time.time()
        self.train_loader.sampler.set_epoch(epoch)
        # Collect panels for slider visualisation across the epoch.  We
        # accumulate only the first and last batch panels; these will
        # provide a concise summary of the epoch when viewed as a
        # WandB slider.
        panel_sequence: List[Any] = []
        for batch_idx, batch in enumerate(self.train_loader):
            image_dict, target = batch['image'], batch['target']
            metadata = batch['metadata']
            # Move data to device
            image = {k: v.to(self.device) for k, v in image_dict.items()}
            target = target.to(self.device)
            # Measure data loading time
            self.training_stats['data_time'].update(time.time() - end_time)
            # Forward and loss computation with autocast for mixed precision
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.enable_mixed_precision,
                dtype=self.precision,
            ):
                logits = self.model(image, output_shape=target.shape[-2:])
                # Compute centre‑pixel loss via the registered criterion.  The
                # criterion accepts the metadata for centre coordinates.
                loss = self.criterion(logits, target, metadata)
            # Backpropagation
            self.optimizer.zero_grad()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Rank {self.rank} got infinite/NaN loss at batch {batch_idx} of epoch {epoch}!"
                )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Update training stats
            self.training_stats['loss'].update(loss.item())
            # Compute logging metrics on centre pixel (again) and update meter
            mse = loss.detach()
            self.training_metrics['MSE'].update(mse.item())
            # Debug: log stats for the first batch of each epoch
            if batch_idx == 0:
                # Access raw preprocessed optical tensor (before conversion to device)
                if 'optical' in image_dict:
                    opt = image_dict['optical'][0]  # (C,T,H,W) or (C,H,W)
                    # Collapse temporal dimension if present
                    if opt.ndim == 4:
                        opt = opt[:, 0]
                    stats = []
                    for c in range(opt.shape[0]):
                        band = opt[c].cpu().numpy()
                        stats.append((float(band.min()), float(band.mean()), float(band.max())))
                    self.logger.debug(f"DEBUG[Preprocessor] Epoch {epoch}, batch {batch_idx}: optical stats per channel (min, mean, max): {stats}")
            # Visualisation: collect first and last batch panels
            if self.use_wandb and self.rank == 0:
                if batch_idx % 100 == 0:
                    # Extract first sample for visualisation
                    vis_image = {k: v[0].detach().cpu() for k, v in image_dict.items()}
                    raw_pack = metadata[0].get('_agbd_raw_image', None)
                    from pangaea.utils.agbd_logging import create_agbd_panel
                    panel = create_agbd_panel(
                        vis_image,
                        logits.detach().cpu()[0].squeeze(),
                        target.detach().cpu()[0],
                        metadata[0],
                        raw_pack,
                    )
                    panel_sequence.append(panel)
            # Logging to wandb: metrics and learning rate  
            if (batch_idx + 1) % self.log_interval == 0:
                self.log(batch_idx + 1, epoch)
                
            # Log metrics to wandb with AGBD-specific names
            if self.use_wandb and self.rank == 0:
                self.wandb.log(
                    {
                        "agbd_train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        **{
                            f"agbd_train_{k}": v.avg
                            for k, v in self.training_metrics.items()
                        },
                    },
                    step=epoch * len(self.train_loader) + batch_idx,
                )
            self.lr_scheduler.step()
            # Measure batch time
            self.training_stats['batch_time'].update(time.time() - end_time)
            end_time = time.time()
        # End of epoch: log individual panels with descriptive names for slider navigation
        if self.use_wandb and self.rank == 0 and panel_sequence:
            try:
                for i, panel in enumerate(panel_sequence):
                    # Create descriptive names: epoch_X_batch_Y_sample_Z
                    batch_name = "first" if i == 0 else "last"
                    panel_name = f"agbd_train_epoch_{epoch}_{batch_name}_batch"
                    self.wandb.log({panel_name: panel})
            except Exception:
                pass
        return

    @torch.no_grad()
    def compute_logging_metrics(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Override base method to do nothing.

        Metrics are updated during ``train_one_epoch`` when computing
        centre‑pixel MSE, so this method is left empty to prevent
        duplicate updates.
        """
        return
