"""
Custom evaluator for the AGBD regression task.

The default ``RegEvaluator`` in PANGAEA computes the mean squared error (MSE)
over the entire output map.  For the AGBD dataset only the centre pixel
in each patch is supervised, and patches are upsampled during
preprocessing.  This evaluator computes MSE and RMSE on the centre
pixel after resizing, using the coordinates stored in the metadata.
It also logs visualisations at the beginning and end of evaluation to
Weights and Biases (wandb) when enabled, showing the raw patch,
preprocessed input and predicted biomass map.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pangaea.engine.evaluator import RegEvaluator


class AGBDEvaluator(RegEvaluator):
    """Evaluator computing centre‑pixel regression metrics with visualisation."""

    @torch.no_grad()
    def _extract_center_values(self, preds: torch.Tensor, target: torch.Tensor, metadata: List[Dict[str, Any]]):
        """Helper to extract centre pixel predictions and targets given metadata.

        Args:
            preds: tensor of shape (B, H, W) or (B,1,H,W).
            target: tensor of shape (B, H, W).
            metadata: list of per‑sample dictionaries containing centre coordinates.

        Returns:
            Tuple of two tensors (pred_center, target_center) each of shape (B,).
        """
        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        pred_centers = []
        target_centers = []
        for i, meta in enumerate(metadata):
            y, x = meta.get('center_pixel_yx', (preds.shape[-2] // 2, preds.shape[-1] // 2))
            pred_centers.append(preds[i, y, x])
            target_centers.append(target[i, y, x])
        pred_centers = torch.stack(pred_centers)
        target_centers = torch.stack(target_centers)
        return pred_centers, target_centers

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        model_name: str,
        model_ckpt_path: str | None = None,
    ) -> Tuple[Dict[str, float], float]:
        """Evaluate the model on the validation set with centre‑pixel supervision.

        Overrides ``RegEvaluator.evaluate`` to compute MSE on the
        centre pixel and to log visualisations to wandb.

        Args:
            model: the model to evaluate (wrapped in DDP).
            model_name: human readable name for logging.
            model_ckpt_path: optional path to a checkpoint to load before evaluation.

        Returns:
            A tuple (metrics, used_time) where metrics is a dictionary with
            keys ``"MSE"`` and ``"RMSE"`` and used_time is the wall
            time in seconds spent evaluating.
        """
        t = time.time()
        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)
            model_name = os.path.basename(model_ckpt_path).split('.')[0]
            # ``model`` may be wrapped in DistributedDataParallel
            if 'model' in model_dict:
                model.module.load_state_dict(model_dict['model'])
            else:
                model.module.load_state_dict(model_dict)
            self.logger.info(f"Loaded model from {model_ckpt_path} for evaluation")

        model.eval()
        tag = f'Evaluating {model_name} on {self.split} set'
        # Accumulate squared error and count across batches
        sqerr = torch.zeros(1, device=self.device)
        n_pixels = torch.zeros(1, device=self.device)
        # Collect panels for slider visualisation
        panel_sequence = []
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=tag)):
            image_dict, target = batch['image'], batch['target']
            metadata = batch['metadata']
            image = {k: v.to(self.device) for k, v in image_dict.items()}
            target = target.to(self.device)
            # Inference
            if self.inference_mode == 'sliding':
                input_size = model.module.encoder.input_size
                logits = self.sliding_inference(model, image, input_size, output_shape=target.shape[-2:],
                                                max_batch=self.sliding_inference_batch)
                logits = logits.squeeze(dim=1)
            elif self.inference_mode == 'whole':
                logits = model(image, output_shape=target.shape[-2:]).squeeze(dim=1)
            else:
                raise NotImplementedError(f"Inference mode {self.inference_mode} is not implemented.")
            # Extract centre pixels
            pred_center, tgt_center = self._extract_center_values(logits, target, metadata)
            # Sum squared error
            sqerr += torch.sum((pred_center - tgt_center) ** 2)
            n_pixels += pred_center.numel()
            # Debug: log statistics of the pre‑processed optical tensor on first batch
            if batch_idx == 0:
                if 'optical' in image_dict:
                    opt = image_dict['optical'][0]
                    if opt.ndim == 4:
                        opt = opt[:, 0]
                    stats = []
                    for c in range(opt.shape[0]):
                        band = opt[c].cpu().numpy()
                        stats.append((float(band.min()), float(band.mean()), float(band.max())))
                    self.logger.debug(
                        f"DEBUG[Evaluator] {self.split} batch {batch_idx}: optical stats per channel (min, mean, max): {stats}"
                    )
            # Collect panels for first and last batch (rank 0 only) using shared utility
            if self.use_wandb and self.rank == 0:
                if batch_idx % 100 == 0:
                    vis_image = {k: v[0].detach().cpu() for k, v in image_dict.items()}
                    raw_pack = metadata[0].get('_agbd_raw_image', None)
                    pred_map = logits.detach().cpu()[0]
                    target_map = target.detach().cpu()[0]
                    from pangaea.utils.agbd_logging import create_agbd_panel
                    panel = create_agbd_panel(vis_image, pred_map, target_map, metadata[0], raw_pack)
                    panel_sequence.append(panel)
        # Reduce across processes
        torch.distributed.all_reduce(sqerr, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(n_pixels, op=torch.distributed.ReduceOp.SUM)
        mse = sqerr / n_pixels
        rmse = torch.sqrt(mse)
        metrics = {"MSE": mse.item(), "RMSE": rmse.item()}
        # Log metrics (base and custom) and panels
        self.log_metrics(metrics)
        if self.use_wandb and self.rank == 0 and panel_sequence:
            try:
                import wandb
                wandb.log({f"agbd_{self.split}_panels": panel_sequence})
            except Exception:
                pass
        used_time = time.time() - t
        return metrics, used_time

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to the logger and optionally to wandb."""
        # Log both base class format (for compatibility) and center-pixel format to console
        # Base class format (but without wandb logging to avoid duplicates)
        header_base = f"[{self.split}] ------- MSE and RMSE --------\n"
        mse_line_base = f"[{self.split}]-------------------\n" + 'MSE \t{:>7}'.format(f"{metrics['MSE']:.3f}") + '\n'
        rmse_line_base = f"[{self.split}]-------------------\n" + 'RMSE \t{:>7}'.format(f"{metrics['RMSE']:.3f}")
        self.logger.info(header_base + mse_line_base + rmse_line_base)
        
        # Custom header for centre‑pixel metrics to the console
        header = f"[{self.split}] ------- Centre Pixel MSE and RMSE --------\n"
        mse_line = f"[{self.split}]-------------------\n" + 'MSE \t{:>7}'.format(f"{metrics['MSE']:.3f}") + '\n'
        rmse_line = f"[{self.split}]-------------------\n" + 'RMSE \t{:>7}'.format(f"{metrics['RMSE']:.3f}")
        self.logger.info(header + mse_line + rmse_line)
        
        # Log to WandB with AGBD-specific names only (no duplicate base class names)
        if self.use_wandb and self.rank == 0:
            import wandb
            wandb.log({f"agbd_{self.split}_MSE": metrics["MSE"], f"agbd_{self.split}_RMSE": metrics["RMSE"]})
