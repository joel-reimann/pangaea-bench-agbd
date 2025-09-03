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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.training_metrics = {
            name: RunningAverageMeter(length=100) for name in ["MSE"]
        }

        self.best_metric = float("inf")

        import operator

        self.best_metric_comp = operator.lt
        self.logger = logging.getLogger()

    def _extract_center_values(
        self, preds: torch.Tensor, target: torch.Tensor, metadata: List[Dict[str, Any]]
    ):

        pred_centers = []
        target_centers = []
        for i, meta in enumerate(metadata):
            y, x = meta.get(
                "center_pixel_yx", (preds.shape[-2] // 2, preds.shape[-1] // 2)
            )
            pred_centers.append(preds[i, y, x])
            target_centers.append(target[i, y, x])
        pred_center = torch.stack(pred_centers)
        target_center = torch.stack(target_centers)
        return pred_center, target_center

    def train_one_epoch(self, epoch: int) -> None:

        self.model.train()

        end_time = time.time()
        self.train_loader.sampler.set_epoch(epoch)

        panel_sequence: List[Any] = []
        for batch_idx, batch in enumerate(self.train_loader):
            image_dict, target = batch["image"], batch["target"]
            metadata = batch["metadata"]

            image = {k: v.to(self.device) for k, v in image_dict.items()}
            target = target.to(self.device)

            self.training_stats["data_time"].update(time.time() - end_time)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.enable_mixed_precision,
                dtype=self.precision,
            ):
                logits = self.model(image, output_shape=target.shape[-2:])

                loss = self.criterion(logits, target, metadata)

            self.optimizer.zero_grad()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Rank {self.rank} got infinite/NaN loss at batch {batch_idx} of epoch {epoch}!"
                )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.training_stats["loss"].update(loss.item())

            mse = loss.detach()
            self.training_metrics["MSE"].update(mse.item())

            if batch_idx == 0:

                if "optical" in image_dict:
                    opt = image_dict["optical"][0]

                    if opt.ndim == 4:
                        opt = opt[:, 0]
                    stats = []
                    for c in range(opt.shape[0]):
                        band = opt[c].cpu().numpy()
                        stats.append(
                            (float(band.min()), float(band.mean()), float(band.max()))
                        )
                    self.logger.debug(
                        f"DEBUG[Preprocessor] Epoch {epoch}, batch {batch_idx}: optical stats per channel (min, mean, max): {stats}"
                    )

            if self.use_wandb and self.rank == 0:
                if batch_idx % 100 == 0:

                    vis_image = {k: v[0].detach().cpu() for k, v in image_dict.items()}
                    raw_pack = metadata[0].get("_agbd_raw_image", None)
                    from pangaea.utils.agbd_logging import create_agbd_panel

                    panel = create_agbd_panel(
                        vis_image,
                        logits.detach().cpu()[0].squeeze(),
                        target.detach().cpu()[0],
                        metadata[0],
                        raw_pack,
                    )
                    panel_sequence.append(panel)

            if (batch_idx + 1) % self.log_interval == 0:
                self.log(batch_idx + 1, epoch)

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

            self.training_stats["batch_time"].update(time.time() - end_time)
            end_time = time.time()

        if self.use_wandb and self.rank == 0 and panel_sequence:
            try:
                for i, panel in enumerate(panel_sequence):

                    batch_name = "first" if i == 0 else "last"
                    panel_name = f"agbd_train_epoch_{epoch}_{batch_name}_batch"
                    self.wandb.log({panel_name: panel})
            except Exception:
                pass
        return

    @torch.no_grad()
    def compute_logging_metrics(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> None:
        return
