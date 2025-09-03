from __future__ import annotations

import os
import time
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

import wandb


from pangaea.engine.evaluator import RegEvaluator


class AGBDEvaluator(RegEvaluator):
    def _downsample_logits(
        self, preds: torch.Tensor, out_hw: Tuple[int, int]
    ) -> torch.Tensor:
        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        if preds.ndim != 3:
            raise RuntimeError(
                f"preds must be (B,1,H,W) or (B,H,W); got {tuple(preds.shape)}"
            )
        mode = getattr(self, "downsample_mode", "avg")
        if mode == "avg":
            x = F.adaptive_avg_pool2d(preds.unsqueeze(1), output_size=out_hw).squeeze(1)
        elif mode in ("nearest", "bilinear", "bicubic", "area"):
            align = False if mode in ("nearest", "area") else True
            x = F.interpolate(
                preds.unsqueeze(1), size=out_hw, mode=mode, align_corners=align
            ).squeeze(1)
        else:
            raise ValueError(f"Unsupported downsample_mode='{mode}'")
        return x

    def _center_from_meta_or_size(
        self, meta: Dict[str, Any], out_hw: Tuple[int, int]
    ) -> Tuple[int, int]:
        if "_orig_hw" in meta:
            oh, ow = meta["_orig_hw"]
            return oh // 2, ow // 2
        h, w = out_hw
        return h // 2, w // 2

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        model_name: str,
        model_ckpt_path: str | None = None,
    ) -> Tuple[Dict[str, float], float]:
        t0 = time.time()

        if model_ckpt_path is not None:
            state = torch.load(
                model_ckpt_path, map_location=self.device, weights_only=False
            )
            m = getattr(model, "module", model)
            m.load_state_dict(
                state["model"]
                if isinstance(state, dict) and "model" in state
                else state
            )
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            self.logger.info(f"[AGBDEvaluator] loaded {model_name}")

        model.eval()

        sqerr = torch.zeros(1, device=self.device)
        n = torch.zeros(1, device=self.device)
        first_logged = False

        tag = f"Evaluating {model_name} on {self.split}"
        for bidx, batch in enumerate(tqdm(self.val_loader, desc=tag)):
            image_dict = {k: v.to(self.device) for k, v in batch["image"].items()}
            target_scalar = batch["target"].to(self.device)
            metadata: List[Dict[str, Any]] = batch["metadata"]

            logits_full = model(image_dict)
            logits_25 = self._downsample_logits(logits_full, out_hw=(25, 25))

            B, H25, W25 = logits_25.shape
            pred_center = torch.empty(B, device=self.device, dtype=logits_25.dtype)
            for i in range(B):
                cy, cx = self._center_from_meta_or_size(metadata[i], (H25, W25))
                pred_center[i] = logits_25[i, cy, cx]

            if not first_logged:
                cy_dbg, cx_dbg = self._center_from_meta_or_size(metadata[0], (H25, W25))
                pc = float(pred_center[0].detach().cpu())
                tc = float(target_scalar[0].detach().cpu())
                shp = tuple(logits_full.shape)
                self.logger.info(
                    f"[AGBDEval] logits={shp} down->(25,25) center@{(cy_dbg,cx_dbg)} pred_center={pc:.3f} target={tc:.3f}"
                )
                first_logged = True

            diff = pred_center - target_scalar
            sqerr += torch.sum(diff * diff)
            n += pred_center.numel()

            if (
                self.use_wandb
                and self.rank == 0
                and (bidx == 0 or bidx == len(self.val_loader) - 1)
            ):
                from pangaea.utils.agbd_logging import create_agbd_panel

                vis_img = {k: v[0].detach().cpu() for k, v in batch["image"].items()}
                raw_pack = metadata[0].get("_agbd_raw_image", None)
                pred_map_vis = (
                    logits_full.detach().cpu().squeeze(1)
                    if logits_full.ndim == 4
                    else logits_full.detach().cpu()
                )[0]
                panel = create_agbd_panel(
                    vis_img,
                    pred_map_vis,
                    float(target_scalar[0].detach().cpu()),
                    metadata[0],
                    raw_pack,
                )
                import wandb

                suffix = "first" if bidx == 0 else "last"
                wandb.log({f"agbd_{self.split}_panel_{suffix}": panel})

        if torch.distributed.is_initialized():
            import torch.distributed as dist

            dist.all_reduce(sqerr, op=dist.ReduceOp.SUM)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)

        mse = (sqerr / n).item()
        rmse = float(torch.sqrt(sqerr / n).item())
        metrics = {"MSE": mse, "RMSE": rmse}

        self.log_metrics(metrics)
        return metrics, time.time() - t0

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self.logger.info(
            f"[{self.split}] Centre-pixel MSE={metrics['MSE']:.3f} RMSE={metrics['RMSE']:.3f}"
        )
        if self.use_wandb and self.rank == 0:

            wandb.log(
                {
                    f"agbd_{self.split}_MSE": metrics["MSE"],
                    f"agbd_{self.split}_RMSE": metrics["RMSE"],
                }
            )
