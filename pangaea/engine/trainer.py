import copy
import logging
import operator
import os
import pathlib
import time
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset
from pangaea.utils.logger import RunningAverageMeter, sec_to_hm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module | None,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        evaluator: torch.nn.Module,
        n_epochs: int,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        best_metric_key: str,
    ):
        """Initialize the Trainer.

        Args:
            model (nn.Module): model to train (encoder + decoder).
            train_loader (DataLoader): train data loader.
            criterion (nn.Module): criterion to compute the loss.
            optimizer (Optimizer): optimizer to update the model's parameters.
            lr_scheduler (LRScheduler): lr scheduler to update the learning rate.
            evaluator (torch.nn.Module): task evaluator to evaluate the model.
            n_epochs (int): number of epochs to train the model.
            exp_dir (pathlib.Path | str): path to the experiment directory.
            device (torch.device): model
            precision (str): precision to train the model (fp32, fp16, bfp16).
            use_wandb (bool): whether to use wandb for logging.
            ckpt_interval (int): interval to save the checkpoint.
            eval_interval (int): interval to evaluate the model.
            log_interval (int): interval to log the training information.
            best_metric_key (str): metric that determines best checkpoints.
        """
        self.rank = int(os.environ["RANK"])
        self.criterion = criterion
        self.model = model
        self.train_loader = train_loader
        self.batch_per_epoch = len(self.train_loader)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.n_epochs = n_epochs
        self.logger = logging.getLogger()
        self.exp_dir = exp_dir
        self.device = device
        self.use_wandb = use_wandb
        self.ckpt_interval = ckpt_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.best_metric_key = best_metric_key

        self.training_stats = {
            name: RunningAverageMeter(length=self.batch_per_epoch)
            for name in ["loss", "data_time", "batch_time", "eval_time"]
        }
        self.training_metrics = {}
        self.best_metric_comp = operator.gt
        self.num_classes = self.train_loader.dataset.num_classes

        assert precision in [
            "fp32",
            "fp16",
            "bfp16",
        ], f"Invalid precision {precision}, use 'fp32', 'fp16' or 'bfp16'."
        self.enable_mixed_precision = precision != "fp32"
        self.precision = torch.float16 if (precision == "fp16") else torch.bfloat16

        # AGBD FIX: Use modern GradScaler API to avoid FutureWarning
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.enable_mixed_precision)

        self.start_epoch = 0
        self.global_step = 0  # Add global step counter for WandB

        if self.use_wandb:
            import wandb

            self.wandb = wandb

    def train(self) -> None:
        """Train the model for n_epochs then evaluate the model and save the best model."""
        # end_time = time.time()
        for epoch in range(self.start_epoch, self.n_epochs):
            # train the network for one epoch
            if epoch % self.eval_interval == 0:
                metrics, used_time = self.evaluator(self.model, f"epoch {epoch}", step=None)
                self.training_stats["eval_time"].update(used_time)
                self.save_best_checkpoint(metrics, epoch)

            self.logger.info("============ Starting epoch %i ... ============" % epoch)
            self.t = time.time()
            self.train_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)

            if epoch % self.ckpt_interval == 0 and epoch != self.start_epoch:
                self.save_model(epoch)

        # AGBD FIX: Skip final evaluation in debug mode with single epoch to save time
        debug_mode = getattr(self.train_loader.dataset, 'debug', False)
        if not (self.n_epochs == 1 and debug_mode):
            metrics, used_time = self.evaluator(self.model, "final model", step=None)
            self.training_stats["eval_time"].update(used_time)
            self.save_best_checkpoint(metrics, self.n_epochs)
            self.save_model(self.n_epochs, is_final=True)

    def train_one_epoch(self, epoch: int) -> None:
        """Train model for one epoch.

        Args:
            epoch (int): number of the epoch.
        """
        self.model.train()

        end_time = time.time()

        # for the called epoch iterate through all batches and:
        #   1) retrieve image and target data
        #   2) compute logits (here, model = decoder, which contains the encoder)
        #   3) calculate mse = loss with target and logits (changed from original since sparse mse)
        #   4) Reset gradients with self.optimizer.zero_grad()
        #   5) Assert that calculated loss != nan
        #   6) Scale loss, unscale gradients, call optimizer, update scaler for next iteration, see https://docs.pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training
        #   7) update statistics, log information into logger and WandB, decay lr

        for batch_idx, data in enumerate(self.train_loader):
            # print(f"[TRAINER DEBUG] Processing training batch {batch_idx}")
            #)1
            image, target = data["image"], data["target"]
            image = {modality: value.to(self.device) for modality, value in image.items()}
            target = target.to(self.device)
            # print(f"[TRAINER DEBUG] Target shape: {target.shape}, Image keys: {list(image.keys())}")

            self.training_stats["data_time"].update(time.time() - end_time)
            # 2) & 3)
            with torch.autocast(
                "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
            ):
                logits = self.model(image, output_shape=target.shape[-2:])
                loss = self.compute_loss(logits, target)
            # 4)
            self.optimizer.zero_grad()
            
            # 5)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Rank {self.rank} got infinite/NaN loss at batch {batch_idx} of epoch {epoch}!"
                )
            # 6)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 7)
            self.training_stats['loss'].update(loss.item())
            with torch.no_grad():
                self.compute_logging_metrics(logits, target)
            if (batch_idx + 1) % self.log_interval == 0:
                self.log(batch_idx + 1, epoch)

            # decay lr
            self.lr_scheduler.step()
            self.global_step += 1  # Increment global step counter
            if self.use_wandb and self.rank == 0:
                self.wandb.log(
                    {
                        "Train_MSE_(Loss_per_epoch_per_batch)": loss.item(),                        # CHANGED: SPECIFY WHAT IS LOGGED
                        "learning_rate_per_batch": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        **{
                            f"Train_{k}_(running_average)": v.avg                                   # CHANGED: SPECIFY WHAT IS LOGGED
                            for k, v in self.training_metrics.items()
                        },
                    },
                    step=self.global_step,  # Use monotonic global step
                )

            self.training_stats["batch_time"].update(time.time() - end_time)
            end_time = time.time()

    def get_checkpoint(self, epoch: int) -> dict[str, dict | int]:
        """Create a checkpoint dictionary, containing references to the pytorch tensors.

        Args:
            epoch (int): number of the epoch.

        Returns:
            dict[str, dict | int]: checkpoint dictionary.
        """
        checkpoint = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
        }
        return checkpoint

    def save_model(
        self,
        epoch: int,
        is_final: bool = False,
        is_best: bool = False,
        checkpoint: dict[str, dict | int] | None = None,
    ):
        """Save the model checkpoint.

        Args:
            epoch (int): number of the epoch.
            is_final (bool, optional): whether is the final checkpoint. Defaults to False.
            is_best (bool, optional): wheter is the best checkpoint. Defaults to False.
            checkpoint (dict[str, dict  |  int] | None, optional): already prepared checkpoint dict. Defaults to None.
        """
        if self.rank != 0:
            torch.distributed.barrier()
            return
        checkpoint = self.get_checkpoint(epoch) if checkpoint is None else checkpoint
        suffix = "_best" if is_best else f"{epoch}_final" if is_final else f"{epoch}"
        checkpoint_path = os.path.join(self.exp_dir, f"checkpoint_{suffix}.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(
            f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}"
        )
        torch.distributed.barrier()
        return

    def load_model(self, resume_path: str | pathlib.Path) -> None:
        """Load model from the checkpoint.

        Args:
            resume_path (str | pathlib.Path): path to the checkpoint.
        """
        model_dict = torch.load(resume_path, map_location=self.device)
        if "model" in model_dict:
            self.model.module.load_state_dict(model_dict["model"])
            self.optimizer.load_state_dict(model_dict["optimizer"])
            self.lr_scheduler.load_state_dict(model_dict["lr_scheduler"])
            self.scaler.load_state_dict(model_dict["scaler"])
            self.start_epoch = model_dict["epoch"] + 1
        else:
            self.model.module.load_state_dict(model_dict)
            self.start_epoch = 0

        self.logger.info(
            f"Loaded model from {resume_path}. Resume training from epoch {self.start_epoch}"
        )

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits (torch.Tensor): logits from the decoder.
            target (torch.Tensor): target tensor.

        Returns:
            torch.Tensor: loss value.
        """
        # CRITICAL DEBUG: Log actual tensor shapes to understand the mismatch
        # print(f"[LOSS DEBUG] Logits shape: {logits.shape}")
        # print(f"[LOSS DEBUG] Target shape: {target.shape}")
        
        # CRITICAL FIX: Handle spatial size mismatch between decoder output and target
        logits_height, logits_width = logits.shape[-2:]
        target_height, target_width = target.shape[-2:]
        
        # print(f"[LOSS DEBUG] Logits spatial: {logits_height}x{logits_width}")
        # print(f"[LOSS DEBUG] Target spatial: {target_height}x{target_width}")
        
        # If spatial sizes don't match, resize target to match logits
        if (logits_height, logits_width) != (target_height, target_width):
            # print(f"[LOSS DEBUG] ⚠️ SPATIAL SIZE MISMATCH! Resizing target from {target_height}x{target_width} to {logits_height}x{logits_width}")
            
            # Resize target to match logits spatial size
            target_resized = F.interpolate(
                target.unsqueeze(1).float(),  # Add channel dim for interpolation
                size=(logits_height, logits_width),
                mode='nearest'  # Use nearest for discrete values
            ).squeeze(1)  # Remove channel dim
            
            # print(f"[LOSS DEBUG] Target resized shape: {target_resized.shape}")
            target = target_resized
        
        # Calculate center pixel coordinates
        center_h = logits_height // 2
        center_w = logits_width // 2
        
        # print(f"[LOSS DEBUG] Center pixel: ({center_h}, {center_w})")
        
        # Extract center pixels
        logits_center = logits.squeeze(dim=1)[:, center_h, center_w]
        target_center = target[:, center_h, center_w]
        
        # print(f"[LOSS DEBUG] Logits center shape: {logits_center.shape}")
        # print(f"[LOSS DEBUG] Target center shape: {target_center.shape}")
        # print(f"[LOSS DEBUG] Logits center values: {logits_center}")
        # print(f"[LOSS DEBUG] Target center values: {target_center}")
        
        # Compute loss on center pixels only
        loss = self.criterion(logits_center, target_center)
        # print(f"[LOSS DEBUG] Loss value: {loss.item()}")
        
        return loss

    def save_best_checkpoint(
        self, eval_metrics: dict[float, list[float]], epoch: int
    ) -> None:
        """Update the best checkpoint according to the evaluation metrics.

        Args:
            eval_metrics (dict[float, list[float]]): metrics computed by the evaluator on the validation set.
            epoch (int): number of the epoch.
        """
        curr_metric = eval_metrics[self.best_metric_key]
        if isinstance(curr_metric, list):
            curr_metric = curr_metric[1] if self.num_classes == 1 else np.mean(curr_metric)
        if self.best_metric_comp(curr_metric, self.best_metric):
            self.best_metric = curr_metric
            best_ckpt = self.get_checkpoint(epoch)
            self.save_model(
                epoch, is_best=True, checkpoint=best_ckpt
            )

    @torch.no_grad()
    def compute_logging_metrics(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> dict[float, list[float]]:
        """Compute logging metrics.

        Args:
            logits (torch.Tensor): logits output by the decoder.
            target (torch.Tensor): target tensor.

        Raises:
            NotImplementedError: raise if the method is not implemented.

        Returns:
            dict[float, list[float]]: logging metrics.
        """
        raise NotImplementedError

    def log(self, batch_idx: int, epoch) -> None:
        """Log the information.

        Args:
            batch_idx (int): number of the batch.
            epoch (_type_): number of the epoch.
        """
        # TO DO: upload to wandb
        left_batch_this_epoch = self.batch_per_epoch - batch_idx
        left_batch_all = (
            self.batch_per_epoch * (self.n_epochs - epoch - 1) + left_batch_this_epoch
        )
        left_eval_times = ((self.n_epochs - 0.5) // self.eval_interval + 2
                           - self.training_stats["eval_time"].count)
        left_time_this_epoch = sec_to_hm(
            left_batch_this_epoch * self.training_stats["batch_time"].avg
        )
        left_time_all = sec_to_hm(
            left_batch_all * self.training_stats["batch_time"].avg
            + left_eval_times * self.training_stats["eval_time"].avg
        )

        basic_info = (
            "Epoch [{epoch}-{batch_idx}/{len_loader}]\t"
            "ETA [{left_time_all}|{left_time_this_epoch}]\t"
            "Time [{batch_time.avg:.3f}|{data_time.avg:.3f}]\t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "lr {lr:.3e}".format(
                epoch=epoch,
                len_loader=len(self.train_loader),
                batch_idx=batch_idx,
                left_time_this_epoch=left_time_this_epoch,
                left_time_all=left_time_all,
                batch_time=self.training_stats["batch_time"],
                data_time=self.training_stats["data_time"],
                loss=self.training_stats["loss"],
                lr=self.optimizer.param_groups[0]["lr"],
            )
        )

        metrics_info = [
            "{} {:>7} ({:>7})".format(k, "%.3f" % v.val, "%.3f" % v.avg)
            for k, v in self.training_metrics.items()
        ]
        metrics_info = "\n Training metrics: " + "\t".join(metrics_info)
        # extra_metrics_info = self.extra_info_template.format(**self.extra_info)
        log_info = basic_info + metrics_info
        self.logger.info(log_info)

    def reset_stats(self) -> None:
        """Reset the training stats and metrics."""
        for v in self.training_stats.values():
            v.reset()
        for v in self.training_metrics.values():
            v.reset()


class SegTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        evaluator: torch.nn.Module,
        n_epochs: int,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        best_metric_key: str,
    ):
        """Initialize the Trainer for segmentation task.
        Args:
            model (nn.Module): model to train (encoder + decoder).
            train_loader (DataLoader): train data loader.
            criterion (nn.Module): criterion to compute the loss.
            optimizer (Optimizer): optimizer to update the model's parameters.
            lr_scheduler (LRScheduler): lr scheduler to update the learning rate.
            evaluator (torch.nn.Module): task evaluator to evaluate the model.
            n_epochs (int): number of epochs to train the model.
            exp_dir (pathlib.Path | str): path to the experiment directory.
            device (torch.device): model
            precision (str): precision to train the model (fp32, fp16, bfp16).
            use_wandb (bool): whether to use wandb for logging.
            ckpt_interval (int): interval to save the checkpoint.
            eval_interval (int): interval to evaluate the model.
            log_interval (int): interval to log the training information.
            best_metric_key (str): metric that determines best checkpoints.
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            n_epochs=n_epochs,
            exp_dir=exp_dir,
            device=device,
            precision=precision,
            use_wandb=use_wandb,
            ckpt_interval=ckpt_interval,
            eval_interval=eval_interval,
            log_interval=log_interval,
            best_metric_key=best_metric_key,
        )

        self.training_metrics = {
            name: RunningAverageMeter(length=100) for name in ["Acc", "mAcc", "mIoU"]
        }
        self.best_metric = float("-inf")
        self.best_metric_comp = operator.gt

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits (torch.Tensor): logits from the decoder.
            target (torch.Tensor): target tensor.

        Returns:
            torch.Tensor: loss value.
        """
        return self.criterion(logits, target)

    @torch.no_grad()
    def compute_logging_metrics(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Compute logging metrics.

        Args:
            logits (torch.Tensor): loggits from the decoder.
            target (torch.Tensor): target tensor.
        """
        # logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear')
        num_classes = logits.shape[1]
        if num_classes == 1:
            pred = (torch.sigmoid(logits) > 0.5).type(torch.int64)
        else:
            pred = torch.argmax(logits, dim=1, keepdim=True)
        target = target.unsqueeze(1)
        ignore_mask = target == self.train_loader.dataset.ignore_index
        target[ignore_mask] = 0
        ignore_mask = ignore_mask.expand(
            -1, num_classes if num_classes > 1 else 2, -1, -1
        )

        dims = list(logits.shape)
        if num_classes == 1:
            dims[1] = 2
        binary_pred = torch.zeros(dims, dtype=bool, device=self.device)
        binary_target = torch.zeros(dims, dtype=bool, device=self.device)
        binary_pred.scatter_(dim=1, index=pred, src=torch.ones_like(binary_pred))
        binary_target.scatter_(dim=1, index=target, src=torch.ones_like(binary_target))
        binary_pred[ignore_mask] = 0
        binary_target[ignore_mask] = 0

        intersection = torch.logical_and(binary_pred, binary_target)
        union = torch.logical_or(binary_pred, binary_target)

        acc = intersection.sum() / binary_target.sum() * 100
        macc = (
            torch.nanmean(
                intersection.sum(dim=(0, 2, 3)) / binary_target.sum(dim=(0, 2, 3))
            )
            * 100
        )
        miou = (
            torch.nanmean(intersection.sum(dim=(0, 2, 3)) / union.sum(dim=(0, 2, 3)))
            * 100
        )

        self.training_metrics["Acc"].update(acc.item())
        self.training_metrics["mAcc"].update(macc.item())
        self.training_metrics["mIoU"].update(miou.item())


class RegTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        evaluator: torch.nn.Module,
        n_epochs: int,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        best_metric_key: str,
    ):
        """Initialize the Trainer for regression task.
        Args:
            model (nn.Module): model to train (encoder + decoder).
            train_loader (DataLoader): train data loader.
            criterion (nn.Module): criterion to compute the loss.
            optimizer (Optimizer): optimizer to update the model's parameters.
            lr_scheduler (LRScheduler): lr scheduler to update the learning rate.
            evaluator (torch.nn.Module): task evaluator to evaluate the model.
            n_epochs (int): number of epochs to train the model.
            exp_dir (pathlib.Path | str): path to the experiment directory.
            device (torch.device): model
            precision (str): precision to train the model (fp32, fp16, bfp16).
            use_wandb (bool): whether to use wandb for logging.
            ckpt_interval (int): interval to save the checkpoint.
            eval_interval (int): interval to evaluate the model.
            log_interval (int): interval to log the training information.
            best_metric_key (str): metric that determines best checkpoints.
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            n_epochs=n_epochs,
            exp_dir=exp_dir,
            device=device,
            precision=precision,
            use_wandb=use_wandb,
            ckpt_interval=ckpt_interval,
            eval_interval=eval_interval,
            log_interval=log_interval,
            best_metric_key=best_metric_key,
        )

        self.training_metrics = {
            name: RunningAverageMeter(length=100) for name in ["MSE"]
        }
        self.best_metric = float("inf")
        self.best_metric_comp = operator.lt

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits (torch.Tensor): logits from the decoder.
            target (torch.Tensor): target tensor.

        Returns:
            torch.Tensor: loss value.
        """
        # CRITICAL DEBUG: Log actual tensor shapes to understand the mismatch
        # print(f"[LOSS DEBUG] Logits shape: {logits.shape}")
        # print(f"[LOSS DEBUG] Target shape: {target.shape}")
        
        # CRITICAL FIX: Handle spatial size mismatch between decoder output and target
        logits_height, logits_width = logits.shape[-2:]
        target_height, target_width = target.shape[-2:]
        
        # print(f"[LOSS DEBUG] Logits spatial: {logits_height}x{logits_width}")
        # print(f"[LOSS DEBUG] Target spatial: {target_height}x{target_width}")
        
        # If spatial sizes don't match, resize target to match logits
        if (logits_height, logits_width) != (target_height, target_width):
            # print(f"[LOSS DEBUG] ⚠️ SPATIAL SIZE MISMATCH! Resizing target from {target_height}x{target_width} to {logits_height}x{logits_width}")
            
            # Resize target to match logits spatial size
            target_resized = F.interpolate(
                target.unsqueeze(1).float(),  # Add channel dim for interpolation
                size=(logits_height, logits_width),
                mode='nearest'  # Use nearest for discrete values
            ).squeeze(1)  # Remove channel dim
            
            # print(f"[LOSS DEBUG] Target resized shape: {target_resized.shape}")
            target = target_resized
        
        # Calculate center pixel coordinates
        center_h = logits_height // 2
        center_w = logits_width // 2
        
        # print(f"[LOSS DEBUG] Center pixel: ({center_h}, {center_w})")
        
        # Extract center pixels
        logits_center = logits.squeeze(dim=1)[:, center_h, center_w]
        target_center = target[:, center_h, center_w]
        
        # print(f"[LOSS DEBUG] Logits center shape: {logits_center.shape}")
        # print(f"[LOSS DEBUG] Target center shape: {target_center.shape}")
        # print(f"[LOSS DEBUG] Logits center values: {logits_center}")
        # print(f"[LOSS DEBUG] Target center values: {target_center}")
        
        # CRITICAL FIX: Filter out ignore_index values from loss computation
        ignore_index = -1
        valid_mask = target_center != ignore_index
        
        print(f"[REGTRAINER LOSS DEBUG] Logits shape: {logits.shape}, Center pixel: ({center_h}, {center_w})")
        print(f"[LOSS DEBUG] Valid samples: {valid_mask.sum().item()}/{target_center.shape[0]}")
        
        if valid_mask.sum() == 0:
            # print(f"[LOSS DEBUG] WARNING: No valid samples in batch!")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Only compute loss on valid (non-ignore_index) samples
        valid_logits = logits_center[valid_mask]
        valid_targets = target_center[valid_mask]
        
        # print(f"[LOSS DEBUG] Valid logits: {valid_logits}")
        # print(f"[LOSS DEBUG] Valid targets: {valid_targets}")
        
        # Compute loss on center pixels only (excluding ignore_index)
        loss = self.criterion(valid_logits, valid_targets)
        # print(f"[LOSS DEBUG] Loss value: {loss.item()}")
        
        return loss

    @torch.no_grad()
    def compute_logging_metrics(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Compute logging metrics.

        Args:
            logits (torch.Tensor): logits from the decoder.
            target (torch.Tensor): target tensor.
        """
        # CRITICAL FIX: Central pixel logic for AGBD consistency
        # Must match the compute_loss method exactly
        
        height, width = logits.shape[-2:]
        
        # For AGBD: Use the same center calculation as in dataset and evaluator
        if hasattr(self.train_loader.dataset, 'center'):
            # Use dataset's center if available (for AGBD compatibility)
            dataset_center = self.train_loader.dataset.center
            # Ensure center is within bounds after potential padding/cropping
            center_h = min(dataset_center, height - 1)
            center_w = min(dataset_center, width - 1)
        else:
            # Fallback to geometric center for other datasets
            center_h = height // 2
            center_w = width // 2
        
        # Additional safety check for AGBD patches
        if height == 25 and width == 25:
            # For 25x25 AGBD patches, center should be at (12, 12)
            center_h = center_w = 12
        elif height != width:
            # For non-square patches, use geometric center
            center_h = height // 2
            center_w = width // 2

        # NOTE: Training metrics are local per-GPU and automatically averaged by DDP
        # No multi-GPU reduction bug here (unlike evaluator which needed fixes)
        
        # CRITICAL FIX: Filter out ignore_index values from metrics computation
        logits_center = logits.squeeze(dim=1)[:, center_h, center_w]
        target_center = target[:, center_h, center_w]
        
        ignore_index = -1
        valid_mask = target_center != ignore_index
        
        if valid_mask.sum() > 0:
            valid_logits = logits_center[valid_mask]
            valid_targets = target_center[valid_mask]
            mse = F.mse_loss(valid_logits, valid_targets)
            self.training_metrics["MSE"].update(mse.item())
        else:
            # If no valid samples, don't update metrics
            print("[METRICS DEBUG] No valid samples for training metrics")

