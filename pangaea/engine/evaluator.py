import logging
import os
import time
from pathlib import Path
import math
import numpy as np
import sklearn.metrics
import wandb
import csv
import random
import traceback

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from pangaea.decoders.knnclassifier import KNNClassifier
from tqdm import tqdm

# AGBD-specific visualization import - required for regression plotting
from agbd_visualization_new import log_agbd_regression_visuals



class Evaluator:
    """
    Evaluator class for evaluating the models.
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for experiment outputs.
        device (torch.device): Device to run the evaluation on (e.g., CPU or GPU).
        use_wandb (bool): Flag to indicate if Weights and Biases (wandb) is used for logging.
        logger (logging.Logger): Logger for logging information.
        classes (list): List of class names in the dataset.
        split (str): Dataset split (e.g., 'train', 'val', 'test').
        ignore_index (int): Index to ignore in the dataset.
        num_classes (int): Number of classes in the dataset.
        max_name_len (int): Maximum length of class names.
        wandb (module): Weights and Biases module for logging (if use_wandb is True).
    Methods:
        __init__(val_loader: DataLoader, exp_dir: str | Path, device: torch.device, use_wandb: bool) -> None:
            Initializes the Evaluator with the given parameters.
        evaluate(model: torch.nn.Module, model_name: str, model_ckpt_path: str | Path | None = None) -> None:
            Evaluates the given model. This method should be implemented by subclasses.
        __call__(model: torch.nn.Module) -> None:
            Calls the evaluator on the given model.
        compute_metrics() -> None:
            Computes evaluation metrics. This method should be implemented by subclasses.
        log_metrics(metrics: dict) -> None:
            Logs the computed metrics. This method should be implemented by subclasses.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            exp_dir: str | Path,
            device: torch.device,
            inference_mode: str = 'sliding',
            sliding_inference_batch: int = None,
            use_wandb: bool = False,
    ) -> None:
        self.rank = int(os.environ["RANK"])
        self.val_loader = val_loader
        self.logger = logging.getLogger()
        self.exp_dir = exp_dir
        self.device = device
        self.inference_mode = inference_mode
        self.sliding_inference_batch = sliding_inference_batch
        self.classes = self.val_loader.dataset.classes
        self.split = self.val_loader.dataset.split
        self.ignore_index = self.val_loader.dataset.ignore_index
        self.num_classes = len(self.classes)
        self.max_name_len = max([len(name) for name in self.classes])
        self.use_wandb = use_wandb
        
        # Compute valid class indices (excluding ignore index)
        self.valid_class_indices = [
            i for i in range(self.num_classes) if i != self.ignore_index
        ]
        self.valid_classes = [self.classes[i] for i in self.valid_class_indices]

    def evaluate(
            self,
            model: torch.nn.Module,
            model_name: str,
            model_ckpt_path: str | Path | None = None,
            step: int | None = None,
    ) -> None:
        raise NotImplementedError

    def __call__(self, model):
        pass

    def compute_metrics(self):
        pass

    def log_metrics(self, metrics):
        pass

    @staticmethod
    def sliding_inference(model, img, input_size, output_shape=None, stride=None, max_batch=None):
        b, c, t, height, width = img[list(img.keys())[0]].shape

        if stride is None:
            h = int(math.ceil(height / input_size))
            w = int(math.ceil(width / input_size))
        else:
            h = math.ceil((height - input_size) / stride) + 1
            w = math.ceil((width - input_size) / stride) + 1

        h_grid = torch.linspace(0, height - input_size, h).round().long()
        w_grid = torch.linspace(0, width - input_size, w).round().long()
        num_crops_per_img = h * w

        for k, v in img.items():
            img_crops = []
            for i in range(h):
                for j in range(w):
                    img_crops.append(v[:, :, :, h_grid[i]:h_grid[i] + input_size, w_grid[j]:w_grid[j] + input_size])
            img[k] = torch.cat(img_crops, dim=0)

        pred = []
        max_batch = max_batch if max_batch is not None else b * num_crops_per_img
        batch_num = int(math.ceil(b * num_crops_per_img / max_batch))
        for i in range(batch_num):
            img_ = {k: v[max_batch * i: min(max_batch * i + max_batch, b * num_crops_per_img)] for k, v in img.items()}
            pred_ = model.forward(img_, output_shape=(input_size, input_size))
            pred.append(pred_)
        pred = torch.cat(pred, dim=0)
        pred = pred.view(num_crops_per_img, b, -1, input_size, input_size).transpose(0, 1)

        merged_pred = torch.zeros((b, pred.shape[2], height, width), device=pred.device)
        pred_count = torch.zeros((b, height, width), dtype=torch.long, device=pred.device)
        for i in range(h):
            for j in range(w):
                merged_pred[:, :, h_grid[i]:h_grid[i] + input_size,
                w_grid[j]:w_grid[j] + input_size] += pred[:, h * i + j]
                pred_count[:, h_grid[i]:h_grid[i] + input_size,
                w_grid[j]:w_grid[j] + input_size] += 1

        merged_pred = merged_pred / pred_count.unsqueeze(1)
        if output_shape is not None:
            merged_pred = F.interpolate(merged_pred, size=output_shape, mode="bilinear")

        return merged_pred


class LinearClassificationEvaluator(Evaluator):
    """RESTORED: Linear classification evaluator that was mistakenly removed in AGBD version"""
    def __init__(
        self,
        val_loader,
        exp_dir: str | Path,
        device: torch.device,
        inference_mode: str = "whole",
        sliding_inference_batch: int = None,
        use_wandb: bool = False,
        multi_label: bool = False,   # Flag to indicate multi-label evaluation
        topk: int = 1,               # For multi-label: if > 1, use top-k selection
    ) -> None:
        super().__init__(val_loader, exp_dir, device, inference_mode, sliding_inference_batch, use_wandb)
        self.multi_label = multi_label
        self.topk = topk
        
    def evaluate(
        self, 
        model: torch.nn.Module, 
        model_name: str, 
        model_ckpt_path: str | Path | None = None,
        step: int | None = None):
        
        t = time.time()
        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)
            self.logger.info(f"Loaded {model_name} for evaluation")
        
        model.eval()
        
        all_preds = []
        all_targets = []
        total_correct = 0
        total_samples = 0
        
        tag = f"Evaluating {model_name} on {self.split} set"
        for batch_idx, data in enumerate(tqdm(self.val_loader, desc=tag)):
            image, target = data["image"], data["target"]
            image = {k: v.to(self.device) for k, v in image.items()}
            target = target.to(self.device)
            
            with torch.no_grad():
                logits = model(image)
            
            if self.multi_label:
                # Multi-label evaluation:
                preds_prob = torch.sigmoid(logits)
                if self.topk > 1:
                    topk_indices = preds_prob.topk(self.topk, dim=1).indices
                    preds = torch.zeros_like(preds_prob, dtype=torch.int)
                    preds.scatter_(1, topk_indices, 1)
                else:
                    preds = (preds_prob > 0.5).int()

                all_preds.append(preds.cpu().numpy())
                all_targets.append(target.cpu().numpy())
            else:
                preds = torch.argmax(logits, dim=1)  

                total_correct += (preds == target).sum().item()
                total_samples += target.numel()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        if self.multi_label:
            accuracy = sklearn.metrics.accuracy_score(all_targets, all_preds)
            precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
                all_targets, all_preds, average="micro", zero_division=0)
        else:
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
                    all_targets, all_preds,labels=list(range(self.num_classes)), average="macro", zero_division=0)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "F1": f1,
        }
        
        self.log_metrics(metrics)
        used_time = time.time() - t
        return metrics, used_time
    
    def __call__(self, model, model_name, model_ckpt_path=None, step=None):
        return self.evaluate(model, model_name, model_ckpt_path, step=step)
    
    def compute_metrics(self):
        pass

    def log_metrics(self, metrics: dict):
        def format_metric(name, value):
            header = f"[{self.split}] ------- {name} --------\n"
            value_str = (
                f"[{self.split}] -------------------\n"
                + f"[{self.split}] Mean".ljust(self.max_name_len, " ")
                + "\t{:>7}".format("%.3f" % value)
            )
            return header + value_str

        acc_str = format_metric("Accuracy", metrics["accuracy"])
        prec_str = format_metric("Precision", metrics["precision"])
        recall_str = format_metric("Recall", metrics["recall"])
        f1_str = format_metric("F1-score", metrics["F1"])
        self.logger.info(acc_str)
        self.logger.info(prec_str)
        self.logger.info(recall_str)
        self.logger.info(f1_str)

        if self.use_wandb and self.rank == 0:
            wandb.log({
                f"{self.split}_accuracy": metrics["accuracy"],
                f"{self.split}_precision": metrics["precision"],
                f"{self.split}_recall": metrics["recall"],
                f"{self.split}_f1": metrics["F1"],
            })


class SegEvaluator(Evaluator):
    """
    SegEvaluator is a class for evaluating segmentation models. It extends the Evaluator class and provides methods
    to evaluate a model, compute metrics, and log the results.
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for saving experiment results.
        device (torch.device): Device to run the evaluation on.
        use_wandb (bool): Flag to indicate whether to use Weights and Biases for logging.
    Methods:
        evaluate(model, model_name='model', model_ckpt_path=None):
            Evaluates the given model on the validation dataset and computes metrics.
        __call__(model, model_name, model_ckpt_path=None):
            Calls the evaluate method. This allows the object to be used as a function.
        compute_metrics(confusion_matrix):
            Computes various metrics such as IoU, precision, recall, F1-score, mean IoU, mean F1-score, and mean accuracy
            from the given confusion matrix.
        log_metrics(metrics):
            Logs the computed metrics. If use_wandb is True, logs the metrics to Weights and Biases.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            exp_dir: str | Path,
            device: torch.device,
            inference_mode: str = 'sliding',
            sliding_inference_batch: int = None,
            use_wandb: bool = False,
            visualization_interval: int = 120,  # new
    ):
        super().__init__(val_loader, exp_dir, device, inference_mode, sliding_inference_batch, use_wandb)
        self.visualization_interval = visualization_interval

    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None, step=None):
        t = time.time()

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)

            self.logger.info(f"Loaded {model_name} for evaluation")
        model.eval()

        tag = f"Evaluating {model_name} on {self.split} set"
        confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), device=self.device
        )

        for batch_idx, data in enumerate(tqdm(self.val_loader, desc=tag)):

            image, target = data["image"], data["target"]
            image = {k: v.to(self.device) for k, v in image.items()}
            target = target.to(self.device)

            if self.inference_mode == "sliding":
                input_size = model.module.encoder.input_size
                logits = self.sliding_inference(model, image, input_size, output_shape=target.shape[-2:],
                                                max_batch=self.sliding_inference_batch)
            elif self.inference_mode == "whole":
                logits = model(image, output_shape=target.shape[-2:])
            else:
                raise NotImplementedError((f"Inference mode {self.inference_mode} is not implemented."))
            if logits.shape[1] == 1:
                pred = (torch.sigmoid(logits) > 0.5).type(torch.int64).squeeze(dim=1)
            else:
                pred = torch.argmax(logits, dim=1)
            valid_mask = target != self.ignore_index
            pred, target = pred[valid_mask], target[valid_mask]
            count = torch.bincount(
                (pred * self.num_classes + target), minlength=self.num_classes ** 2
            )
            confusion_matrix += count.view(self.num_classes, self.num_classes)

            # ------------------------------ VISUALIZATION ------------------------------
            # Save some intermediate results (only preds that have valid values only and if wandb is activated)
            if torch.all(torch.flatten(valid_mask)) and self.use_wandb and self.rank == 0:
                # Log every visualization_interval batches, or for specific model_name
                should_log = False
                if (batch_idx % self.visualization_interval == 0) or (model_name in ["epoch 5", "epoch 10", "epoch 30", "epoch 60", "checkpoint__best"]):
                    should_log = True
                if should_log:
                    pred_saved = pred.clone()
                    target_saved = target.clone()
                    pred_np = pred_saved.cpu().numpy()
                    target_np = target_saved.cpu().numpy()
                    # Try to infer a square shape, fallback to flat if not possible
                    dim_2 = int(np.sqrt(len(pred_np)))
                    if dim_2 * dim_2 == len(pred_np):
                        img_pred = wandb.Image(pred_np.reshape((dim_2, dim_2)))
                        img_target = wandb.Image(target_np.reshape((dim_2, dim_2)))
                    else:
                        img_pred = wandb.Image(pred_np)
                        img_target = wandb.Image(target_np)
                    # Sortable keys
                    if batch_idx < 10:
                        wandb.log({f"batch_00{batch_idx}_{model_name}_pred": img_pred})
                        wandb.log({f"batch_00{batch_idx}_{model_name}_target": img_target})
                    elif batch_idx < 100:
                        wandb.log({f"batch_0{batch_idx}_{model_name}_pred": img_pred})
                        wandb.log({f"batch_0{batch_idx}_{model_name}_target": img_target})
                    else:
                        wandb.log({f"batch_{batch_idx}_{model_name}_pred": img_pred})
                        wandb.log({f"batch_{batch_idx}_{model_name}_target": img_target})
            # ------------------------------ VISUALIZATION ------------------------------


        torch.distributed.all_reduce(
            confusion_matrix, op=torch.distributed.ReduceOp.SUM
        )
        metrics = self.compute_metrics(confusion_matrix.cpu())
        self.log_metrics(metrics)

        used_time = time.time() - t

        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name, model_ckpt_path=None, step=None):
        return self.evaluate(model, model_name, model_ckpt_path, step=step)

    def compute_metrics(self, confusion_matrix):
        # Calculate IoU for each class
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
        iou = (intersection / (union + 1e-6)) * 100

        # Calculate precision and recall for each class
        precision = intersection / (confusion_matrix.sum(dim=0) + 1e-6) * 100
        recall = intersection / (confusion_matrix.sum(dim=1) + 1e-6) * 100

        # Calculate F1-score for each class
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        # Calculate mean IoU, mean F1-score, and mean Accuracy
        valid = self.valid_class_indices
            
        miou = iou[valid].mean().item() if valid else 0.0
        mf1 = f1[valid].mean().item() if valid else 0.0
        macc = (intersection.sum() / (confusion_matrix.sum() + 1e-6)).item() * 100

        # Convert metrics to CPU and to Python scalars
        iou = [iou[i].item() for i in valid]
        f1 = [f1[i].item() for i in valid]
        precision = [precision[i].item() for i in valid]
        recall = [recall[i].item() for i in valid]
        
        # Prepare the metrics dictionary
        metrics = {
            "IoU": iou,
            "mIoU": miou,
            "F1": f1,
            "mF1": mf1,
            "mAcc": macc,
            "Precision": precision,
            "Recall": recall,
        }

        return metrics

    def log_metrics(self, metrics):
        def format_metric(name, values, mean_value):
            header = f"------- {name} --------\n"
            metric_str = (
                    "\n".join(
                        c.ljust(self.max_name_len, " ") + "\t{:>7}".format("%.3f" % num)
                        for c, num in zip(self.valid_classes, values)
                    )
                    + "\n"
            )
            mean_str = (
                    "-------------------\n"
                    + "Mean".ljust(self.max_name_len, " ")
                    + "\t{:>7}".format("%.3f" % mean_value)
            )
            return header + metric_str + mean_str

        iou_str = format_metric("IoU", metrics["IoU"], metrics["mIoU"])
        f1_str = format_metric("F1-score", metrics["F1"], metrics["mF1"])

        precision_mean = sum(metrics["Precision"]) / len(metrics["Precision"]) if metrics["Precision"] else 0.0
        recall_mean = sum(metrics["Recall"]) / len(metrics["Recall"]) if metrics["Recall"] else 0.0

        precision_str = format_metric("Precision", metrics["Precision"], precision_mean)
        recall_str = format_metric("Recall", metrics["Recall"], recall_mean)

        macc_str = f"Mean Accuracy: {metrics['mAcc']:.3f} \n"

        self.logger.info(iou_str)
        self.logger.info(f1_str)
        self.logger.info(precision_str)
        self.logger.info(recall_str)
        self.logger.info(macc_str)

        if self.use_wandb and self.rank == 0:
            wandb.log(
                {
                    f"{self.split}_mIoU": metrics["mIoU"],
                    f"{self.split}_mF1": metrics["mF1"],
                    f"{self.split}_mAcc": metrics["mAcc"],
                    **{
                        f"{self.split}_IoU_{c}": v
                        for c, v in zip(self.valid_classes, metrics["IoU"])
                    },
                    **{
                        f"{self.split}_F1_{c}": v
                        for c, v in zip(self.valid_classes, metrics["F1"])
                    },
                    **{
                        f"{self.split}_Precision_{c}": v
                        for c, v in zip(self.valid_classes, metrics["Precision"])
                    },
                    **{
                        f"{self.split}_Recall_{c}": v
                        for c, v in zip(self.valid_classes, metrics["Recall"])
                    },
                }
            )


class RegEvaluator(Evaluator):
    """
    RegEvaluator is a subclass of Evaluator designed for regression tasks. It evaluates a given model on a validation dataset and computes metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for saving experiment results.
        device (torch.device): Device to run the evaluation on (e.g., CPU or GPU).
        use_wandb (bool): Flag to indicate whether to log metrics to Weights and Biases (wandb).
    Methods:
        evaluate(model, model_name='model', model_ckpt_path=None):
            Evaluates the model on the validation dataset and computes MSE and RMSE.
        __call__(model, model_name='model', model_ckpt_path=None):
            Calls the evaluate method. This allows the object to be used as a function.
        log_metrics(metrics):
            Logs the computed metrics (MSE and RMSE) to the logger and optionally to wandb.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            exp_dir: str | Path,
            device: torch.device,
            inference_mode: str = 'sliding',
            sliding_inference_batch: int = None,
            use_wandb: bool = False,
            visualization_interval: int = 120,  # new
    ):
        super().__init__(val_loader, exp_dir, device, inference_mode, sliding_inference_batch, use_wandb)
        self.visualization_interval = visualization_interval

    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None, testing_bool=False, step=None):
        t = time.time()
        self.testing_bool=testing_bool

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)  # added 'weights_only=False'
            model_name = os.path.basename(model_ckpt_path).split('.')[0]
            if 'model' in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)

            self.logger.info(f"Loaded model from {model_ckpt_path} for evaluation")

        model.eval()

        tag = f'Evaluating {model_name} on {self.split} set'

        mse = torch.zeros(1, device=self.device)
        mean_absolute_error = torch.zeros(1, device=self.device)
        mean_error = torch.zeros(1, device=self.device)
        total_samples = 0  # Track total samples for proper averaging


        for batch_idx, data in enumerate(tqdm(self.val_loader, desc=tag)):
            # CRITICAL DEBUG: Track data through evaluation pipeline
            # print(f"\n[EVALUATOR DEBUG] Batch {batch_idx}")
            # print(f"[EVALUATOR DEBUG] Data keys: {list(data.keys())}")
            
            # if 'image' in data:
            #     print(f"[EVALUATOR DEBUG] Image modalities: {list(data['image'].keys())}")
            #     for mod, tensor in data['image'].items():
            #         print(f"[EVALUATOR DEBUG] {mod} input shape: {tensor.shape}")
            #         print(f"[EVALUATOR DEBUG] {mod} input range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
            
            # if 'target' in data:
            #     print(f"[EVALUATOR DEBUG] Target input shape: {data['target'].shape}")
            #     print(f"[EVALUATOR DEBUG] Target input range: [{data['target'].min().item():.6f}, {data['target'].max().item():.6f}]")
            #     print(f"[EVALUATOR DEBUG] Target unique values (first 10): {torch.unique(data['target'])[:10]}")
            
            image, target = data['image'], data['target']
            image = {k: v.to(self.device) for k, v in image.items()}
            target = target.to(self.device)

            if self.inference_mode == "sliding":
                input_size = model.module.encoder.input_size
                # print(f"[EVALUATOR DEBUG] Using sliding inference with input_size: {input_size}")
                logits = self.sliding_inference(model, image, input_size, output_shape=target.shape[-2:],
                                                max_batch=self.sliding_inference_batch).squeeze(dim=1)
            elif self.inference_mode == "whole":
                # print(f"[EVALUATOR DEBUG] Using whole image inference")
                logits = model(image, output_shape=target.shape[-2:]).squeeze(dim=1)
            else:
                raise NotImplementedError((f"Inference mode {self.inference_mode} is not implemented."))
            
            # print(f"[EVALUATOR DEBUG] Model output (logits) shape: {logits.shape}")
            # print(f"[EVALUATOR DEBUG] Model output range: [{logits.min().item():.6f}, {logits.max().item():.6f}]")
            # print(f"[EVALUATOR DEBUG] Model output mean: {logits.mean().item():.6f}")
            
            # CRITICAL FIX: Handle spatial size mismatch between decoder output and target
            # Special handling for AGBD to prevent target interpolation that destroys single-pixel supervision
            
            logits_height, logits_width = logits.shape[-2:]
            target_height, target_width = target.shape[-2:]
            
            # Check if this is AGBD dataset
            is_agbd = hasattr(self.val_loader.dataset, 'dataset_name') and self.val_loader.dataset.dataset_name == 'agbd'
            
            if logits_height != target_height or logits_width != target_width:
                if is_agbd:
                    # AGBD-specific handling: DO NOT interpolate targets
                    # Instead, track where the GEDI pixel should be in the larger logits space
                    print(f"[AGBD EVALUATOR] Size mismatch - Logits: {(logits_height, logits_width)}, Target: {(target_height, target_width)}")
                    
                    # Find the GEDI pixel in the original target
                    valid_mask = target != self.ignore_index
                    valid_pixels = torch.nonzero(valid_mask.view(-1, target_height, target_width), as_tuple=False)
                    
                    if len(valid_pixels) > 0:
                        # Group by batch
                        gedi_locations = {}
                        for pixel in valid_pixels:
                            batch_idx = pixel[0].item()
                            if batch_idx not in gedi_locations:
                                gedi_locations[batch_idx] = []
                            gedi_locations[batch_idx].append((pixel[1].item(), pixel[2].item()))
                        
                        # Create new target tensor matching logits size
                        new_target = torch.full_like(logits.squeeze(1), self.ignore_index, dtype=target.dtype)
                        
                        # Map GEDI pixels to corresponding locations in larger space
                        scale_h = logits_height / target_height
                        scale_w = logits_width / target_width
                        
                        for batch_idx in range(target.shape[0]):
                            if batch_idx in gedi_locations:
                                for gedi_h, gedi_w in gedi_locations[batch_idx]:
                                    # Scale coordinates to larger space
                                    new_gedi_h = int(gedi_h * scale_h)
                                    new_gedi_w = int(gedi_w * scale_w)
                                    
                                    # Clamp to bounds
                                    new_gedi_h = max(0, min(new_gedi_h, logits_height - 1))
                                    new_gedi_w = max(0, min(new_gedi_w, logits_width - 1))
                                    
                                    # Set the biomass value at scaled location
                                    original_value = target[batch_idx, gedi_h, gedi_w]
                                    new_target[batch_idx, new_gedi_h, new_gedi_w] = original_value
                                    
                                    print(f"[AGBD EVALUATOR] Batch {batch_idx}: GEDI pixel ({gedi_h},{gedi_w}) → ({new_gedi_h},{new_gedi_w}), Value: {original_value:.2f}")
                        
                        target = new_target
                    else:
                        print(f"[AGBD EVALUATOR] WARNING: No valid GEDI pixels found in target")
                        # Create empty target matching logits size
                        target = torch.full_like(logits.squeeze(1), self.ignore_index, dtype=target.dtype)
                        
                else:
                    # Standard interpolation for non-AGBD datasets
                    target = F.interpolate(
                        target.unsqueeze(1).float(),  # Add channel dim and convert to float
                        size=(logits_height, logits_width),
                        mode='nearest'
                    ).squeeze(1)  # Remove channel dim
            
            # For AGBD: Calculate center pixel location based on current spatial size
            height, width = logits.shape[-2:]
            
            if height == 25 and width == 25:
                # Original AGBD patch size
                center_h = center_w = 12
            elif height == 32 and width == 32:
                # Padded AGBD patch size (25->32 with symmetric padding)
                # Center moves from (12,12) to (15,15) due to 3.5px padding on each side
                center_h = center_w = 15
            else:
                # Fallback to geometric center for other sizes
                center_h = height // 2
                center_w = width // 2
            
            # print(f"[EVALUATOR DEBUG] Using geometric center: ({center_h}, {center_w}) for {height}x{width} patch")
            
            # Create central pixel coordinates for visualization
            central_pixel_coords = [(center_h, center_w)] * logits.shape[0]
            
            assert logits.shape == target.shape, f"Shape error in evaluator.py: logits.shape = {logits.shape} != target.shape {target.shape}"

            # Extract central pixel values for current batch
            central_logits = logits[:, center_h, center_w]  # [batch_size]
            central_targets = target[:, center_h, center_w]  # [batch_size]
            
            # CRITICAL FIX: Handle ignore_index for padded values (same as segmentation evaluator)
            # This was missing and causing padded values to be included in loss calculation!
            if self.ignore_index is not None:
                valid_mask = central_targets != self.ignore_index
                if valid_mask.sum() == 0:
                    # Skip this batch if all values are padded/ignored
                    print(f"[WARN] Batch {batch_idx}: All central pixels are ignore_index ({self.ignore_index}), skipping")
                    continue
                
                # Only use valid (non-padded) samples for metrics
                central_logits = central_logits[valid_mask]
                central_targets = central_targets[valid_mask]
                batch_size = central_logits.shape[0]
                
                print(f"[DEBUG] Batch {batch_idx}: {valid_mask.sum().item()}/{len(valid_mask)} valid samples (ignore_index={self.ignore_index})")
            else:
                batch_size = central_logits.shape[0]
            
            # Calculate batch size for proper averaging
            total_samples += batch_size

            # Calculate metrics for current batch - sum up for later averaging
            batch_mse = F.mse_loss(central_logits, central_targets, reduction='sum')
            mse += batch_mse

            # Calculate extra metrics for logging
            mse_to_print = F.mse_loss(central_logits, central_targets, reduction='mean')

            error_per_sample = (central_logits - central_targets)
            mean_error += torch.sum(error_per_sample)  # Sum for later averaging

            absolute_error_per_sample = torch.abs(error_per_sample)
            mean_absolute_error += torch.sum(absolute_error_per_sample)  # Sum for later averaging

            # Define variable, so that metrics are only saved when testing, not when validating
            if self.testing_bool:
                path_to_csv = os.path.join(self.exp_dir, f"test_metrics/metrics.csv")
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(path_to_csv), exist_ok=True)
                with open(path_to_csv, 'a') as file:
                    writer = csv.writer(file)
                    for i in range(central_targets.shape[0]):
                        writer.writerow([batch_idx, i, central_targets[i].item(), central_logits[i].item(), mse_to_print.item()])
                    file.close()

            # Note: Advanced AGBD visualization handles all visualization needs
            # The comprehensive visualization above replaces the basic approach below
            
            # ------------------------------ ADVANCED VISUALIZATION (AGBD) ------------------------------
            # Enhanced AGBD visualization with proper biomass units and multi-modal displays
            # Addresses meeting requirements for better visualizations logged to WandB
            if self.use_wandb and self.rank == 0 and ((batch_idx % self.visualization_interval == 0) or (model_name in ["epoch 0", "epoch 10", "checkpoint__best"])):
                try:
                    # Create central pixel coordinates list for batch - WARNING: these may not be accurate after random cropping!
                    batch_size = logits.shape[0]
                    # Use new comprehensive AGBD visualization utility
                    log_agbd_regression_visuals(
                        inputs=data,  # Pass full data dictionary with image bands
                        pred=logits,  # Predicted biomass maps
                        target=target,  # Ground truth biomass maps
                        band_order=getattr(self.val_loader.dataset, 'band_order', None),
                        wandb_run=wandb,
                        step=None,  # Don't use step for visualizations - let WandB auto-increment
                        prefix=self.split,
                        max_samples=1,  # Reduce to 1 sample per batch to minimize logging
                        dataset=self.val_loader.dataset,
                        model_name=model_name  # Pass model name for visualization title
                    )
                except Exception as e:
                    print(f"[WARN] Advanced AGBD visualization failed for batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback to basic visualization
                    try:
                        # CLONE PREDICTED AND GROUND TRUTH TENSORS FOR FALLBACK VIZ
                        pred_saved = logits.clone()
                        target_saved = target.clone()

                        # Create index for random image in the batch
                        max_batch_size = logits.shape[0]
                        i = random.randint(0, max_batch_size-1)

                        # Use the corrected central pixel coordinates
                        logits_pxl = pred_saved[i, center_h, center_w]
                        target_pxl = target_saved[i, center_h, center_w]
                        
                        # Create simple WandB images as fallback
                        pred_np = pred_saved.cpu().numpy()
                        target_np = target_saved.cpu().numpy()
                        
                        # LOG SIMPLE IMAGES TO WANDB AS FALLBACK
                        img_pred = wandb.Image(pred_np[i,:,:], caption=f"prediction (value: {logits_pxl:.3f} Mg/ha)")
                        img_target = wandb.Image(target_np[i,:,:], caption=f"ground truth (value: {target_pxl:.3f} Mg/ha)")

                        wandb.log({f"fallback_batch_{batch_idx}_{model_name}_#{i}: MSE = {mse_to_print:.3f}": (img_pred, img_target)})
                    except Exception as fallback_e:
                        print(f"[WARN] Even fallback visualization failed: {fallback_e}")
                        import traceback
                        traceback.print_exc()
            # ------------------------------ ADVANCED VISUALIZATION (AGBD) ------------------------------


        # CRITICAL FIX: Multi-GPU reduction for all metrics (MEETING NOTES REQUIREMENT)
        # "she sums everything up, torch reduce -> she uses sum she divides by dataset lens by one gpu not all gives inflated error"
        # "divide by number of gpus or use torch reduce average"
        # "for mae there is no reduce for gpus so she just reports mae for first needs to be collected for all gpus"
        
        # Convert total_samples to tensor for reduction
        total_samples_tensor = torch.tensor(total_samples, device=self.device, dtype=torch.float32)
        
        # Only apply distributed reduction if actually running in distributed mode
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            # CORRECT APPROACH: Sum all metrics across GPUs, then divide by total samples across all GPUs
            # This ensures proper averaging and fixes the "inflated error" bug
            
            # Sum metrics and total samples across all GPUs
            torch.distributed.all_reduce(mse, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(mean_absolute_error, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(mean_error, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
            
            # Convert back to python int for logging
            total_samples_all_gpus = int(total_samples_tensor.item())
            
            # Now divide by total samples across ALL GPUs (this was the bug!)
            # Previous code was dividing by samples on one GPU only, causing inflated errors
            mse = mse / total_samples_all_gpus
            mean_absolute_error = mean_absolute_error / total_samples_all_gpus  
            mean_error = mean_error / total_samples_all_gpus
            
        else:
            # Single GPU case - divide by number of samples on this GPU  
            mse = mse / total_samples
            mean_absolute_error = mean_absolute_error / total_samples
            mean_error = mean_error / total_samples
            total_samples_all_gpus = total_samples

        other_metrics = {"MAE": mean_absolute_error.item(), "ME": mean_error.item()}

        metrics = {"MSE": mse.item(), "RMSE": torch.sqrt(mse).item()}
        self.log_metrics(metrics, other_metrics)
        # ===================== until here ... - CHANGED FROM ORIGINAL: IMPLEMENT SPARSE MSE, AND VISUALIZATION - from here =====================


        used_time = time.time() - t

        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name='model', model_ckpt_path=None, step=None):
        return self.evaluate(model, model_name, model_ckpt_path, step)

    def log_metrics(self, metrics, other_metrics):
        header = "------- MSE and RMSE --------\n"
        mse = "-------------------\n" + 'MSE \t{:>7}'.format('%.3f' % metrics['MSE']) + '\n'
        rmse = "-------------------\n" + 'RMSE \t{:>7}'.format('%.3f' % metrics['RMSE'])
        self.logger.info(header + mse + rmse)

        if self.use_wandb and self.rank == 0:
            # ADAPTED TO LOG ADDED METRICS, SEE ABOVE
            wandb.log({f"{self.split}_MSE_(mean_over_test_batches)": metrics["MSE"], f"{self.split}_RMSE_(mean_over_test_batches)": metrics["RMSE"], f"{self.split}_MAE_(mean_over_test_batches)": other_metrics["MAE"], f"{self.split}_ME_(mean_over_test_batches)": other_metrics["ME"]})
