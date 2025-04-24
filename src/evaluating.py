import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from omegaconf import DictConfig
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_manager import DataManager
from src.utils import calculate_total_loss


class Evaluator:
    def __init__(
        self,
        data_manager: DataManager,
        model: nn.Module,
        config: DictConfig,
        accelerator: Accelerator,
    ) -> None:
        """
        Initialize the Evaluator.
        """
        self.data_manager = data_manager
        self.model = model
        self.config = config
        self.accelerator = accelerator

    @staticmethod
    def get_top_k_predictions(predictions: np.ndarray, k: int) -> np.ndarray:
        """
        Get the top-k predictions for each sample.

        Args:
            predictions (np.ndarray): Array of predicted probabilities for each class
                                    Shape: (n_samples, n_classes)

        Returns:
            np.ndarray: Top-k predicted class indices for each sample
                       Shape: (n_samples, k)
        """
        return np.argsort(predictions, axis=1)[:, -k:]

    @staticmethod
    def compute_metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Evaluate predictions using multiple metrics.

        Args:
            y_true (np.ndarray): True class indices, shape (n_samples,)
            y_pred (np.ndarray): Predicted probabilities, shape (n_samples, n_classes)

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        y_pred_top = Evaluator.get_top_k_predictions(y_pred, k=1).squeeze()
        return {
            "precision": precision_score(
                y_true, y_pred_top, average="macro", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred_top, average="macro", zero_division=0
            ),
            "f1": f1_score(y_true, y_pred_top, average="macro", zero_division=0),
        }

    def _evaluation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Performs a single evaluation (validation or test) step."""
        images, species_labels, images_names = batch
        plant_labels = (species_labels != -1).clone().detach().to(dtype=torch.float32)

        # Gather labels
        labels = self.data_manager.gather_all_labels(
            species_labels, plant_labels, images_names
        )

        # Forward pass (no gradients)
        with torch.no_grad():
            outputs = self.model(
                pixel_values=images, labels=labels, plant_mask=labels["plant"] == 1
            )

            loss = calculate_total_loss(
                outputs, self.model.module.head_names, self.config
            )

        return loss, outputs, labels

    def _evaluate_head_batch(
        self, logits: torch.Tensor, labels: torch.Tensor, prefix: str, step: int
    ) -> torch.Tensor:
        """Calculates and logs batch metrics for a specific head."""
        probs = torch.nn.functional.softmax(logits, dim=1).cpu()
        labels = labels.view(-1).cpu()  # Ensure labels are 1D for evaluator

        metrics = Evaluator.compute_metric(labels, probs)
        self.accelerator.log(
            {
                f"{prefix}/batch_precision": metrics["precision"],
                f"{prefix}/batch_recall": metrics["recall"],
                f"{prefix}/batch_f1": metrics["f1"],
                f"{prefix}/step": step,
            },
        )

        return probs

    def evaluate_on_dataloader(
        self, dataloader: DataLoader, prefix: str, epoch: int
    ) -> dict[str, float]:
        """
        Performs evaluation on a given dataloader (validation or test).

        Args:
            dataloader: The DataLoader to evaluate on.
            prefix: The prefix string for logging (e.g., "val", "test").
            epoch: The current epoch number (optional, mainly for validation).

        Returns:
            A dictionary containing overall loss and per-head metrics.
        """
        if not dataloader:
            self.accelerator.print(
                f"Skipping evaluation for {prefix} as dataloader is missing."
            )
            return {}

        self.model.eval()
        all_preds: dict[str, list[np.ndarray]] = {
            name: [] for name in self.model.module.head_names
        }
        all_labels: dict[str, list[np.ndarray]] = {
            name: [] for name in self.model.module.head_names
        }
        running_loss = 0.0
        results = {}

        for iteration, batch in tqdm(
            enumerate(dataloader),
            desc=f"Evaluating ({prefix.capitalize()})",
            total=len(dataloader),
        ):
            batch_loss, outputs, labels = self._evaluation_step(batch)

            batch_loss = self.accelerator.gather_for_metrics(batch_loss)
            outputs = self.accelerator.gather_for_metrics(outputs)
            labels = self.accelerator.gather_for_metrics(labels)

            running_loss += batch_loss.sum().item()

            # Log batch loss (optional, can be verbose)
            # self._log_loss(outputs, batch_loss, prefix=prefix, epoch=epoch, iteration=iteration)

            # Process each head for evaluation
            plant_mask = labels["species"] != -1
            for head_name in self.model.module.head_names:
                logits_key = f"logits_{head_name}"
                logits = outputs[logits_key]

                if head_name == "plant":
                    lbls = labels[head_name].view(-1, 1)  # Ensure 1D
                    selected_logits = logits.view(-1, 1)
                else:
                    # For other heads, only evaluate on plant samples
                    lbls = labels[head_name][plant_mask]
                    selected_logits = logits[plant_mask]

                # Evaluate batch metrics and get probabilities
                probs = self._evaluate_head_batch(
                    selected_logits,
                    lbls,
                    prefix=f"{prefix}/{head_name}",
                    step=iteration * self.accelerator.num_processes
                    + epoch * len(dataloader)
                    + self.accelerator.process_index,
                )

                # Store predictions and labels for epoch-level evaluation
                all_preds[head_name].append(probs.numpy())
                all_labels[head_name].append(lbls.cpu().numpy())

        avg_loss = running_loss / len(dataloader)
        self.accelerator.log(
            {f"{prefix}/epoch/loss": avg_loss, f"{prefix}/epoch": epoch}
        )
        self.accelerator.print(
            f"{prefix.capitalize()} Epoch: {epoch + 1}/{self.config.training.epochs}, Avg Validation Loss: {avg_loss:.4f}"
        )
        results[f"{prefix}_loss"] = avg_loss

        for head_name in self.model.module.head_names:
            y_pred = np.vstack(all_preds[head_name])
            y_true = np.concatenate(all_labels[head_name])
            y_true = y_true.ravel()  # Ensure y_true is 1D

            metrics = Evaluator.compute_metric(y_true, y_pred)
            self.accelerator.log(
                {
                    f"{prefix}/{head_name}/epoch/precision": metrics["precision"],
                    f"{prefix}/{head_name}/epoch/recall": metrics["recall"],
                    f"{prefix}/{head_name}/epoch/f1": metrics["f1"],
                    f"{prefix}/{head_name}/epoch/step": epoch,
                },
            )

            self.accelerator.print(
                f"{prefix.capitalize()} Epoch {epoch + 1}/{self.config.training.epochs}, Head: {head_name}, "
                f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
            )
            results[f"{prefix}_{head_name}_precision"] = metrics["precision"]
            results[f"{prefix}_{head_name}_recall"] = metrics["recall"]
            results[f"{prefix}_{head_name}_f1"] = metrics["f1"]

        self.model.train()
        return results
