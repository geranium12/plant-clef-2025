import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from omegaconf import DictConfig
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_manager import DataManager
from src.utils import calculate_total_loss, log_loss


class Evaluator:
    def __init__(self, accelerator: Accelerator) -> None:
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
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        data_manager: DataManager,
        model: nn.Module,
        config: DictConfig,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Performs a single evaluation (validation or test) step."""
        images, species_labels, images_names = batch
        plant_labels = (species_labels != -1).clone().detach().to(dtype=torch.float32)

        # Gather labels
        labels = data_manager.gather_all_labels(
            species_labels, plant_labels, images_names
        )

        # Forward pass (no gradients)
        with torch.no_grad():
            outputs = model(
                pixel_values=images, labels=labels, plant_mask=labels["plant"] == 1
            )

            loss = calculate_total_loss(
                outputs=outputs,
                head_names=model.module.head_names
                if self.accelerator.num_processes > 1
                else model.head_names,
                config=config,
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
                f"{prefix}/batch/precision": metrics["precision"],
                f"{prefix}/batch/recall": metrics["recall"],
                f"{prefix}/batch/f1": metrics["f1"],
                f"{prefix}/batch/step": step,
            },
        )

        return probs

    def evaluate_on_dataloader(
        self,
        model: nn.Module,
        config: DictConfig,
        data_manager: DataManager,
        dataloader: DataLoader,
        prefix: str,
        call_time: int,
    ) -> dict[str, float]:
        """
        Performs evaluation on a given dataloader (validation or test).

        Args:
            model: The model to evaluate.
            config: The configuration object.
            data_manager: The data manager for gathering labels.
            dataloader: The DataLoader to evaluate on.
            prefix: The prefix string for logging (e.g., "val", "test").
            call_time: The current call time for this evaluation so that the log is shown nicely on wandb.

        Returns:
            A dictionary containing overall loss and per-head metrics.
        """
        if not dataloader:
            self.accelerator.print(
                f"Skipping evaluation for {prefix} as dataloader is missing."
            )
            return {}

        model.eval()

        head_names = (
            model.module.head_names
            if self.accelerator.num_processes > 1
            else model.head_names
        )
        all_preds: dict[str, list[np.ndarray]] = {name: [] for name in head_names}
        all_labels: dict[str, list[np.ndarray]] = {name: [] for name in head_names}
        running_loss = 0.0
        results = {}

        for iteration, batch in tqdm(
            enumerate(dataloader),
            desc=f"Evaluating ({prefix.capitalize()})",
            total=len(dataloader),
        ):
            batch_loss, outputs, labels = self._evaluation_step(
                batch, data_manager, model, config
            )

            batch_loss = self.accelerator.gather_for_metrics(batch_loss)
            outputs = self.accelerator.gather_for_metrics(outputs)
            labels = self.accelerator.gather_for_metrics(labels)

            running_loss += batch_loss.sum().item()

            batch_loss = batch_loss.mean()
            for head_name in head_names:
                loss_key = f"loss_{head_name}"
                if loss_key in outputs:
                    outputs[loss_key] = outputs[loss_key].mean()

            log_loss(
                outputs=outputs,
                loss=batch_loss.sum(),
                prefix=prefix,
                head_names=head_names,
                accelerator=self.accelerator,
                step=iteration + call_time * len(dataloader),
            )

            # Process each head for evaluation
            plant_mask = labels["species"] != -1
            for head_name in head_names:
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
                    step=iteration + call_time * len(dataloader),
                )

                # Store predictions and labels for epoch-level evaluation
                all_preds[head_name].append(probs.numpy())
                all_labels[head_name].append(lbls.cpu().numpy())

        avg_loss = running_loss / len(dataloader)
        self.accelerator.log(
            {f"{prefix}/avg_loss": avg_loss, f"{prefix}/step": call_time}
        )
        self.accelerator.print(
            f"{prefix.capitalize()} Call Time: {call_time}, Avg Validation Loss: {avg_loss:.4f}"
        )
        results[f"{prefix}_avg_loss"] = avg_loss

        for head_name in head_names:
            y_pred = np.vstack(all_preds[head_name])
            y_true = np.concatenate(all_labels[head_name])
            y_true = y_true.ravel()  # Ensure y_true is 1D

            metrics = Evaluator.compute_metric(y_true, y_pred)
            self.accelerator.log(
                {
                    f"{prefix}/{head_name}/precision": metrics["precision"],
                    f"{prefix}/{head_name}/recall": metrics["recall"],
                    f"{prefix}/{head_name}/f1": metrics["f1"],
                    f"{prefix}/{head_name}/step": call_time,
                },
            )

            self.accelerator.print(
                f"{prefix.capitalize()} Call Time {call_time}, Head: {head_name}, "
                f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
            )
            results[f"{prefix}_{head_name}_precision"] = metrics["precision"]
            results[f"{prefix}_{head_name}_recall"] = metrics["recall"]
            results[f"{prefix}_{head_name}_f1"] = metrics["f1"]

        model.train()
        return results
