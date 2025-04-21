from dataclasses import dataclass

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from tqdm import tqdm

import src.augmentation  # Assuming this module exists and is correct
import wandb
from src.data import DataSplit
from src.data_manager import DataManager
from src.evaluating import Evaluator


@dataclass
class ModelInfo:
    input_size: int
    mean: float
    std: float


class Trainer:
    """Handles the model training and evaluation process."""

    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        device: torch.device,
        data_manager: DataManager,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.data_manager = data_manager
        self.evaluator = Evaluator() if data_manager.val_dataloader else None
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.training.lr,
        )

    def _log_loss(
        self,
        outputs: dict[str, torch.Tensor],
        loss: torch.Tensor,
        epoch: int | None = None,
        iteration: int | None = None,
        suffix: str = "",
    ) -> None:
        """Logs loss components to wandb and console."""
        log_data = {
            f"{suffix}/loss": loss.item(),
            f"{suffix}/epoch": epoch,
            f"{suffix}/iter": iteration,
        }
        print_str = f"{suffix.capitalize()} Iter: {iteration}, Loss: {loss.item():.4f}"
        for head in self.model.head_names:
            loss_key = f"loss_{head}"
            if loss_key in outputs:
                log_data[f"{suffix}/{loss_key}"] = outputs[loss_key].item()
                print_str += (
                    f", Loss {head.capitalize()}: {outputs[loss_key].item():.4f}"
                )

        wandb.log(log_data)
        print(print_str)

    def _calculate_total_loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculates the weighted total loss."""
        total_loss = torch.tensor(0.0, device=self.device)
        weights = self.config.training.loss_weights
        for head in self.model.head_names:
            loss_key = f"loss_{head}"
            if loss_key in outputs:
                weight = getattr(
                    weights, head, 0.0
                )  # Get weight, default to 0 if not specified
                total_loss += weight * outputs[loss_key]
        return total_loss

    def _train_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Performs a single training step."""
        images, species_labels, images_names, plant_labels = batch
        images = images.to(self.device)
        species_labels = species_labels.to(self.device)
        plant_labels = plant_labels.to(self.device)

        # Apply augmentation
        augmentation = src.augmentation.get_random_data_augmentation(self.config)
        images = augmentation(images)

        # Gather labels
        labels = self.data_manager.gather_all_labels(
            species_labels, plant_labels, images_names, self.device
        )

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(
            pixel_values=images, labels=labels, plant_mask=labels["plant"] == 1
        )

        # Calculate loss
        loss = self._calculate_total_loss(outputs)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss, outputs

    def _validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Performs a single validation step."""
        images, species_labels, images_names, plant_labels = batch
        images = images.to(self.device)
        species_labels = species_labels.to(self.device)
        plant_labels = plant_labels.to(self.device)

        # Gather labels
        labels = self.data_manager.gather_all_labels(
            species_labels, plant_labels, images_names, self.device
        )

        # Forward pass (no gradients)
        with torch.no_grad():
            outputs = self.model(
                pixel_values=images, labels=labels, plant_mask=labels["plant"] == 1
            )
            loss = self._calculate_total_loss(outputs)

        return loss, outputs, labels

    def _evaluate_head_batch(
        self, logits: torch.Tensor, labels: torch.Tensor, suffix: str
    ) -> torch.Tensor:
        probs = torch.nn.functional.softmax(logits, dim=1).cpu()
        labels = labels.view(-1).cpu()  # Ensure labels are 1D for evaluator

        metrics = self.evaluator.evaluate(labels, probs)  # type: ignore
        wandb.log(
            {
                f"{suffix}/batch_precision": metrics["precision"],
                f"{suffix}/batch_recall": metrics["recall"],
                f"{suffix}/batch_f1": metrics["f1"],
            }
        )

        return probs

    def _evaluate_epoch(self, epoch: int, train_iter: int) -> None:
        """Performs evaluation for one epoch."""
        if not self.evaluator or not self.data_manager.val_dataloader:
            return

        self.model.eval()
        all_preds: dict[str, list[np.ndarray]] = {
            name: [] for name in self.model.head_names
        }
        all_labels: dict[str, list[np.ndarray]] = {
            name: [] for name in self.model.head_names
        }
        running_val_loss = 0.0

        for val_iteration, val_batch in tqdm(
            enumerate(self.data_manager.val_dataloader),
            desc="Validating",
            total=len(self.data_manager.val_dataloader),
        ):
            val_loss, outputs_val, labels = self._validation_step(val_batch)
            running_val_loss += val_loss.item()

            # Log batch validation loss
            self._log_loss(
                outputs_val,
                val_loss,
                epoch=epoch,
                iteration=val_iteration,
                suffix="val",
            )

            # Process each head for evaluation
            plant_mask = labels["plant"] == 1
            for head_name in self.model.head_names:
                logits_key = f"logits_{head_name}"
                logits = outputs_val[logits_key]

                # Select appropriate labels and logits based on head type
                if head_name == "plant":
                    lbls = labels[head_name].view(
                        -1, 1
                    )  # Plant labels apply to all samples
                    selected_logits = logits.view(-1, 1)
                else:
                    # For other heads, only evaluate on plant samples
                    lbls = labels[head_name][plant_mask]
                    selected_logits = logits[plant_mask]

                # Evaluate batch metrics and get probabilities
                probs = self._evaluate_head_batch(
                    selected_logits, lbls, suffix=f"val/{head_name}"
                )

                # Store predictions and labels for epoch-level evaluation
                all_preds[head_name].append(probs.numpy())
                all_labels[head_name].append(lbls.cpu().numpy())

        # Calculate and log epoch-level metrics
        avg_val_loss = running_val_loss / len(self.data_manager.val_dataloader)
        wandb.log({"val/epoch_loss": avg_val_loss, "epoch": epoch})
        print(
            f"Epoch: {epoch + 1}/{self.config.training.epochs}, Avg Validation Loss: {avg_val_loss:.4f}"
        )

        for head_name in self.model.head_names:
            y_pred = np.vstack(all_preds[head_name])
            y_true = np.concatenate(all_labels[head_name])
            y_true = y_true.ravel()  # Ensure y_true is 1D

            metrics = self.evaluator.evaluate(y_true, y_pred)
            wandb.log(
                {
                    f"val/{head_name}/epoch_precision": metrics["precision"],
                    f"val/{head_name}/epoch_recall": metrics["recall"],
                    f"val/{head_name}/epoch_f1": metrics["f1"],
                    "epoch": epoch,
                }
            )
            print(
                f"Val Epoch {epoch + 1} Head: {head_name}, Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
            )

        self.model.train()

    def train(self) -> tuple[nn.Module, ModelInfo]:
        """Runs the main training loop."""
        self.model.train()

        for epoch in range(self.config.training.epochs):
            running_loss = 0.0
            for iteration, batch in tqdm(
                enumerate(self.data_manager.train_dataloader),
                desc=f"Epoch {epoch + 1}/{self.config.training.epochs} Training",
                total=len(self.data_manager.train_dataloader),
            ):
                loss, outputs = self._train_step(batch)
                running_loss += loss.item()

                # Log training step loss
                self._log_loss(
                    outputs, loss, epoch=epoch, iteration=iteration, suffix="train"
                )

                # Perform validation periodically
                if (
                    self.evaluator is not None
                    and (iteration + 1) % self.config.evaluating.every == 0
                ):
                    self._evaluate_epoch(epoch=epoch, train_iter=iteration)

            avg_epoch_loss = running_loss / len(self.data_manager.train_dataloader)
            wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch})
            print(
                f"Epoch: {epoch + 1}/{self.config.training.epochs}, Avg Training Loss: {avg_epoch_loss:.4f}"
            )

            # Run validation at the end of each epoch if not done periodically
            if self.evaluator is not None and self.config.evaluating.every > len(
                self.data_manager.train_dataloader
            ):
                self._evaluate_epoch(
                    epoch=epoch, train_iter=len(self.data_manager.train_dataloader) - 1
                )

        data_config = timm.data.resolve_model_data_config(self.model)

        model_info = ModelInfo(
            input_size=data_config["input_size"][1],  # Assuming (C, H, W)
            mean=data_config["mean"],
            std=data_config["std"],
        )

        print(f"Model info: {model_info}")

        return self.model, model_info


def train(
    model: nn.Module,
    config: DictConfig,
    device: torch.device,
    df_metadata: pd.DataFrame,
    plant_data_split: DataSplit | None = None,
    non_plant_data_split: DataSplit | None = None,
) -> tuple[torch.nn.Module, ModelInfo]:
    """
    Main function to set up and run the training process.

    Args:
        model: The neural network model.
        config: Configuration object (OmegaConf).
        device: The device to train on (e.g., 'cuda', 'cpu').
        df_metadata: DataFrame containing metadata for images.
        plant_data_split: Optional DataSplit object for plant data.
        non_plant_data_split: Optional DataSplit object for non-plant data.

    Returns:
        A tuple containing the trained model and model information.
    """

    # Initialize data management
    data_manager = DataManager(
        config=config,
        plant_data_split=plant_data_split,
        non_plant_data_split=non_plant_data_split,
        df_metadata=df_metadata,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        data_manager=data_manager,
    )

    # Run training
    trained_model, model_info = trainer.train()

    return trained_model, model_info
