from dataclasses import dataclass

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from tqdm import tqdm

import src.augmentation
import wandb
from src.data import DataSplit, ImageSampleInfo
from src.data_manager import DataManager
from src.evaluating import Evaluator
from src.utils import calculate_total_loss


@dataclass
class ModelInfo:
    input_size: int
    mean: float
    std: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        device: torch.device,
        data_manager: DataManager,
        evaluator: Evaluator | None = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.data_manager = data_manager
        self.evaluator = evaluator
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.training.lr,
        )

    def _log_loss(
        self,
        outputs: dict[str, torch.Tensor],
        loss: torch.Tensor,
        prefix: str,
        epoch: int,
        iteration: int,
    ) -> None:
        """Logs loss components to wandb and console."""
        log_data = {
            f"{prefix}/loss": loss.item(),
            f"{prefix}/epoch": epoch,
            f"{prefix}/iter": iteration,
        }

        print_str = f"{prefix.capitalize()} Loss: {loss.item():.4f}"

        for head in self.model.head_names:
            loss_key = f"loss_{head}"
            if loss_key in outputs:
                log_data[f"{prefix}/{loss_key}"] = outputs[loss_key].item()
                print_str += (
                    f", Loss {head.capitalize()}: {outputs[loss_key].item():.4f}"
                )

        wandb.log(log_data)
        print(print_str)

    def _train_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Performs a single training step."""
        images, species_labels, images_names = batch
        images = images.to(self.device)
        species_labels = species_labels.to(self.device)
        plant_labels = (
            (species_labels != -1)
            .clone()
            .detach()
            .to(dtype=torch.float32, device=self.device)
        )

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
        loss = calculate_total_loss(
            outputs, self.model.head_names, self.config, self.device
        )

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss, outputs

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

                self._log_loss(
                    outputs, loss, prefix="train", epoch=epoch, iteration=iteration
                )

                # Perform validation periodically
                if (
                    self.evaluator is not None
                    and (iteration + 1) % self.config.evaluating.every == 0
                ):
                    _ = self.evaluator.evaluate_on_dataloader(
                        dataloader=self.data_manager.val_dataloader,
                        prefix="val",
                        epoch=epoch,
                    )

            avg_epoch_loss = running_loss / len(self.data_manager.train_dataloader)
            wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch})
            print(
                f"Epoch: {epoch + 1}/{self.config.training.epochs}, Avg Training Loss: {avg_epoch_loss:.4f}"
            )

            # Run validation at the end of each epoch if not done periodically
            if self.evaluator is not None and self.config.evaluating.every > len(
                self.data_manager.train_dataloader
            ):
                _ = self.evaluator.evaluate_on_dataloader(
                    dataloader=self.data_manager.val_dataloader,
                    prefix="val",
                    epoch=epoch,
                )

        # --- Training Loop Finished ---

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
    plant_data_image_info: list[ImageSampleInfo],
    plant_data_split: DataSplit | None = None,
    non_plant_data_split: DataSplit | None = None,
) -> tuple[torch.nn.Module, ModelInfo]:
    """
    Main function to set up and run the training and final testing process.

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
        plant_data_image_info=plant_data_image_info,
        plant_data_split=plant_data_split,
        non_plant_data_split=non_plant_data_split,
        df_metadata=df_metadata,
    )

    evaluator = Evaluator(data_manager, model, config, device)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        data_manager=data_manager,
        evaluator=evaluator if data_manager.val_dataloader else None,
    )

    # Run training
    print("Starting training...")
    trained_model, model_info = trainer.train()
    print("Training finished.")

    # Run final evaluation on the test set
    print("Starting testing...")
    test_results = evaluator.evaluate_on_dataloader(
        dataloader=data_manager.test_dataloader,
        prefix="test",
        epoch=0,
    )
    print("Testing finished.")
    print("Final Test Results:")
    for key, value in test_results.items():
        print(f"- {key}: {value:.4f}")

    return trained_model, model_info
