import os
from dataclasses import dataclass

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from omegaconf import DictConfig
from tqdm import tqdm

import src.augmentation
from src.data import DataSplit
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
        data_manager: DataManager,
        accelerator: Accelerator,
        evaluator: Evaluator | None = None,
    ):
        self.model = model
        self.config = config
        self.accelerator = accelerator
        self.data_manager = data_manager
        self.evaluator = evaluator
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.training.lr,
        )

        (
            self.optimizer,
            self.data_manager.train_dataloader,
            self.data_manager.val_dataloader,
            self.data_manager.test_dataloader,
        ) = self.accelerator.prepare(
            self.optimizer,
            self.data_manager.train_dataloader,
            self.data_manager.val_dataloader,
            self.data_manager.test_dataloader,
        )

    def _log_loss(
        self,
        outputs: dict[str, torch.Tensor],
        loss: torch.Tensor,
        prefix: str,
        step: int,
    ) -> None:
        """Logs loss components to wandb and console."""
        log_data = {
            f"{prefix}/loss": loss.item(),
        }

        print_str = f"{prefix.capitalize()} Loss: {loss.item():.4f}"

        for head in self.model.module.head_names:
            loss_key = f"loss_{head}"
            if loss_key in outputs:
                log_data[f"{prefix}/{loss_key}"] = outputs[loss_key].item()
                print_str += (
                    f", Loss {head.capitalize()}: {outputs[loss_key].item():.4f}"
                )

        log_data[f"{prefix}/step"] = step
        self.accelerator.log(log_data)
        self.accelerator.print(print_str)

    def _train_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Performs a single training step."""
        images, species_labels, images_names = batch
        plant_labels = (species_labels != -1).clone().detach().to(dtype=torch.float32)

        # Apply augmentation
        augmentation = src.augmentation.get_random_data_augmentation(self.config)
        images = augmentation(images)

        # Gather labels
        labels = self.data_manager.gather_all_labels(
            species_labels, plant_labels, images_names
        )

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(
            pixel_values=images, labels=labels, plant_mask=labels["plant"] == 1
        )

        # Calculate loss
        loss = calculate_total_loss(outputs, self.model.module.head_names, self.config)

        # Backward pass and optimization
        self.accelerator.backward(loss)
        self.optimizer.step()

        return loss, outputs

    def train(self) -> nn.Module:
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
                    outputs,
                    loss,
                    prefix="train",
                    step=iteration * self.accelerator.num_processes
                    + epoch * len(self.data_manager.train_dataloader)
                    + self.accelerator.process_index,
                )

                # Save model periodically
                if (iteration + 1) % self.config.models.save.every == 0:
                    save_folder = os.path.join(
                        self.config.models.folder,
                        self.config.models.save.folder,
                        "checkpoints",
                    )
                    self.accelerator.save_state(save_folder)
                    self.accelerator.print(
                        f"Checkpoint saved at iteration {iteration + 1} of epoch {epoch + 1} to {save_folder}"
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
            self.accelerator.log(
                {"train/epoch/loss": avg_epoch_loss, "train/epoch/step": epoch},
            )
            self.accelerator.print(
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

        return self.model


def train(
    model: nn.Module,
    config: DictConfig,
    df_metadata: pd.DataFrame,
    accelerator: Accelerator,
    plant_data_split: DataSplit | None = None,
    non_plant_data_split: DataSplit | None = None,
) -> tuple[torch.nn.Module, ModelInfo]:
    """
    Main function to set up and run the training and final testing process.

    Args:
        model: The neural network model.
        config: Configuration object (OmegaConf).
        accelerator: Accelerator object for distributed training.
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

    model = accelerator.prepare(model)

    evaluator = Evaluator(data_manager, model, config, accelerator)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config,
        data_manager=data_manager,
        accelerator=accelerator,
        evaluator=evaluator if data_manager.val_dataloader else None,
    )

    # Run training
    accelerator.print("Starting training...")
    trained_model = trainer.train()
    accelerator.print("Training finished.")

    accelerator.print("Saving the trained model...")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(trained_model)
    save_path = os.path.join(
        config.project_path,
        config.models.save.folder,
    )
    accelerator.save_model(
        unwrapped_model,
        save_path,
    )
    accelerator.print(f"Model saved to {save_path}")

    # # Run final evaluation on the test set
    # accelerator.print("Starting testing...")
    # test_results = evaluator.evaluate_on_dataloader(
    #     dataloader=data_manager.test_dataloader,
    #     prefix="test",
    #     epoch=0,
    # )
    # accelerator.print("Testing finished.")
    # accelerator.print("Final Test Results:")
    # for key, value in test_results.items():
    #     accelerator.print(f"- {key}: {value:.4f}")

    data_config = timm.data.resolve_model_data_config(unwrapped_model)
    model_info = ModelInfo(
        input_size=data_config["input_size"][1],  # Assuming (C, H, W)
        mean=data_config["mean"],
        std=data_config["std"],
    )
    accelerator.print(f"Model info: {model_info}")

    return unwrapped_model, model_info
