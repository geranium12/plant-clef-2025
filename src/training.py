import os
from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from omegaconf import DictConfig
from tqdm import tqdm

import src.augmentation
from src.data_manager import DataManager
from src.evaluating import Evaluator
from src.utils import calculate_total_loss, log_loss


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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.training.scheduler.factor,
            patience=self.config.training.scheduler.patience,
            verbose=True,
        )
        self.optimizer, self.scheduler = accelerator.prepare(
            self.optimizer, self.scheduler
        )

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
        loss = calculate_total_loss(
            outputs,
            self.model.module.head_names
            if isinstance(self.model, nn.parallel.DistributedDataParallel)
            else self.model.head_names,
            self.config,
        )

        # Backward pass and optimization
        self.accelerator.backward(loss)
        self.optimizer.step()

        return loss, outputs

    def train(self) -> nn.Module:
        """Runs the main training loop."""
        self.model.train()
        head_names = (
            self.model.module.head_names
            if isinstance(self.model, nn.parallel.DistributedDataParallel)
            else self.model.head_names
        )
        for epoch in range(self.config.training.epochs):
            running_loss = 0.0
            for iteration, batch in tqdm(
                enumerate(self.data_manager.train_dataloader),
                desc=f"Epoch {epoch + 1}/{self.config.training.epochs} Training",
                total=len(self.data_manager.train_dataloader),
            ):
                loss, outputs = self._train_step(batch)
                running_loss += loss.item()

                log_loss(
                    outputs=outputs,
                    loss=loss,
                    head_names=head_names,
                    accelerator=self.accelerator,
                    prefix="train",
                    step=iteration + epoch * len(self.data_manager.train_dataloader),
                )

                self.accelerator.log(
                    {
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/step": iteration
                        + epoch * len(self.data_manager.train_dataloader),
                    }
                )

                # Save model periodically
                if (iteration + 1) % self.config.models.save.every == 0:
                    save_folder = os.path.join(
                        self.config.project_path,
                        self.config.models.folder,
                        self.config.models.save.folder,
                        f"checkpoint_ep{epoch + 1}_it{iteration + 1}",
                    )
                    self.accelerator.save_state(save_folder)
                    self.accelerator.print(
                        f"Checkpoint saved at iteration {iteration + 1} of epoch {epoch + 1} to {save_folder}"
                    )

                # Perform validation periodically
                if self.evaluator is not None:
                    if (iteration + 1) % self.config.evaluating.every == 0:
                        val_metrics = self.evaluator.evaluate_on_dataloader(
                            model=self.model,
                            config=self.config,
                            data_manager=self.data_manager,
                            dataloader=self.data_manager.val_dataloader,
                            prefix="val",
                            call_time=int(iteration / self.config.evaluating.every),
                        )

                        # Step the scheduler based on validation loss
                        self.scheduler.step(val_metrics["val_avg_loss"])
                else:
                    # If no validation is performed, step based on training loss
                    self.scheduler.step(loss.item())

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
                    model=self.model,
                    config=self.config,
                    data_manager=self.data_manager,
                    dataloader=self.data_manager.val_dataloader,
                    prefix="val",
                    call_time=epoch,
                )

        return self.model


def train(
    model: nn.Module,
    data_manager: DataManager,
    config: DictConfig,
    accelerator: Accelerator,
) -> tuple[torch.nn.Module, ModelInfo]:
    """
    Main function to set up and run the training and final testing process.

    Args:
        model: The neural network model.
        data_manager: DataManager object to handle data loading and processing.
        config: Configuration object (OmegaConf).
        accelerator: Accelerator object for distributed training.

    Returns:
        A tuple containing the trained model and model information.
    """

    model = accelerator.prepare(model)
    (
        data_manager.train_dataloader,
        data_manager.val_dataloader,
        data_manager.test_dataloader,
    ) = accelerator.prepare(
        data_manager.train_dataloader,
        data_manager.val_dataloader,
        data_manager.test_dataloader,
    )

    evaluator = Evaluator(accelerator)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config,
        data_manager=data_manager,
        accelerator=accelerator,
        evaluator=evaluator if data_manager.val_dataloader else None,
    )

    unwrapped_model = None
    if config.training.enabled:
        # Run training
        accelerator.print("Starting training...")
        trained_model = trainer.train()
        accelerator.print("Training finished.")

        accelerator.print("Saving the trained model...")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(trained_model)
        save_path = os.path.join(
            config.project_path,
            config.models.folder,
            config.models.save.folder,
            "final_model",
        )
        accelerator.save_model(
            unwrapped_model,
            save_path,
        )
        accelerator.print(f"Model saved to {save_path}")

    if config.evaluation.test_enabled:
        # Run final evaluation on the test set
        accelerator.print("Starting testing...")
        test_results = evaluator.evaluate_on_dataloader(
            model=model,
            config=config,
            data_manager=data_manager,
            dataloader=data_manager.test_dataloader,
            prefix="test",
            call_time=0,
        )
        accelerator.print("Testing finished.")
        accelerator.print("Final Test Results:")
        for key, value in test_results.items():
            accelerator.print(f"- {key}: {value:.4f}")

    data_config = timm.data.resolve_model_data_config(
        unwrapped_model if unwrapped_model else model
    )
    model_info = ModelInfo(
        input_size=data_config["input_size"][1],  # Assuming (C, H, W)
        mean=data_config["mean"],
        std=data_config["std"],
    )
    accelerator.print(f"Model info: {model_info}")

    return unwrapped_model, model_info
