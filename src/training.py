import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import src.augmentation  # Assuming this module exists and is correct
import wandb
from src.data import ConcatenatedDataset, DataSplit, NonPlantDataset, PlantDataset
from src.evaluating import Evaluator
from src.utils import (
    family_name_to_id,
    genus_name_to_id,
    image_path_to_organ_name,
    organ_name_to_id,
    species_id_to_name,
    species_name_to_new_id,
)
from utils.build_hierarchies import (
    check_utils_folder,
    get_genus_family_from_species,
    read_plant_taxonomy,
)


@dataclass
class ModelInfo:
    input_size: int
    mean: float
    std: float


@dataclass
class DataManager:
    config: DictConfig
    plant_data_split: DataSplit | None
    non_plant_data_split: DataSplit | None
    df_metadata: pd.DataFrame

    def __post_init__(self) -> None:
        self._load_mappings_and_taxonomy()
        self._setup_datasets()
        self._setup_dataloaders()

    def _load_mappings_and_taxonomy(self) -> None:
        """Loads taxonomic tree and mapping files."""
        self.plant_tree = read_plant_taxonomy(self.config)
        utils_folder = check_utils_folder(self.config)

        def read_mapping(mapping_file: str) -> pd.DataFrame:
            return pd.read_csv(os.path.join(utils_folder, mapping_file))

        self.species_mapping = read_mapping(self.config.data.utils.species_mapping)
        self.genus_mapping = read_mapping(self.config.data.utils.genus_mapping)
        self.family_mapping = read_mapping(self.config.data.utils.family_mapping)
        self.organ_mapping = read_mapping(self.config.data.utils.organ_mapping)

    def _create_dataset(self, split_type: str) -> Dataset:
        """Creates a concatenated dataset for a given split ('train' or 'val')."""
        plant_indices = (
            getattr(self.plant_data_split, f"{split_type}_indices", None)
            if self.plant_data_split
            else None
        )
        non_plant_indices = (
            getattr(self.non_plant_data_split, f"{split_type}_indices", None)
            if self.non_plant_data_split
            else None
        )

        image_folder_train = os.path.join(
            self.config.project_path,
            self.config.data.folder,
            self.config.data.train_folder,
        )
        image_folder_other = os.path.join(
            self.config.project_path,
            self.config.data.folder,
            self.config.data.other.folder,
        )
        image_size = (self.config.image_width, self.config.image_height)

        datasets_to_concat = []
        if (
            plant_indices is not None or split_type == "train"
        ):  # Always include plants for training if no split provided
            datasets_to_concat.append(
                PlantDataset(
                    image_folder=image_folder_train,
                    image_size=image_size,
                    indices=plant_indices,
                )
            )
        if (
            non_plant_indices is not None or split_type == "train"
        ):  # Always include non-plants for training if no split provided
            datasets_to_concat.append(
                NonPlantDataset(
                    image_folder=image_folder_other,
                    image_size=image_size,
                    indices=non_plant_indices,
                )
            )

        if not datasets_to_concat:
            raise ValueError(f"No data available for split type: {split_type}")

        return ConcatenatedDataset(datasets_to_concat)

    def _setup_datasets(self) -> None:
        """Sets up train and validation datasets."""
        self.train_dataset = self._create_dataset("train")
        if self.plant_data_split is not None:
            self.val_dataset = self._create_dataset("val")
        else:
            self.val_dataset = None

    def _setup_dataloaders(self) -> None:
        """Sets up train and validation dataloaders."""
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.training.shuffle,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
        )
        if self.val_dataset:
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.config.evaluating.batch_size,
                shuffle=self.config.evaluating.shuffle,
                num_workers=self.config.evaluating.num_workers,
                pin_memory=True,
            )
        else:
            self.val_dataloader = None

    def gather_all_labels(
        self,
        species_labels: torch.Tensor,
        plant_labels: torch.Tensor,
        images_names: list[str],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """
        Gathers and transforms labels for species, genus, family, and organ
        based on the initial species and plant labels.
        """
        labels = {"plant": plant_labels}
        batch_size = len(species_labels)
        new_species_ids = torch.full((batch_size,), -1, dtype=torch.long)
        genus_ids = torch.full((batch_size,), -1, dtype=torch.long)
        family_ids = torch.full((batch_size,), -1, dtype=torch.long)
        organ_ids = torch.full((batch_size,), -1, dtype=torch.long)

        # Process only plant samples (where species_id != -1)
        plant_mask = species_labels != -1
        plant_indices = torch.where(plant_mask)[0]

        for idx in plant_indices:
            i = idx.item()  # Get the actual index in the batch
            species_id = species_labels[i].item()
            species_name = species_id_to_name(species_id, self.species_mapping)
            genus_name, family_name = get_genus_family_from_species(
                self.plant_tree, species_name
            )
            image_name = images_names[i]
            organ_name = image_path_to_organ_name(image_name, self.df_metadata)

            new_species_ids[i] = species_name_to_new_id(
                species_name, self.species_mapping
            )
            genus_ids[i] = genus_name_to_id(genus_name, self.genus_mapping)
            family_ids[i] = family_name_to_id(family_name, self.family_mapping)
            organ_ids[i] = organ_name_to_id(organ_name, self.organ_mapping)

        labels["species"] = new_species_ids.to(device)
        labels["genus"] = genus_ids.to(device)
        labels["family"] = family_ids.to(device)
        labels["organ"] = organ_ids.to(device)
        return labels


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
