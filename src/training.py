import os
from dataclasses import (
    dataclass,
)

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import (
    DictConfig,
)
from pandas import pd
from torch.utils.data import DataLoader

import src.augmentation
from build_hierarchies import get_taxonomy_from_species, read_plant_taxonomy
from src.data import TrainDataset


@dataclass
class ModelInfo:
    input_size: int
    mean: float
    std: float


def train(
    model: nn.Module,
    config: DictConfig,
    device: torch.device,
    df_species_ids: pd.DataFrame,
    train_indices: list[int],
    val_indices: list[int],
) -> tuple[torch.nn.Module, ModelInfo, float]:
    plant_tree = read_plant_taxonomy(config)
    model = timm.create_model(
        config.models.name,
        pretrained=config.models.pretrained,
        num_classes=len(df_species_ids),
        checkpoint_path=os.path.join(
            config.project_path, config.models.folder, config.models.checkpoint_file
        ),
    )
    model = model.to(device)
    model = model.eval()

    for augmentation_name in config.training.augmentations:
        augmentation = src.augmentation.get_data_augmentation(config, augmentation_name)

        train_dataset = TrainDataset(
            image_folder=os.path.join(
                config.project_path,
                config.data.folder,
                config.data.train_folder,
            ),
            image_size=(config.image_width, config.image_height),
            transform=augmentation,
            indices=train_indices,
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.training.batch_size,
            shuffle=config.training.shuffle,
            num_workers=config.training.num_workers,
        )

        model.to(device)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=config.training.lr
        )

        model.train()
        for epoch in range(config.training.epochs):
            running_loss = 0.0
            for batch in train_dataloader:
                image, species_labels = batch["image"].to(device)
                labels = {}
                labels["species"] = species_labels

                genus_labels = []
                family_labels = []
                for species in species_labels:
                    genus, family = get_taxonomy_from_species(plant_tree, species)
                    genus_labels.append(genus)
                    family_labels.append(family)

                genus_labels = torch.Tensor(genus_labels).to(device)
                family_labels = torch.Tensor(family_labels).to(device)
                labels["genus"] = genus_labels
                labels["family"] = family_labels

                optimizer.zero_grad()
                outputs = model(image, labels=labels)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_dataloader)
            print(f"Epoch [{epoch + 1}/{config.training.epochs}], Loss: {avg_loss:.4f}")

    data_config = timm.data.resolve_model_data_config(model)
    model_info = ModelInfo(
        data_config["input_size"][1],
        data_config["mean"],
        data_config["std"],
    )

    return model, model_info, avg_loss
