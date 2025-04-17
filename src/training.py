import os
from dataclasses import (
    dataclass,
)
from typing import Optional

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import (
    DictConfig,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.augmentation
import wandb
from build_hierarchies import (
    check_utils_folder,
    get_taxonomy_from_species,
    read_plant_taxonomy,
)
from src.data import ConcatenatedDataset, DataSplit, TrainDataset, UnlabeledDataset
from src.utils import (
    family_name_to_id,
    genus_name_to_id,
    image_path_to_organ_name,
    organ_name_to_id,
    species_id_to_name,
    species_name_to_new_id,
)


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
    df_metadata: pd.DataFrame,
    labeled_data_split: Optional[DataSplit] = None,
    unlabeled_data_split: Optional[DataSplit] = None,
) -> tuple[torch.nn.Module, ModelInfo]:
    plant_tree = read_plant_taxonomy(config)
    folder_name = check_utils_folder(config)
    species_mapping = pd.read_csv(
        os.path.join(folder_name, config.data.utils.species_mapping)
    )
    genus_mapping = pd.read_csv(
        os.path.join(folder_name, config.data.utils.genus_mapping)
    )
    family_mapping = pd.read_csv(
        os.path.join(folder_name, config.data.utils.family_mapping)
    )
    organ_mapping = pd.read_csv(
        os.path.join(folder_name, config.data.utils.organ_mapping)
    )

    model.to(device)
    model.train()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.training.lr
    )

    for augmentation_name in config.training.augmentations:
        augmentation = src.augmentation.get_data_augmentation(config, augmentation_name)
        train_dataset = ConcatenatedDataset(
            [
                TrainDataset(
                    image_folder=os.path.join(
                        config.project_path,
                        config.data.folder,
                        config.data.train_folder,
                    ),
                    image_size=(config.image_width, config.image_height),
                    transform=augmentation,
                    indices=(
                        labeled_data_split.train_indices if labeled_data_split else None
                    ),
                ),
                UnlabeledDataset(
                    image_folder=os.path.join(
                        config.project_path,
                        config.data.folder,
                        config.data.other.folder,
                    ),
                    image_size=(config.image_width, config.image_height),
                    transform=augmentation,
                    indices=unlabeled_data_split.train_indices
                    if unlabeled_data_split
                    else None,
                ),
            ]
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.training.batch_size,
            shuffle=config.training.shuffle,
            num_workers=config.training.num_workers,
            pin_memory=True,
        )

        for epoch in range(config.training.epochs):
            running_loss = 0.0
            for iteration, batch in tqdm(
                enumerate(train_dataloader),
                desc="Training",
                total=len(train_dataloader),
            ):
                images, species_labels, images_names, plant_labels = batch
                images = images.to(device)
                species_labels = species_labels.to(device)
                plant_labels = plant_labels.to(device)
                labels = {}
                labels["plant"] = plant_labels
                plant_mask = plant_labels == 1

                new_species_labels = []
                genus_labels = []
                family_labels = []
                organ_labels = []
                for i, species_id in enumerate(species_labels):
                    if species_id is None:
                        new_species_labels.append(-1)
                        genus_labels.append(-1)
                        family_labels.append(-1)
                        organ_labels.append(-1)
                    else:
                        species_name = species_id_to_name(
                            species_id.item(), species_mapping
                        )
                        genus_name, family_name = get_taxonomy_from_species(
                            plant_tree, species_name
                        )
                        new_species_id = species_name_to_new_id(
                            species_name, species_mapping
                        )
                        genus_id = genus_name_to_id(genus_name, genus_mapping)
                        family_id = family_name_to_id(family_name, family_mapping)
                        new_species_labels.append(new_species_id)
                        genus_labels.append(genus_id)
                        family_labels.append(family_id)

                        image_name = images_names[i]
                        organ_name = image_path_to_organ_name(image_name, df_metadata)
                        organ_id = organ_name_to_id(organ_name, organ_mapping)
                        organ_labels.append(organ_id)

                new_species_labels = torch.tensor(new_species_labels).to(device)
                genus_labels = torch.tensor(genus_labels).to(device)
                family_labels = torch.tensor(family_labels).to(device)
                organ_labels = torch.tensor(organ_labels).to(device)
                labels["species"] = new_species_labels
                labels["genus"] = genus_labels
                labels["family"] = family_labels
                labels["organ"] = organ_labels

                optimizer.zero_grad()
                outputs = model(
                    pixel_values=images, labels=labels, plant_mask=plant_mask
                )

                loss_species = outputs["loss_species"]
                loss_genus = outputs["loss_genus"]
                loss_family = outputs["loss_family"]
                loss_plant = outputs["loss_plant"]
                loss_organ = outputs["loss_organ"]

                loss = (
                    config.training.loss_weights.species * loss_species
                    + config.training.loss_weights.genus * loss_genus
                    + config.training.loss_weights.family * loss_family
                    + config.training.loss_weights.plant * loss_plant
                    + config.training.loss_weights.organ * loss_organ
                )

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                wandb.log(
                    {
                        "epoch": epoch,
                        "iter": iteration,
                        "loss": loss.item(),
                        "loss_species": loss_species.item(),
                        "loss_genus": loss_genus.item(),
                        "loss_family": loss_family.item(),
                        "loss_plant": loss_plant.item(),
                        "loss_organ": loss_organ.item(),
                    }
                )

                print(
                    f"Iter: {iteration}, Loss: {loss.item():.4f}, Loss Species: {loss_species.item():.4f}, "
                    f"Loss Genus: {loss_genus.item():.4f}, Loss Family: {loss_family.item():.4f}, "
                    f"Loss Plant: {loss_plant.item():.4f}, Loss Organ: {loss_organ.item():.4f}"
                )

            avg_loss = running_loss / len(train_dataloader)
            print(f"Epoch [{epoch + 1}/{config.training.epochs}], Loss: {avg_loss:.4f}")

    data_config = timm.data.resolve_model_data_config(model)
    model_info = ModelInfo(
        data_config["input_size"][1],
        data_config["mean"],
        data_config["std"],
    )

    return model, model_info
