import os
from dataclasses import (
    dataclass,
)
from typing import Optional

import pandas as pd
import timm
import torch
from omegaconf import (
    DictConfig,
)
from torch.utils.data import DataLoader, Dataset

import src.augmentation
from src.data import TrainDataset, UnlabeledDataset


def fine_tune(config: DictConfig, model: torch.nn.Module, dataset: Dataset) -> None:
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        num_workers=config.training.num_workers,
    )

    for batch_images in train_loader:
        images, labels = batch_images
        break  # TODO: Use the data to fine-tune the model


@dataclass
class ModelInfo:
    input_size: int
    mean: float
    std: float


def train(
    config: DictConfig,
    device: torch.device,
    df_species_ids: pd.DataFrame,
    train_indices: Optional[list[int]],
    val_indices: Optional[list[int]],  # TODO: Use val_indices
) -> tuple[torch.nn.Module, ModelInfo]:
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

        fine_tune(config, model, train_dataset)

    for augmentation_name in config.training.augmentations:
        augmentation = src.augmentation.get_data_augmentation(config, augmentation_name)
        train_dataset = UnlabeledDataset(
            image_folder=os.path.join(
                config.project_path, config.data.folder, config.data.other.folder
            ),
            image_size=(config.image_width, config.image_height),
            transform=augmentation,
            indices=None,
        )
        # TODO: Use unlabeled rock/soil data to fine-tune the model

    data_config = timm.data.resolve_model_data_config(model)
    model_info = ModelInfo(
        data_config["input_size"][1],
        data_config["mean"],
        data_config["std"],
    )

    return model, model_info
