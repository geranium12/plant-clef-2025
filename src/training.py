import os
from dataclasses import (
    dataclass,
)

import pandas as pd
import timm
import torch
from omegaconf import (
    DictConfig,
)
from torch.utils.data import DataLoader, Dataset

import src.augmentation
from src.data import TrainDataset


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

    augmentations = src.augmentation.create_augmentations(config)

    for augmentation_name in config.training.augmentations:
        augmentation = augmentations[augmentation_name]
        train_dataset = TrainDataset(
            image_folder=os.path.join(
                config.project_path,
                config.data.folder,
                config.data.train_folder,
            ),
            image_size=(config.image_width, config.image_height),
            transform=augmentation,
        )

        fine_tune(config, model, train_dataset)

    data_config = timm.data.resolve_model_data_config(model)
    model_info = ModelInfo(
        data_config["input_size"][1],
        data_config["mean"],
        data_config["std"],
    )

    return model, model_info
