import os
from dataclasses import dataclass

import pandas as pd
import safetensors
import timm
import torch
import torch.nn as nn
from accelerate import Accelerator
from omegaconf import DictConfig

import wandb
from src.vit_multi_head_classifier import ViTMultiHeadClassifier


@dataclass
class ModelInfo:
    input_size: int
    mean: float
    std: float


def load_model(
    df_metadata: pd.DataFrame,
    num_species: int,
    num_genus: int,
    num_family: int,
    num_organ: int,
    num_plant: int,
    model_config: DictConfig,
    project_path: str,
) -> nn.Module:
    if model_config.load_5heads_model:
        model_path = os.path.join(
            project_path,
            model_config.folder,
            model_config.checkpoint_file,
            "model.safetensors",
        )
        backbone = timm.create_model(
            model_config.name,
            pretrained=False,
            num_classes=num_species,
        )
        model = ViTMultiHeadClassifier(
            backbone=backbone,
            num_labels_species=num_species,
            num_labels_organ=num_organ,
            num_labels_genus=num_genus,
            num_labels_family=num_family,
            num_labels_plant=num_plant,
            freeze_backbone=model_config.freeze_backbone,
            freeze_species_head=model_config.freeze_species_head,
            classifier_type=model_config.classifier_type,
            freeze_plant_head=model_config.freeze_plant_head,
            freeze_organ_head=model_config.freeze_organ_head,
        )
        safetensors.torch.load_model(model, model_path)
    else:
        model = timm.create_model(
            model_config.name,
            pretrained=model_config.pretrained,
            num_classes=len(df_metadata["species_id"].unique()),
            checkpoint_path=os.path.join(
                project_path, model_config.folder, model_config.checkpoint_file
            ),
        )
        model = ViTMultiHeadClassifier(
            backbone=model,
            num_labels_species=num_species,
            num_labels_organ=num_organ,
            num_labels_genus=num_genus,
            num_labels_family=num_family,
            num_labels_plant=num_plant,
            freeze_backbone=model_config.freeze_backbone,
            freeze_species_head=model_config.freeze_species_head,
            classifier_type=model_config.classifier_type,
            freeze_plant_head=model_config.freeze_plant_head,
            freeze_organ_head=model_config.freeze_organ_head,
        )
    return model


def species_id_to_name(species_id: int, species_mapping: pd.DataFrame) -> str:
    species_row = species_mapping[species_mapping["species_id"] == species_id]
    if species_row.empty:
        raise ValueError(f"Species ID {species_id} not found in metadata")
    return str(species_row["species_name"].iloc[0])


def genus_name_to_id(genus_name: str, genus_mapping: pd.DataFrame) -> int:
    genus_row = genus_mapping[genus_mapping["genus_name"] == genus_name]
    if genus_row.empty:
        raise ValueError(f"Genus name '{genus_name}' not found in mapping")

    return int(genus_row["genus_id"].iloc[0])


def family_name_to_id(family_name: str, family_mapping: pd.DataFrame) -> int:
    family_row = family_mapping[family_mapping["family_name"] == family_name]
    if family_row.empty:
        raise ValueError(f"Family name '{family_name}' not found in mapping")

    return int(family_row["family_id"].iloc[0])


def organ_name_to_id(organ_name: str, organ_mapping: pd.DataFrame) -> int:
    organ_row = organ_mapping[organ_mapping["organ_name"] == organ_name]
    if organ_row.empty:
        raise ValueError(f"Organ name '{organ_name}' not found in mapping")

    return int(organ_row["organ_id"].iloc[0])


def image_path_to_organ_name(image_path: str, df_metadata: pd.DataFrame) -> str:
    row = df_metadata[df_metadata["image_name"] == image_path]
    if row.empty:
        raise ValueError(f"Image path '{image_path} not found in metadata")

    return str(row["organ"].iloc[0])


def calculate_total_loss(
    outputs: dict[str, torch.Tensor],
    head_names: list[str],
    config: DictConfig,
) -> torch.Tensor:
    """Calculates the weighted total loss."""
    total_loss = torch.tensor(0.0, device=outputs[f"loss_{head_names[0]}"].device)
    weights = config.training.loss_weights
    for head in head_names:
        loss_key = f"loss_{head}"
        if loss_key in outputs:
            weight = getattr(
                weights, head, 0.0
            )  # Get weight, default to 0 if not specified
            total_loss += weight * outputs[loss_key]
    return total_loss


def log_loss(
    outputs: dict[str, torch.Tensor],
    loss: torch.Tensor,
    head_names: list[str],
    accelerator: Accelerator,
    prefix: str,
    step: int,
) -> None:
    """Logs loss components to wandb and console."""
    log_data = {
        f"{prefix}/loss": loss.item(),
    }

    print_str = f"{prefix.capitalize()} Loss: {loss.item():.4f}"

    for head in head_names:
        loss_key = f"loss_{head}"
        if loss_key in outputs:
            log_data[f"{prefix}/{loss_key}"] = outputs[loss_key].item()
            print_str += f", Loss {head.capitalize()}: {outputs[loss_key].item():.4f}"

    log_data[f"{prefix}/step"] = step
    accelerator.log(log_data)
    accelerator.print(print_str)


def define_metrics() -> None:
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/step")
    wandb.define_metric("val/*", step_metric="val/step")
    wandb.define_metric("test/step")
    wandb.define_metric("test/*", step_metric="test/step")
