import os

import pandas as pd
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig


def load_model(
    config: DictConfig, device: torch.device, df_species_ids: pd.DataFrame
) -> nn.Module:
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
    return model


def save_model(
    model: torch.nn.Module,
    config: DictConfig,
) -> None:
    # check if the model folder exists, if not create it
    os.makedirs(
        os.path.join(
            config.project_path,
            config.models.save_folder,
        ),
        exist_ok=True,
    )

    model_path = os.path.join(
        config.project_path,
        config.models.save_folder,
        config.models.save_file,
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


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
    device: torch.device,
) -> torch.Tensor:
    """Calculates the weighted total loss."""
    total_loss = torch.tensor(0.0, device=device)
    weights = config.training.loss_weights
    for head in head_names:
        loss_key = f"loss_{head}"
        if loss_key in outputs:
            weight = getattr(
                weights, head, 0.0
            )  # Get weight, default to 0 if not specified
            total_loss += weight * outputs[loss_key]
    return total_loss
