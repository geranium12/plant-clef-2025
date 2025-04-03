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

    data_config = timm.data.resolve_model_data_config(model)
    model_info = ModelInfo(
        data_config["input_size"][1],
        data_config["mean"],
        data_config["std"],
    )

    return model, model_info
