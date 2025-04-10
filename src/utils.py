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
