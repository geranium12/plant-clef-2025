from typing import Any

import torch
import torch.nn as nn

from src.vit_multi_head_classifier import ViTMultiHeadClassifier


class MergedModel(nn.Module):  # type: ignore
    def __init__(
        self,
        species_model: ViTMultiHeadClassifier,
        genus_model: ViTMultiHeadClassifier,
        family_model: ViTMultiHeadClassifier,
    ) -> None:
        super().__init__()

        self.species_model = species_model
        self.genus_model = genus_model
        self.family_model = family_model

    # inference-only forward pass
    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        species_logits = self.species_model(pixel_values=pixel_values)["logits_species"]
        genus_logits = self.genus_model(pixel_values=pixel_values)["logits_genus"]
        family_logits = self.family_model(pixel_values=pixel_values)["logits_family"]

        return {
            "logits_species": species_logits,
            "logits_genus": genus_logits,
            "logits_family": family_logits,
        }

    def __str__(self) -> str:
        return (
            f"MergedModel(\n"
            f"  (species_model): {self.species_model}\n"
            f"  (genus_model): {self.genus_model}\n"
            f"  (family_model): {self.family_model}\n"
        )
