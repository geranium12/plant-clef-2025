from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from transformers import ViTModel


class ViTMultiHeadClassifier(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        backbone: Any = ViTModel,
        num_labels_organ: int = 5,
        num_labels_genus: int = 100,
        num_labels_family: int = 50,
        num_labels_plant: int = 1,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = deepcopy(backbone)
        hidden_size = self.backbone.num_features

        if hasattr(self.backbone, "head"):
            self.classifier_species = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                deepcopy(self.backbone.head),
            )
            self.backbone.head = nn.Identity()
            self.backbone.head_drop = nn.Identity()

        self.classifier_organ = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_labels_organ),
        )
        self.classifier_genus = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_labels_genus),
        )
        self.classifier_family = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_labels_family),
        )
        self.classifier_plant = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_labels_plant),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        outputs = self.backbone.forward_features(pixel_values)
        cls_output = outputs[:, 0]  # Get the CLS token embedding (first token)
        logits_organ = self.classifier_organ(cls_output)
        logits_species = self.classifier_species(cls_output)
        logits_genus = self.classifier_genus(cls_output)
        logits_family = self.classifier_family(cls_output)
        logits_plant = self.classifier_plant(cls_output).squeeze()

        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCEWithLogitsLoss()
        if labels is not None:
            loss_organ = ce_loss(logits_organ, labels["organ"])
            loss_species = ce_loss(logits_species, labels["species"])
            loss_genus = ce_loss(logits_genus, labels["genus"])
            loss_family = ce_loss(logits_family, labels["family"])
            loss_plant = bce_loss(logits_plant, labels["plant"])
            # TODO: Do we want weigh losses?
            loss = loss_organ + loss_genus + loss_family + loss_plant
            return {
                "loss": loss,
                "loss_organ": loss_organ,
                "loss_species": loss_species,
                "loss_genus": loss_genus,
                "loss_family": loss_family,
                "loss_plant": loss_plant,
                "logits_organ": logits_organ,
                "logits_species": logits_species,
                "logits_genus": logits_genus,
                "logits_family": logits_family,
                "logits_plant": logits_plant,
            }
        else:
            return {
                "logits_organ": logits_organ,
                "logits_species": logits_species,
                "logits_genus": logits_genus,
                "logits_family": logits_family,
                "logits_plant": logits_plant,
            }

    def __str__(self) -> str:
        return f"""ViTMultiHeadClassifier(
  (backbone): {self.backbone}
  (classifier_species): {self.classifier_species}
  (classifier_organ): {self.classifier_organ}
  (classifier_genus): {self.classifier_genus}
  (classifier_family): {self.classifier_family}
  (classifier_plant): {self.classifier_plant}
)"""
