from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from transformers import ViTModel


class ViTMultiHeadClassifier:
    def __init__(
        self,
        backbone: Any = ViTModel,
        # backbone_model_name="google/vit-base-patch16-224",
        num_labels_organ: int = 5,
        num_labels_genus: int = 100,
        num_labels_family: int = 50,
        num_labels_plant: int = 2,
        freeze_backbone: bool = True,
    ) -> None:
        # self.backbone = ViTModel.from_pretrained(backbone_model_name)
        self.backbone = deepcopy(backbone)
        hidden_size = self.backbone.config.hidden_size  # typically 768
        self.classifier_organ = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels_organ),
        )
        self.classifier_genus = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels_genus),
        )
        self.classifier_family = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels_family),
        )
        self.classifier_plant = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels_plant),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self, pixel_values: torch.Tensor, labels: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        # pixel_values: (batch, 3, 224, 224)
        outputs = self.backbone(pixel_values=pixel_values)
        # Use the CLS token representation
        cls_output = outputs.last_hidden_state[:, 0]
        logits_organ = self.classifier_organ(cls_output)
        logits_genus = self.classifier_genus(cls_output)
        logits_family = self.classifier_family(cls_output)
        logits_plant = self.classifier_plant(cls_output)

        if labels is not None:
            loss_organ = F.cross_entropy(logits_organ, labels["organ"])
            loss_genus = F.cross_entropy(logits_genus, labels["genus"])
            loss_family = F.cross_entropy(logits_family, labels["family"])
            loss_plant = F.cross_entropy(logits_plant, labels["plant"])
            # TODO: Do we want weigh losses?
            loss = loss_organ + loss_genus + loss_family + loss_plant
            return {
                "loss": loss,
                "loss_organ": loss_organ,
                "loss_genus": loss_genus,
                "loss_family": loss_family,
                "loss_plant": loss_plant,
                "logits_organ": logits_organ,
                "logits_genus": logits_genus,
                "logits_family": logits_family,
                "logits_plant": logits_plant,
            }
        else:
            return {
                "logits_organ": logits_organ,
                "logits_genus": logits_genus,
                "logits_family": logits_family,
                "logits_plant": logits_plant,
            }
