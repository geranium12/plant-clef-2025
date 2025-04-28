from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from transformers import ViTModel


class ViTMultiHeadClassifier(nn.Module):  # type: ignore
    def __init__(
        self,
        backbone: Any = ViTModel,
        num_labels_organ: int = 5,
        num_labels_genus: int = 100,
        num_labels_family: int = 50,
        num_labels_plant: int = 1,
        num_labels_species: int | None = None,
        freeze_backbone: bool = True,
        freeze_species_head: bool = False,
    ) -> None:
        super().__init__()

        # Deep copy the backbone and retrieve the hidden size
        self.backbone = deepcopy(backbone)
        hidden_size = getattr(self.backbone, "num_features", None)
        if hidden_size is None:
            raise AttributeError("Backbone must have a 'num_features' attribute.")

        # Build species classifier using the backbone head if available,
        # else fallback to a default classifier.
        if hasattr(self.backbone, "head") and (
            num_labels_species is None
            or num_labels_species == self.backbone.head.out_features
        ):
            self.classifier_species = nn.Sequential(
                *(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    deepcopy(self.backbone.head),
                )
                if not freeze_species_head
                else deepcopy(self.backbone.head)
            )
            # Replace the backbone head with an identity function
            self.backbone.head = nn.Identity()
            if hasattr(self.backbone, "head_drop"):
                self.backbone.head_drop = nn.Identity()
        else:
            if num_labels_species is None:
                raise ValueError(
                    "num_labels_species must be provided if no head is available."
                )
            self.classifier_species = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, num_labels_species),
            )

        # Create the remaining classifier heads using a helper method.
        self.classifier_organ = self._make_classifier(hidden_size, num_labels_organ)
        self.classifier_genus = self._make_classifier(hidden_size, num_labels_genus)
        self.classifier_family = self._make_classifier(hidden_size, num_labels_family)
        self.classifier_plant = self._make_classifier(hidden_size, num_labels_plant)

        # Optionally freeze backbone parameters.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if freeze_species_head:
            for param in self.classifier_species.parameters():
                param.requires_grad = False

        self.head_names = ["species", "genus", "family", "plant", "organ"]
        self.pretrained_cfg = self.backbone.pretrained_cfg

    @staticmethod
    def _make_classifier(hidden_size: int, num_labels: int) -> nn.Sequential:
        """Helper method that creates a classifier block."""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: dict[str, torch.Tensor] | None = None,
        plant_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            pixel_values (torch.Tensor): The input tensor with image pixels.
            labels (Optional[Dict[str, torch.Tensor]]): Dictionary containing targets for
                each head. Expected keys: "species", "genus", "family", "organ", "plant".
            plant_mask (Optional[torch.Tensor]): Boolean mask to apply on the loss computation.

        Returns:
            A dictionary that contains logits for each head and, if labels are provided,
            computed losses.
        """
        features = self.backbone.forward_features(pixel_values)
        cls_output = features[:, 0]  # Use the CLS token embedding

        # Compute logits for each head.
        logits = {
            "organ": self.classifier_organ(cls_output),
            "species": self.classifier_species(cls_output),
            "genus": self.classifier_genus(cls_output),
            "family": self.classifier_family(cls_output),
            "plant": self.classifier_plant(cls_output).squeeze(),
        }

        # If no labels are provided, simply return the logits.
        if labels is None:
            return {f"logits_{key}": value for key, value in logits.items()}

        # Compute losses using CrossEntropy for multi-class and BCEWithLogits for plant.
        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCEWithLogitsLoss()
        losses = {}

        for key in ["species", "genus", "family", "organ"]:
            if plant_mask is not None:
                losses[f"loss_{key}"] = ce_loss(
                    logits[key][plant_mask], labels[key][plant_mask]
                )
            else:
                losses[f"loss_{key}"] = ce_loss(logits[key], labels[key])
        losses["loss_plant"] = bce_loss(logits["plant"], labels["plant"])

        return {**losses, **{f"logits_{k}": v for k, v in logits.items()}}

    def __str__(self) -> str:
        return (
            f"ViTMultiHeadClassifier(\n"
            f"  (backbone): {self.backbone}\n"
            f"  (classifier_species): {self.classifier_species}\n"
            f"  (classifier_organ): {self.classifier_organ}\n"
            f"  (classifier_genus): {self.classifier_genus}\n"
            f"  (classifier_family): {self.classifier_family}\n"
            f"  (classifier_plant): {self.classifier_plant}\n)"
        )
