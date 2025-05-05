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
        classifier_type: str = "one_layer",  # Options: "one_layer", "two_layer_act"
    ) -> None:
        super().__init__()

        if not isinstance(num_labels_species, int) or num_labels_species <= 0:
            raise ValueError("num_labels_species must be a positive integer.")

        # Deep copy the backbone and retrieve the hidden size
        self.backbone = deepcopy(backbone)
        hidden_size = getattr(self.backbone, "num_features", None)
        if hidden_size is None:
            raise AttributeError("Backbone must have a 'num_features' attribute.")

        if num_labels_species is None:
            raise ValueError("num_labels_species must be provided.")

        self.classifier_species = self._make_classifier(
            hidden_size, num_labels_species, classifier_type
        )
        self.classifier_organ = self._make_classifier(
            hidden_size, num_labels_organ, classifier_type
        )
        self.classifier_genus = self._make_classifier(
            hidden_size, num_labels_genus, classifier_type
        )
        self.classifier_family = self._make_classifier(
            hidden_size, num_labels_family, classifier_type
        )
        self.classifier_plant = self._make_classifier(
            hidden_size, num_labels_plant, classifier_type
        )

        if hasattr(self.backbone, "head") and not isinstance(
            self.backbone.head, nn.Identity
        ):
            if num_labels_species != self.backbone.head.out_features:
                raise ValueError(
                    f"num_labels_species ({num_labels_species}) must match "
                    f"the output size of the backbone head ({self.backbone.head.out_features})."
                )
            self.classifier_species[-1] = deepcopy(self.backbone.head)
            self.backbone.head = nn.Identity()
            if hasattr(self.backbone, "head_drop"):
                self.backbone.head_drop = nn.Identity()

        # Optionally freeze backbone parameters.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if freeze_species_head:
            for param in self.classifier_species.parameters():
                param.requires_grad = False

        self.head_names = ["species", "genus", "family", "plant", "organ"]
        self.pretrained_cfg = self.backbone.pretrained_cfg

        self.classifier_type = classifier_type

    def _make_classifier(
        self,
        hidden_size: int,
        num_labels: int,
        classifier_type: str = "one_layer",
    ) -> nn.Sequential:
        """Helper method that creates a classifier block."""
        if classifier_type == "one_layer":
            return nn.Sequential(
                nn.Linear(hidden_size, num_labels),
            )
        elif classifier_type == "two_layer_act":
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, num_labels),
            )
        else:
            raise ValueError(
                f"Invalid classifier_type '{classifier_type}'. "
                "Expected 'one_layer' or 'two_layer_act'."
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
