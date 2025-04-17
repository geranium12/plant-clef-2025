import os
from dataclasses import (
    dataclass,
)
from typing import Any, Optional

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import (
    DictConfig,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from treelib import Tree

import src.augmentation
import wandb
from src.data import ConcatenatedDataset, DataSplit, NonPlantDataset, PlantDataset
from src.evaluating import Evaluator
from src.utils import (
    family_name_to_id,
    genus_name_to_id,
    image_path_to_organ_name,
    organ_name_to_id,
    species_id_to_name,
    species_name_to_new_id,
)
from utils.build_hierarchies import (
    check_utils_folder,
    get_genus_family_from_species,
    read_plant_taxonomy,
)


@dataclass
class ModelInfo:
    input_size: int
    mean: float
    std: float


def gather_all_labels(
    species_labels: torch.Tensor,
    plant_labels: torch.Tensor,
    species_mapping: pd.DataFrame,
    genus_mapping: pd.DataFrame,
    family_mapping: pd.DataFrame,
    organ_mapping: pd.DataFrame,
    images_names: list[str],
    df_metadata: pd.DataFrame,
    plant_tree: Tree,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    labels = {}
    labels["plant"] = plant_labels

    new_species_labels = []
    genus_labels = []
    family_labels = []
    organ_labels = []
    for i, species_id in enumerate(species_labels):
        if species_id == -1:
            new_species_labels.append(-1)
            genus_labels.append(-1)
            family_labels.append(-1)
            organ_labels.append(-1)
        else:
            species_name = species_id_to_name(species_id.item(), species_mapping)
            genus_name, family_name = get_genus_family_from_species(
                plant_tree, species_name
            )
            new_species_id = species_name_to_new_id(species_name, species_mapping)
            genus_id = genus_name_to_id(genus_name, genus_mapping)
            family_id = family_name_to_id(family_name, family_mapping)
            new_species_labels.append(new_species_id)
            genus_labels.append(genus_id)
            family_labels.append(family_id)

            image_name = images_names[i]
            organ_name = image_path_to_organ_name(image_name, df_metadata)
            organ_id = organ_name_to_id(organ_name, organ_mapping)
            organ_labels.append(organ_id)

    new_species_labels = torch.tensor(new_species_labels).to(device)
    genus_labels = torch.tensor(genus_labels).to(device)
    family_labels = torch.tensor(family_labels).to(device)
    organ_labels = torch.tensor(organ_labels).to(device)
    labels["species"] = new_species_labels
    labels["genus"] = genus_labels
    labels["family"] = family_labels
    labels["organ"] = organ_labels
    return labels


def train(
    model: nn.Module,
    config: DictConfig,
    device: torch.device,
    df_species_ids: pd.DataFrame,
    df_metadata: pd.DataFrame,
    labeled_data_split: Optional[DataSplit] = None,
    unlabeled_data_split: Optional[DataSplit] = None,
) -> tuple[torch.nn.Module, ModelInfo]:
    plant_tree = read_plant_taxonomy(config)
    folder_name = check_utils_folder(config)

    def read_mapping(mapping_file: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(folder_name, mapping_file))

    species_mapping = read_mapping(config.data.utils.species_mapping)
    genus_mapping = read_mapping(config.data.utils.genus_mapping)
    family_mapping = read_mapping(config.data.utils.family_mapping)
    organ_mapping = read_mapping(config.data.utils.organ_mapping)

    train_dataset = ConcatenatedDataset(
        [
            PlantDataset(
                image_folder=os.path.join(
                    config.project_path,
                    config.data.folder,
                    config.data.train_folder,
                ),
                image_size=(config.image_width, config.image_height),
                indices=(
                    labeled_data_split.train_indices if labeled_data_split else None
                ),
            ),
            NonPlantDataset(
                image_folder=os.path.join(
                    config.project_path,
                    config.data.folder,
                    config.data.other.folder,
                ),
                image_size=(config.image_width, config.image_height),
                indices=unlabeled_data_split.train_indices
                if unlabeled_data_split
                else None,
            ),
        ]
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    if labeled_data_split is not None:
        val_dataset = ConcatenatedDataset(
            [
                PlantDataset(
                    image_folder=os.path.join(
                        config.project_path,
                        config.data.folder,
                        config.data.train_folder,
                    ),
                    image_size=(config.image_width, config.image_height),
                    indices=labeled_data_split.val_indices,
                ),
                NonPlantDataset(
                    image_folder=os.path.join(
                        config.project_path,
                        config.data.folder,
                        config.data.other.folder,
                    ),
                    image_size=(config.image_width, config.image_height),
                    indices=unlabeled_data_split.val_indices,  # type: ignore
                ),
            ]
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=config.evaluating.batch_size,
            shuffle=config.evaluating.shuffle,
            num_workers=config.evaluating.num_workers,
            pin_memory=True,
        )
        evaluator = Evaluator()

    model.to(device)
    model.train()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.training.lr
    )

    for epoch in range(config.training.epochs):
        running_loss = 0.0
        for iteration, batch in tqdm(
            enumerate(train_dataloader),
            desc="Training",
            total=len(train_dataloader),
        ):
            images, species_labels, images_names, plant_labels = batch
            images = images.to(device)
            species_labels = species_labels.to(device)
            plant_labels = plant_labels.to(device)

            augmentation = src.augmentation.get_random_data_augmentation(config)
            images = augmentation(images)

            labels = gather_all_labels(
                species_labels,
                plant_labels,
                species_mapping,
                genus_mapping,
                family_mapping,
                organ_mapping,
                images_names,
                df_metadata,
                plant_tree,
                device,
            )

            optimizer.zero_grad()
            outputs = model(
                pixel_values=images, labels=labels, plant_mask=labels["plant"] == 1
            )

            loss = (
                config.training.loss_weights.species * outputs["loss_species"]
                + config.training.loss_weights.genus * outputs["loss_genus"]
                + config.training.loss_weights.family * outputs["loss_family"]
                + config.training.loss_weights.plant * outputs["loss_plant"]
                + config.training.loss_weights.organ * outputs["loss_organ"]
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            def log_loss(
                outputs: dict[str, torch.Tensor],
                loss: torch.Tensor,
                epoch: int | None = None,
                iteration: int | None = None,
                suffix: str = "",
            ) -> None:
                wandb.log(
                    {
                        f"{suffix}/loss": loss.item(),
                        f"{suffix}/loss_species": outputs["loss_species"].item(),
                        f"{suffix}/loss_genus": outputs["loss_genus"].item(),
                        f"{suffix}/loss_family": outputs["loss_family"].item(),
                        f"{suffix}/loss_plant": outputs["loss_plant"].item(),
                        f"{suffix}/loss_organ": outputs["loss_organ"].item(),
                        f"{suffix}/epoch": epoch,
                        f"{suffix}/iter": iteration,
                    }
                )

                print(
                    f"{suffix.capitalize()} Iter: {iteration}, Loss: {loss.item():.4f}, Loss Species: {outputs['loss_species'].item():.4f}, "
                    f"Loss Genus: {outputs['loss_genus'].item():.4f}, Loss Family: {outputs['loss_family'].item():.4f}, "
                    f"Loss Plant: {outputs['loss_plant'].item():.4f}, Loss Organ: {outputs['loss_organ'].item():.4f}"
                )

            log_loss(outputs, loss, epoch=epoch, iteration=iteration)

            if (
                labeled_data_split is not None
                and iteration % config.evaluating.every == 0
            ):
                model.eval()
                all_preds: dict[str, list[Any]] = {v: [] for v in model.head_names}
                all_labels: dict[str, list[Any]] = {v: [] for v in model.head_names}
                running_val_loss = 0.0
                for val_iteration, val_batch in tqdm(
                    enumerate(val_dataloader),
                    desc="Validating",
                    total=len(val_dataloader),
                ):
                    images, species_labels, images_names, plant_labels = val_batch
                    images = images.to(device)
                    species_labels = species_labels.to(device)
                    plant_labels = plant_labels.to(device)

                    labels = gather_all_labels(
                        species_labels,
                        plant_labels,
                        species_mapping,
                        genus_mapping,
                        family_mapping,
                        organ_mapping,
                        images_names,
                        df_metadata,
                        plant_tree,
                        device,
                    )

                    with torch.no_grad():
                        outputs_val = model(
                            pixel_values=images,
                            labels=labels,
                            plant_mask=labels["plant"] == 1,
                        )
                        val_loss = (
                            config.training.loss_weights.species
                            * outputs_val["loss_species"]
                            + config.training.loss_weights.genus
                            * outputs_val["loss_genus"]
                            + config.training.loss_weights.family
                            * outputs["loss_family"]
                            + config.training.loss_weights.plant
                            * outputs_val["loss_plant"]
                            + config.training.loss_weights.organ
                            * outputs_val["loss_organ"]
                        )

                        running_val_loss += val_loss.item()

                        log_loss(
                            outputs_val, val_loss, iteration=val_iteration, suffix="val"
                        )

                        def evaluate_head(
                            logits: torch.Tensor, labels: torch.Tensor, suffix: str
                        ) -> torch.Tensor:
                            probs = torch.nn.functional.softmax(logits, dim=1).cpu()
                            metrics = evaluator.evaluate(labels.cpu(), probs)
                            wandb.log(
                                {
                                    f"{suffix}/batch_precision": metrics["precision"],
                                    f"{suffix}/batch_recall": metrics["recall"],
                                    f"{suffix}/batch_f1": metrics["f1"],
                                }
                            )
                            return probs

                        for head_name in model.head_names:
                            probs = evaluate_head(
                                outputs_val[f"logits_{head_name}"].view(-1, 1)
                                if head_name == "plant"
                                else outputs_val[f"logits_{head_name}"][
                                    labels["plant"] == 1
                                ],
                                labels[head_name].view(-1, 1)
                                if head_name == "plant"
                                else labels[head_name][labels["plant"] == 1],
                                suffix=head_name,
                            )

                            all_preds[head_name].append(probs.numpy())
                            all_labels[head_name].append(
                                labels[head_name][labels["plant"] == 1].cpu().numpy()
                            )

                for head_name in model.head_names:
                    y_pred = np.vstack(all_preds[head_name])
                    y_true = np.concatenate(all_labels[head_name])
                    metrics = evaluator.evaluate(y_true, y_pred)
                    wandb.log(
                        {
                            f"val/{head_name}/precision": metrics["precision"],
                            f"val/{head_name}/recall": metrics["recall"],
                            f"val/{head_name}/f1": metrics["f1"],
                        }
                    )
                    print(
                        f"Head Name: {head_name}, Precision: {metrics['precision']:.4f}, "
                        f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
                    )

                model.train()

        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{config.training.epochs}], Loss: {avg_loss:.4f}")

    data_config = timm.data.resolve_model_data_config(model)
    model_info = ModelInfo(
        data_config["input_size"][1],
        data_config["mean"],
        data_config["std"],
    )

    return model, model_info
