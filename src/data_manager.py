import os
from dataclasses import dataclass

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.data import ConcatenatedDataset, DataSplit, NonPlantDataset, PlantDataset
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
class DataManager:
    config: DictConfig
    plant_data_split: DataSplit | None
    non_plant_data_split: DataSplit | None
    df_metadata: pd.DataFrame

    def __post_init__(self) -> None:
        self._load_mappings_and_taxonomy()
        self._setup_datasets()
        self._setup_dataloaders()

    def _load_mappings_and_taxonomy(self) -> None:
        """Loads taxonomic tree and mapping files."""
        self.plant_tree = read_plant_taxonomy(self.config)
        utils_folder = check_utils_folder(self.config)

        def read_mapping(mapping_file: str) -> pd.DataFrame:
            return pd.read_csv(os.path.join(utils_folder, mapping_file))

        self.species_mapping = read_mapping(self.config.data.utils.species_mapping)
        self.genus_mapping = read_mapping(self.config.data.utils.genus_mapping)
        self.family_mapping = read_mapping(self.config.data.utils.family_mapping)
        self.organ_mapping = read_mapping(self.config.data.utils.organ_mapping)

    def _create_dataset(self, split_type: str) -> Dataset:
        """Creates a concatenated dataset for a given split ('train' or 'val')."""
        plant_indices = (
            getattr(self.plant_data_split, f"{split_type}_indices", None)
            if self.plant_data_split
            else None
        )
        non_plant_indices = (
            getattr(self.non_plant_data_split, f"{split_type}_indices", None)
            if self.non_plant_data_split
            else None
        )

        image_folder_train = os.path.join(
            self.config.project_path,
            self.config.data.folder,
            self.config.data.train_folder,
        )
        image_folder_other = os.path.join(
            self.config.project_path,
            self.config.data.folder,
            self.config.data.other.folder,
        )
        image_size = (self.config.image_width, self.config.image_height)

        datasets_to_concat = []
        if (
            plant_indices is not None or split_type == "train"
        ):  # Always include plants for training if no split provided
            datasets_to_concat.append(
                PlantDataset(
                    image_folder=image_folder_train,
                    image_size=image_size,
                    indices=plant_indices,
                )
            )
        if (
            non_plant_indices is not None or split_type == "train"
        ):  # Always include non-plants for training if no split provided
            datasets_to_concat.append(
                NonPlantDataset(
                    image_folder=image_folder_other,
                    image_size=image_size,
                    indices=non_plant_indices,
                )
            )

        if not datasets_to_concat:
            raise ValueError(f"No data available for split type: {split_type}")

        return ConcatenatedDataset(datasets_to_concat)

    def _setup_datasets(self) -> None:
        """Sets up train and validation datasets."""
        self.train_dataset = self._create_dataset("train")
        if self.plant_data_split is not None:
            self.val_dataset = self._create_dataset("val")
            self.test_dataset = self._create_dataset("test")
        else:
            self.val_dataset = None
            self.test_dataset = None

    def _setup_dataloaders(self) -> None:
        """Sets up train and validation dataloaders."""
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.training.shuffle,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
        )
        if self.val_dataset:
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.config.evaluating.batch_size,
                shuffle=self.config.evaluating.shuffle,
                num_workers=self.config.evaluating.num_workers,
                pin_memory=True,
            )
        else:
            self.val_dataloader = None
        if self.test_dataset:
            self.test_dataloader = DataLoader(
                dataset=self.test_dataset,
                batch_size=self.config.evaluating.batch_size,
                shuffle=self.config.evaluating.shuffle,
                num_workers=self.config.evaluating.num_workers,
                pin_memory=True,
            )
        else:
            self.test_dataloader = None

    def gather_all_labels(
        self,
        species_labels: torch.Tensor,
        plant_labels: torch.Tensor,
        images_names: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Gathers and transforms labels for species, genus, family, and organ
        based on the initial species and plant labels.
        """
        labels = {"plant": plant_labels}
        batch_size = len(species_labels)
        new_species_ids = torch.full(
            (batch_size,), -1, dtype=torch.long, device=species_labels.device
        )
        genus_ids = torch.full(
            (batch_size,), -1, dtype=torch.long, device=species_labels.device
        )
        family_ids = torch.full(
            (batch_size,), -1, dtype=torch.long, device=species_labels.device
        )
        organ_ids = torch.full(
            (batch_size,), -1, dtype=torch.long, device=species_labels.device
        )

        # Process only plant samples (where species_id != -1)
        plant_mask = species_labels != -1
        plant_indices = torch.where(plant_mask)[0]

        for idx in plant_indices:
            i = idx.item()  # Get the actual index in the batch
            species_id = species_labels[i].item()
            species_name = species_id_to_name(species_id, self.species_mapping)
            genus_name, family_name = get_genus_family_from_species(
                self.plant_tree, species_name
            )
            image_name = images_names[i]
            organ_name = image_path_to_organ_name(image_name, self.df_metadata)

            new_species_ids[i] = species_name_to_new_id(
                species_name, self.species_mapping
            )
            genus_ids[i] = genus_name_to_id(genus_name, self.genus_mapping)
            family_ids[i] = family_name_to_id(family_name, self.family_mapping)
            organ_ids[i] = organ_name_to_id(organ_name, self.organ_mapping)

        labels["species"] = new_species_ids
        labels["genus"] = genus_ids
        labels["family"] = family_ids
        labels["organ"] = organ_ids
        return labels
