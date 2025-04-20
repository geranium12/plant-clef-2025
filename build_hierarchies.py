import json
import os

import hydra
import pandas as pd
import numpy as np
from omegaconf import (
    DictConfig,
)
from tqdm import tqdm
from treelib import Tree

from src import data
from typing import Optional, Union

import torch

# family -> genus -> species
def build_plant_taxonomy(
    config: DictConfig,
    df_metadata: pd.DataFrame,
) -> None:
    # Ensure that the columns exist
    assert "family" in df_metadata.columns, (
        "Metadata DataFrame must contain 'family' column"
    )
    assert "genus" in df_metadata.columns, (
        "Metadata DataFrame must contain 'genus' column"
    )
    assert "species" in df_metadata.columns, (
        "Metadata DataFrame must contain 'species' column"
    )

    # Initialize the tree with a root node
    plant_tree = Tree()
    plant_tree.create_node("Plant", "plant")

    # Build the tree structure:
    # For each family, add that as a child of the root.
    # Then, for each genus in the family, add it under the family.
    # Finally, for each species in the genus, add it under the genus.
    for family in tqdm(df_metadata["family"].unique(), desc="Processing families"):
        plant_tree.create_node(family, family, parent="plant")
        family_genuses = df_metadata[df_metadata["family"] == family]["genus"].unique()
        for genus in family_genuses:
            plant_tree.create_node(genus, genus, parent=family)
            genus_species = df_metadata[
                (df_metadata["family"] == family) & (df_metadata["genus"] == genus)
            ]["species"].unique()

            for species in genus_species:
                plant_tree.create_node(species, species, parent=genus)

    # Display a small portion of the tree
    print("\nSample of plant taxonomy tree:")
    for node in plant_tree.all_nodes()[:10]:  # Show first 10 nodes
        print(f"{'  ' * plant_tree.depth(node)}- {node.tag}")

    # Convert the tree to a dictionary and then to JSON
    tree_dict = plant_tree.to_dict(with_data=True)
    json_string = json.dumps(tree_dict, indent=4)

    # Save the JSON string to a file
    folder_path = os.path.join(
        config.project_path,
        config.data.folder,
        config.data.utils.folder,
    )
    os.makedirs(
        folder_path,
        exist_ok=True,
    )
    with open(
        os.path.join(
            folder_path,
            config.data.utils.plant_taxonomy_file,
        ),
        "w",
    ) as json_file:
        json_file.write(json_string)


def read_plant_taxonomy(config: DictConfig) -> Tree:
    with open(
        os.path.join(
            config.project_path,
            config.data.folder,
            config.data.utils.folder,
            config.data.utils.plant_taxonomy_file,
        ),
    ) as json_file:
        tree_dict = json.load(json_file)

    plant_tree = Tree()
    plant_tree.from_dict(tree_dict)
    return plant_tree


def print_organ_distribution(df_metadata: pd.DataFrame) -> None:
    # Check if any species has multiple organs
    species_organs = df_metadata.groupby("species")["organ"].nunique()
    organ_counts = species_organs.value_counts().sort_index()

    print("\nSpecies organ distribution:")
    for num_organs, count in organ_counts.items():
        print(
            f"  - {count} species have {num_organs} organ{'s' if num_organs > 1 else ''}"
        )


# organ -> species
def build_organ_hierarchy(config: DictConfig, df_metadata: pd.DataFrame) -> None:
    # Ensure that the columns exist
    assert "organ" in df_metadata.columns, (
        "Metadata DataFrame must contain 'organ' column"
    )
    assert "species" in df_metadata.columns, (
        "Metadata DataFrame must contain 'species' column"
    )

    # Build a Dict where, for each species, it has a Dict with 0s/1s of all organ categories
    organ_categories = df_metadata["organ"].unique().tolist()
    organ_hierarchy: dict[str, dict[str, int]] = {}
    for species in tqdm(df_metadata["species"].unique(), desc="Processing species"):
        organ_hierarchy[species] = {}
        organs = set(df_metadata[df_metadata["species"] == species]["organ"].unique())
        for organ in organ_categories:
            if organ in organs:
                organ_hierarchy[species][organ] = 1
            else:
                organ_hierarchy[species][organ] = 0

    # Convert the Dict to a DataFrame
    organ_hierarchy_df = pd.DataFrame.from_dict(organ_hierarchy, orient="index")
    organ_hierarchy_df = organ_hierarchy_df.reset_index().rename(
        columns={"index": "species"}
    )
    print(organ_hierarchy_df.head())

    # Save the DataFrame to a csv file
    folder_path = os.path.join(
        config.project_path,
        config.data.folder,
        config.data.utils.folder,
    )
    os.makedirs(
        folder_path,
        exist_ok=True,
    )
    organ_hierarchy_df.to_csv(
        os.path.join(
            folder_path,
            config.data.utils.organ_hierarchy_file,
        ),
        index=False,
    )


def read_organ_hierarchy(config: DictConfig) -> pd.DataFrame:
    species_organs_df = pd.read_csv(
        os.path.join(
            config.project_path,
            config.data.folder,
            config.data.utils.folder,
            config.data.utils.organ_hierarchy_file,
        )
    )
    return species_organs_df


# family -> genus -> species
class PlantHierarchy(object):

    families: np.ndarray[np.str_]
    genera: np.ndarray[np.str_]
    species: np.ndarray[np.str_]

    species_to_genus: np.ndarray[np.int32]
    genus_to_family: np.ndarray[np.int32]

    def __init__(self, df_metadata: Optional[Union[pd.DataFrame,str]]=None):
        if isinstance(df_metadata, pd.DataFrame):
            self.build_hierarchy(df_metadata)
        elif isinstance(df_metadata, str):
            self.load(df_metadata)

        
    def build_hierarchy(self, df_metadata: pd.DataFrame, *, verbose: bool=False):
        #tqdm(df_metadata["species"].unique(), desc="Processing species")
        if verbose:
            print('Building Species Index')
        data = np.unique(df_metadata[['species','genus','family']].to_numpy(dtype=str), axis=0)
        self.species = data[:,0].copy()

        if verbose:
            print('Building Genus Index')
        genfam, self.species_to_genus = np.unique(data[:,[1,2]], axis=0, return_inverse=True)
        self.genera = genfam[:,0].copy()

        if verbose:
            print('Building Family Index')
        self.families, self.genus_to_family = np.unique(genfam[:,1], return_inverse=True)

        if verbose:
            print('Building Finished')

    def family_to_index(self, family: np.ndarray[str])-> np.ndarray[np.int32]:
        return np.searchsorted(self.families, family)

    def genus_to_index(self, genus: np.ndarray[str])-> np.ndarray[np.int32]:
        return np.searchsorted(self.genera, genus)

    def species_to_index(self, species: np.ndarray[str])-> np.ndarray[np.int32]:
        return np.searchsorted(self.species, species)

    def batch_to_ids(self, batch: np.ndarray[str])-> torch.Tensor: # ints 
        '''
        batch: nx3 array of strings, columns in order ('species', 'genus', 'family')
        
        output: nx3 tensor of indices
        '''
        nm = batch.shape
        if len(nm) < 2:
            return torch.from_numpy(self.species_to_index(batch))
        
        out = np.empty(nm, dtype=int)
        out[:,0] = self.species_to_index(batch[:,0])
        if batch.shape[1] > 1:
            out[:,1] = self.genus_to_index(batch[:,1])
        if batch.shape[2] > 2:
            out[:,2] = self.family_to_index(batch[:,2])

        return torch.from_numpy(out)

    def merge_probabilities(self,
                            species: torch.Tensor, # floats
                            genus: Optional[torch.Tensor]=None, # floats
                            family: Optional[torch.Tensor]=None # float
                            )-> torch.Tensor: # floats
        probs = species
        if genus is not None:
            probs = probs * genus[self.species_to_genus]
        if family is not None:
            species_to_family = self.genus_to_family[self.species_to_genus]
            probs = probs * family[species_to_family]
        return probs
        
    def save(self, file_name: str='./taxonomy.npz'):
        np.savez_compressed(file_name,
                            families=self.families,
                            genera=self.genera,
                            species=self.species,
                            species_to_genus=self.species_to_genus,
                            genus_to_family=self.genus_to_family
                            )
    
    def load(self, file_name: str='./taxonomy.npz'):
        data = np.load(file_name)
        self.families = data['families']
        self.genera = data['genera']
        self.species = data['species']
        self.species_to_genus = data['species_to_genus']
        self.genus_to_family = data['genus_to_family']

    
@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config",
)
def main(
    config: DictConfig,
) -> None:
    # Read the CSV file
    df_metadata, _, _ = data.load(config)

    PH = PlantHierarchy()
    PH.build_hierarchy(df_metadata, verbose=True)

    # Print how many organs has each species in the training dataset
    print_organ_distribution(df_metadata)

    build_plant_taxonomy(config, df_metadata)

    build_organ_hierarchy(config, df_metadata)

    

if __name__ == "__main__":
    main()
