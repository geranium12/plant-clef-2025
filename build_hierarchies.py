import json
import os

import hydra
import pandas as pd
from omegaconf import (
    DictConfig,
)
from tqdm import tqdm
from treelib import Tree

from src import data


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


def get_taxonomy_from_species(plant_tree: Tree, species: str) -> tuple[str, str]:
    # Find the species node
    species_node = plant_tree.get_node(species)
    if species_node is None:
        raise ValueError(f"Species '{species}' not found in taxonomy tree")

    # Get the parent (genus) and grandparent (family) nodes
    genus_node = plant_tree.parent(species)
    if genus_node is None:
        raise ValueError(f"Genus not found for species '{species}'")

    family_node = plant_tree.parent(genus_node.identifier)
    if family_node is None:
        raise ValueError(f"Family not found for species '{species}'")

    return genus_node.identifier, family_node.identifier


def get_organ_number(df_metadata: pd.DataFrame) -> int:
    # Check if any species has multiple organs
    species_organs = df_metadata.groupby("species")["organ"].nunique()
    organ_counts = species_organs.value_counts().sort_index()

    print("\nSpecies organ distribution:")
    for num_organs, count in organ_counts.items():
        print(
            f"  - {count} species have {num_organs} organ{'s' if num_organs > 1 else ''}"
        )

    return len(organ_counts)


def get_plant_tree_number(plant_tree: Tree) -> tuple[int, int, int]:
    # Get all nodes at each level
    species_nodes = [
        node for node in plant_tree.all_nodes() if plant_tree.depth(node) == 2
    ]
    genus_nodes = [
        node for node in plant_tree.all_nodes() if plant_tree.depth(node) == 1
    ]
    family_nodes = [
        node for node in plant_tree.all_nodes() if plant_tree.depth(node) == 0
    ]

    print("\nTaxonomy statistics:")
    print(f"  - Number of species: {len(species_nodes)}")
    print(f"  - Number of genera: {len(genus_nodes)}")
    print(f"  - Number of families: {len(family_nodes)}")

    return len(species_nodes), len(genus_nodes), len(family_nodes)


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

    # Print how many organs has each species in the training dataset
    get_organ_number(df_metadata)

    build_plant_taxonomy(config, df_metadata)

    build_organ_hierarchy(config, df_metadata)


if __name__ == "__main__":
    main()
