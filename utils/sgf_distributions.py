import os
import hydra
from omegaconf import (
    DictConfig,
    OmegaConf,
)
import src.data as data
from src.utils import family_name_to_id, genus_name_to_id, species_id_to_name
from utils.build_hierarchies import (
    get_organ_number,
    get_plant_tree_number,
    read_plant_taxonomy,
)
from utils.build_hierarchies import (
    check_utils_folder,
    get_genus_family_from_species,
    read_plant_taxonomy,
)

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def pipeline(
        config: DictConfig,
        species_index_to_id: dict[int, int],
        species_id_to_index: dict[int, int],
):
    
    df_metadata = data.load_metadata(config)

    plant_tree = read_plant_taxonomy(config)

    folder_path = check_utils_folder(config)

    species_mapping = pd.read_csv(
        os.path.join(
            folder_path,
            config.data.utils.species_mapping,
        ),
        index_col=False,
    )

    genus_mapping = pd.read_csv(
        os.path.join(
            folder_path,
            config.data.utils.genus_mapping,
        ),
        index_col=False,
    )

    family_mapping = pd.read_csv(
        os.path.join(
            folder_path,
            config.data.utils.family_mapping,
        ),
        index_col=False,
    )

    species_to_other = sorted(
        [
            (
                species_index,
                get_genus_family_from_species(
                    plant_tree, species_id_to_name(species_id, species_mapping)
                ),
            )
            for species_index, species_id in species_index_to_id.items()
            if species_id != 0
        ]
    )

    species_to_genus_list = []
    species_to_family_list = []
    for _, (genus, family) in species_to_other:
        gid = genus_name_to_id(genus, genus_mapping)
        fid = family_name_to_id(family, family_mapping)
        species_to_genus_list.append(gid)
        species_to_family_list.append(fid)

    species_to_genus = torch.tensor(species_to_genus_list, dtype=torch.int64)
    species_to_family = torch.tensor(species_to_family_list, dtype=torch.int64)

    print('Species:', len(species_to_genus))
    print('Genus:', len(np.unique(species_to_genus)))
    print('Family:', len(np.unique(species_to_family)))

    genus_counts = np.bincount(species_to_genus_list)
    genus_counts = sorted(genus_counts[genus_counts > 0], reverse=True)
    family_counts = np.bincount(species_to_family_list)
    family_counts = sorted(family_counts[family_counts > 0], reverse=True)
    #plt.hist(genus_counts)
    #plt.savefig('./genus.pdf')
    #plt.close()
    
    #plt.hist(family_counts)
    #plt.savefig('./family.pdf')
    #plt.close()

    print('GENUS')
    sm = 0
    for i, c in enumerate(genus_counts):
        sm += c
        print(i, sm/sum(genus_counts))
    # Save genus_counts to CSV
    pd.DataFrame({
        'genus_rank': list(range(len(genus_counts))),
        'genus_counts': genus_counts,
    }).to_csv('./sgenus.csv', index=False)

    print('FAMILY')
    sm = 0
    for i, c in enumerate(family_counts):
        sm += c
        print(i, sm/sum(family_counts))
    # Save family_counts to CSV
    pd.DataFrame({
        'family_rank': list(range(len(family_counts))),
        'family_counts': family_counts,
    }).to_csv('./sfamily.csv', index=False)

    sns.lineplot(genus_counts)
    plt.savefig('./sgenus.pdf')
    plt.close()

    sns.lineplot(family_counts)
    plt.savefig('./sfamily.pdf')
    plt.close()

    species_counts = np.bincount(df_metadata[['species_id', 'genus', 'family']]['species_id'])
    species_counts = sorted(species_counts[species_counts>0], reverse=True)
    
    print('SPECIES')
    sm = 0
    for i, c in enumerate(species_counts):
        sm += c
        print(i, sm/sum(species_counts))

    # Save species_counts to CSV
    pd.DataFrame({
        'species_rank': list(range(len(species_counts))),
        'species_counts': species_counts,
    }).to_csv('./sspecies.csv', index=False)
    sns.lineplot(species_counts)
    plt.savefig('./sspecies.pdf')
    plt.close()
    
@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config",
)
def main(
    config: DictConfig,
) -> None:
    plant_data_image_info, rare_species = data.get_plant_data_image_info(
        os.path.join(
            config.project_path,
            config.data.folder,
            config.data.train_folder,
        ),
        combine_classes_threshold=config.data.combine_classes_threshold,
    )
    species_id_to_index = {
        sid: idx
        for idx, sid in enumerate(
                sorted({info.species_id for info in plant_data_image_info})
        )
    }
    species_index_to_id = {idx: sid for sid, idx in species_id_to_index.items()}
    # NOTE: accelerator logs on main process only -> loss from only one GPU is logged
    # Gathering loss from all GPUs will slow down training
    pipeline(config, species_index_to_id, species_id_to_index)

if __name__ == '__main__':
    main()
