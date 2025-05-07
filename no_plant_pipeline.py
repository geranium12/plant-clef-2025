import os

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA

import hydra
from accelerate import Accelerator
from omegaconf import (
    DictConfig,
    OmegaConf,
)
from torch.utils.data import DataLoader

from src.data import PlantDataset, NonPlantDataset
import src.data as data
from src import prediction, submission, training
from src.data_manager import DataManager
from src.utils import define_metrics, load_model
from src.vit_multi_head_classifier import ViTMultiHeadClassifier
from utils.build_hierarchies import (
    get_organ_number,
    get_plant_tree_number,
    read_plant_taxonomy,
)
from src.augmentation import get_random_data_augmentation
import gc

def pipeline(
    config: DictConfig,
    accelerator: Accelerator,
) -> None:
    df_metadata = data.load_metadata(config)

    print('get plant data')
    plant_data_image_info, rare_species = data.get_plant_data_image_info(
        os.path.join(
            config.project_path,
            config.data.folder,
            config.data.train_folder,
        ),
        combine_classes_threshold=config.data.combine_classes_threshold,
    )

    print('plant data split')
    plant_data_split = (
        None
        if config.training.use_all_data
        else data.get_labeled_data_split(
            plant_data_image_info=plant_data_image_info,
            val_size=config.training.val_size,
            test_size=config.training.test_size,
        )
    )
    print('non plant data split')
    non_plant_data_split = (
        None
        if config.training.use_all_data
        else data.get_unlabeled_data_split(
            os.path.join(
                config.project_path,
                config.data.folder,
                config.data.other.folder,
            ),
            config.training.val_size,
            config.training.test_size,
        )
    )

    print('plant tree')
    plant_tree = read_plant_taxonomy(config)
    _, num_labels_genus, num_labels_family = get_plant_tree_number(plant_tree)

    print('combine treshlod')
    if config.data.combine_classes_threshold > 0:
        used_species_ids = {
            info.species_id
            for info in plant_data_image_info
            if info.species_id not in rare_species
        }
        species_id_to_index = {
            sid: idx + 1 for idx, sid in enumerate(sorted(used_species_ids))
        }
        for sid in rare_species:
            species_id_to_index[sid] = 0
        species_index_to_id = {
            species_id_to_index[sid]: sid for sid in used_species_ids
        }
        assert 0 not in species_index_to_id
        assert 0 not in species_id_to_index
        species_index_to_id[0] = 0
    else:
        species_id_to_index = {
            sid: idx
            for idx, sid in enumerate(
                sorted({info.species_id for info in plant_data_image_info})
            )
        }
        species_index_to_id = {idx: sid for sid, idx in species_id_to_index.items()}


    print('data manager')
    # Initialize data management

    plant_indices = (
        getattr(plant_data_split, "train_indices", None)
        if plant_data_split
        else None
    )
    non_plant_indices = (
        getattr(non_plant_data_split, f"train_indices", None)
        if non_plant_data_split
        else None
    )

    image_folder_other = os.path.join(
        config.project_path,
        config.data.folder,
        config.data.other.folder,
    )
    image_size = (config.image_width, config.image_height)
    plant_dataset = PlantDataset(
        plant_data_image_info=plant_data_image_info,
        image_size=image_size,
        indices=plant_indices,
    )
    nonplant_dataset = NonPlantDataset(
        image_folder=image_folder_other,
        image_size=image_size,
        indices=non_plant_indices,
    )

    X = []
    y = []
    for iteration, (batch_plant, batch_nonplant) in tqdm(
        enumerate(zip(plant_dataset, nonplant_dataset)),
            desc="Bitches",
            total=min(len(plant_dataset),len(nonplant_dataset)),
    ):
        
        image_plant, _, _ = batch_plant
        image_nonplant, _, _ = batch_nonplant
        
        # Apply augmentation
        image_plant = image_plant.numpy().reshape(3, -1)
        image_plant = np.round(10 * image_plant, decimals=0).astype(int)
        image_plant[1] *= 11
        image_plant[2] *= 121
        image_plant = image_plant.sum(axis=0)
        image_plant = np.bincount(image_plant, minlength=11**3)

        image_nonplant = image_nonplant.numpy().reshape(3, -1)
        image_nonplant = np.round(10 * image_nonplant, decimals=0).astype(int)
        image_nonplant[1] *= 11
        image_nonplant[2] *= 121
        image_nonplant = image_nonplant.sum(axis=0)
        image_nonplant = np.bincount(image_nonplant, minlength=11**3)
        
        
        
        X.extend([image_plant, image_nonplant])
        y.extend([1,0])

    X = np.array(X)
    y = np.array(y)

    gc.collect()
    
    #logr = LogisticRegression(class_weight='balanced').fit(X,y)
    #y_pred = logr.predict(X)
    #A = precision_recall_fscore_support(y,y_pred)
    #print(A)

    print('Random Forest')
    
    rndmfrst = RandomForestClassifier().fit(X,y)
    y_pred = rndmfrst.predict(X)
    A = precision_recall_fscore_support(y,y_pred)
    print(A)

    plant_indices = (
        getattr(plant_data_split, "test_indices", None)
        if plant_data_split
        else None
    )
    non_plant_indices = (
        getattr(non_plant_data_split, f"test_indices", None)
        if non_plant_data_split
        else None
    )

    image_folder_other = os.path.join(
        config.project_path,
        config.data.folder,
        config.data.other.folder,
    )
    image_size = (config.image_width, config.image_height)
    plant_dataset = PlantDataset(
        plant_data_image_info=plant_data_image_info,
        image_size=image_size,
        indices=plant_indices,
    )
    nonplant_dataset = NonPlantDataset(
        image_folder=image_folder_other,
        image_size=image_size,
        indices=non_plant_indices,
    )

    gc.collect()

    X_test = []
    y_test = []
    for iteration, (batch_plant, batch_nonplant) in tqdm(
        enumerate(zip(plant_dataset, nonplant_dataset)),
            desc="Bitches",
            total=min(len(plant_dataset),len(nonplant_dataset)),
    ):
        
        image_plant, _, _ = batch_plant
        image_nonplant, _, _ = batch_nonplant
        
        # Apply augmentation
        
        image_plant = image_plant.numpy().reshape(3, -1)
        image_plant = np.round(10 * image_plant, decimals=0).astype(int)
        image_plant[1] *= 11
        image_plant[2] *= 121
        image_plant = image_plant.sum(axis=0)
        image_plant = np.bincount(image_plant, minlength=11**3)

        
        image_nonplant = image_nonplant.numpy().reshape(3, -1)
        image_nonplant = np.round(10 * image_nonplant, decimals=0).astype(int)
        image_nonplant[1] *= 11
        image_nonplant[2] *= 121
        image_nonplant = image_nonplant.sum(axis=0)
        image_nonplant = np.bincount(image_nonplant, minlength=11**3)
        
        
        X_test.extend([image_plant, image_nonplant])
        y_test.extend([1,0])

        
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    gc.collect()

    #y_pred = logr.predict(X_test)
    #A = precision_recall_fscore_support(y_test,y_pred)
    #print(A)

    print('Random Forest')
    
    y_pred = rndmfrst.predict(X_test)
    A = precision_recall_fscore_support(y_test,y_pred)
    print(A)
    


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config"
)
def main(
    config: DictConfig,
) -> None:
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        config.project_name,
        config=OmegaConf.to_container(config),
        init_kwargs={
            "wandb": {
                "name": f"{config.models.name}_{'pretrained' if config.models.pretrained else 'from-scratch'}",
                "mode": "disabled",
            }
        },
    )

    if accelerator.is_main_process:
        define_metrics()

    # NOTE: accelerator logs on main process only -> loss from only one GPU is logged
    # Gathering loss from all GPUs will slow down training
    pipeline(config, accelerator)

    accelerator.end_training()


if __name__ == "__main__":
    main()
