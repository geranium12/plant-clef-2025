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


def pipeline(
    config: DictConfig,
    accelerator: Accelerator,
) -> None:
    df_metadata = data.load_metadata(config)

    plant_data_image_info, rare_species = data.get_plant_data_image_info(
        os.path.join(
            config.project_path,
            config.data.folder,
            config.data.train_folder,
        ),
        combine_classes_threshold=config.data.combine_classes_threshold,
    )

    plant_data_split = (
        None
        if config.training.use_all_data
        else data.get_labeled_data_split(
            plant_data_image_info=plant_data_image_info,
            val_size=config.training.val_size,
            test_size=config.training.test_size,
        )
    )
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

    plant_tree = read_plant_taxonomy(config)
    _, num_labels_genus, num_labels_family = get_plant_tree_number(plant_tree)

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


    # Initialize data management
    data_manager = DataManager(
        config=config,
        plant_data_image_info=plant_data_image_info,
        plant_data_split=plant_data_split,
        non_plant_data_split=non_plant_data_split,
        df_metadata=df_metadata,
        species_id_to_idx=species_id_to_index,
    )


    X = []
    y = []
    for iteration, batch in tqdm(
        enumerate(data_manager.train_dataloader),
            desc="Bitches",
            total=len(data_manager.train_dataloader),
    ):
        
        images, species_labels, images_names = batch
        plant_labels = (species_labels != -1).clone().detach().to(dtype=torch.float32)
        
        # Apply augmentation
        #augmentation = get_random_data_augmentation(config)
        #images = augmentation(images)
        
        # Gather labels
        labels = data_manager.gather_all_labels(
            species_labels, plant_labels, images_names
        )
        X.append(images.cpu().numpy())
        y.append(plant_labels.cpu().numpy())
        if iteration >= 10:
            break

    X = np.concatenate(X, axis=0)
    dims = X.shape
    X = X.reshape(X.shape[0], -1)
    X = np.round(10 * X, decimals=0).astype(int)
    X = X.reshape(X.shape[0], 3, -1)
    X[:,1] *= 11
    X[:,2] *= 121
    X = X.sum(axis=1)
    X = np.apply_along_axis(lambda x: np.bincount(x, minlength=11**3), axis=1, arr=X)
    y = np.concatenate(y, axis=0)
    #pca = PCA(n_components=50)
    #pca.fit(X[y==1.])
    #X = pca.transform(X)
    
    logr = LogisticRegression(class_weight='balanced').fit(X,y)
    y_pred = logr.predict(X)
    A = precision_recall_fscore_support(y,y_pred)
    print(A)
    
    
    rndmfrst = RandomForestClassifier().fit(X,y)
    y_pred = rndmfrst.predict(X)
    A = precision_recall_fscore_support(y,y_pred)
    print(A)

    X_test = []
    y_test = []
    for iteration, batch in tqdm(
        enumerate(data_manager.test_dataloader),
            desc="Bitches",
            total=len(data_manager.test_dataloader),
    ):
        
        images, species_labels, images_names = batch
        plant_labels = (species_labels != -1).clone().detach().to(dtype=torch.float32)
        
        # Apply augmentation
        augmentation = get_random_data_augmentation(config)
        images = augmentation(images)
        
        # Gather labels
        labels = data_manager.gather_all_labels(
            species_labels, plant_labels, images_names
        )
        X_test.append(images.cpu().numpy())
        y_test.append(plant_labels.cpu().numpy())
        if iteration >= 10:
            break

    
    X_test = np.concatenate(X_test, axis=0)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test = np.round(10 * X_test, decimals=0).astype(int)
    X_test = X_test.reshape(X_test.shape[0], 3, -1)
    X_test[:,1] *= 11
    X_test[:,2] *= 121
    X_test = X_test.sum(axis=1)
    X_test = np.apply_along_axis(lambda x: np.bincount(x, minlength=11**3), axis=1, arr=X_test)
    #X_test = pca.transform(X_test)
    y_test = np.concatenate(y_test, axis=0)

    y_pred = logr.predict(X_test)
    A = precision_recall_fscore_support(y_test,y_pred)
    print(A)
    
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
