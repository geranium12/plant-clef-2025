import os

import numpy as np
import pandas as pd

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
    check_utils_folder,
    get_genus_family_from_species,
)
from src.utils import (
    species_id_to_name,
    genus_name_to_id,
    family_name_to_id
)


def pipeline(
    config: DictConfig,
    accelerator: Accelerator,
) -> None:
    df_metadata = data.load_metadata(config)

    plant_data_image_info = data.get_plant_data_image_info(
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

    species_id_to_index = {
        sid: idx
        for idx, sid in enumerate(
            sorted({info.species_id for info in plant_data_image_info})
        )
    }
    species_index_to_id = {idx: sid for sid, idx in species_id_to_index.items()}

    model = load_model(
        config=config,
        num_classes=len(df_metadata["species_id"].unique()),
    )
    model = ViTMultiHeadClassifier(
        backbone=model,
        num_labels_organ=get_organ_number(df_metadata),
        num_labels_genus=num_labels_genus,
        num_labels_family=num_labels_family,
        num_labels_plant=1,
        num_labels_species=len(species_id_to_index),
    )
    print(model)

    # Initialize data management
    data_manager = DataManager(
        config=config,
        plant_data_image_info=plant_data_image_info,
        plant_data_split=plant_data_split,
        non_plant_data_split=non_plant_data_split,
        df_metadata=df_metadata,
        species_id_to_idx=species_id_to_index,
    )

    model, model_info = training.train(
        model=model,
        data_manager=data_manager,
        config=config,
        accelerator=accelerator,
    )

    submission_dataloader = DataLoader(
        dataset=data.TestDataset(
            image_folder=os.path.join(
                config.project_path,
                config.data.folder,
                config.data.test_folder,
            ),
            patch_size=model_info.input_size,
            stride=int(model_info.input_size / 2),
            use_pad=True,
        ),
        batch_size=1,  # config.training.batch_size, TODO: FIX
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    submission_dataloader = accelerator.prepare(submission_dataloader)

    image_predictions = prediction.predict(
        dataloader=submission_dataloader,
        model=model,
        model_info=model_info,
        batch_size=config.training.batch_size,
        top_k_tile=config.training.top_k_tile,
        class_map=species_index_to_id,
        min_score=config.training.min_score,
        accelerator=accelerator,
    )

    submission.submit(
        config,
        image_predictions,
    )


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config",
)
def main(
    config: DictConfig,
) -> None:
    #accelerator = Accelerator(log_with="wandb")
    #accelerator.init_trackers(
    #    config.project_name,
    #    config=OmegaConf.to_container(config),
    #    init_kwargs={
    #        "wandb": {
    #            "name": f"{config.models.name}_{'pretrained' if config.models.pretrained else 'from-scratch'}",
    #        }
    #    },
    #)

    #if accelerator.is_main_process:
    #    define_metrics()

    # NOTE: accelerator logs on main process only -> loss from only one GPU is logged
    # Gathering loss from all GPUs will slow down training
    #pipeline(config, accelerator)

    #accelerator.end_training()
    
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

    plant_data_image_info = data.get_plant_data_image_info(
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

    species_to_other = sorted([
        (species_index, get_genus_family_from_species(plant_tree, species_id_to_name(species_id, species_mapping)))
        for species_id, species_index in species_id_to_index.items()
    ])

    species_to_genus = []
    species_to_family = []
    for _, (genus, family) in species_to_other:
        gid = genus_name_to_id(genus, genus_mapping)
        fid = family_name_to_id(family, family_mapping)
        species_to_genus.append(gid)
        species_to_family.append(fid)


    species_to_genus = np.array(species_to_genus)
    species_to_family = np.array(species_to_family)

    breakpoint()
    

if __name__ == "__main__":
    main()
