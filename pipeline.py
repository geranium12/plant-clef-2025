import os

import hydra
import timm
from accelerate import Accelerator
from omegaconf import (
    DictConfig,
    OmegaConf,
)
from torch.utils.data import DataLoader

import src.data as data
from src import augmentation, prediction, submission, training
from src.data_manager import DataManager
from src.merged_model import MergedModel
from src.utils import ModelInfo, define_metrics, load_model
from utils.build_hierarchies import (
    get_organ_number,
    get_plant_tree_number,
    read_plant_taxonomy,
)


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

    if config.merge.enabled:
        species_model = load_model(
            model_config=config.merge.species_model,
            df_metadata=df_metadata,
            num_species=len(species_id_to_index),
            num_genus=num_labels_genus,
            num_family=num_labels_family,
            num_organ=get_organ_number(df_metadata),
            num_plant=1,
            project_path=config.project_path,
        )
        genus_model = load_model(
            model_config=config.merge.genus_model,
            df_metadata=df_metadata,
            num_species=len(species_id_to_index),
            num_genus=num_labels_genus,
            num_family=num_labels_family,
            num_organ=get_organ_number(df_metadata),
            num_plant=1,
            project_path=config.project_path,
        )
        family_model = load_model(
            model_config=config.merge.family_model,
            df_metadata=df_metadata,
            num_species=len(species_id_to_index),
            num_genus=num_labels_genus,
            num_family=num_labels_family,
            num_organ=get_organ_number(df_metadata),
            num_plant=1,
            project_path=config.project_path,
        )
        model = MergedModel(
            species_model=species_model,
            genus_model=genus_model,
            family_model=family_model,
        )
    else:
        model = load_model(
            model_config=config.models,
            df_metadata=df_metadata,
            num_species=len(species_index_to_id),
            num_genus=num_labels_genus,
            num_family=num_labels_family,
            num_organ=get_organ_number(df_metadata),
            num_plant=1,
            project_path=config.project_path,
        )

    accelerator.print(model)

    if config.merge.enabled:
        data_config = timm.data.resolve_model_data_config(model.species_model)
    else:
        data_config = timm.data.resolve_model_data_config(model)
    model_info = ModelInfo(
        input_size=data_config["input_size"][1],  # Assuming (C, H, W)
        mean=data_config["mean"],
        std=data_config["std"],
    )
    accelerator.print(f"Model info: {model_info}")
    accelerator.print(f"{timm.data.create_transform(**data_config, is_training=False)}")

    # Initialize data management
    data_manager = DataManager(
        config=config,
        plant_data_image_info=plant_data_image_info,
        plant_data_split=plant_data_split,
        non_plant_data_split=non_plant_data_split,
        df_metadata=df_metadata,
        species_id_to_idx=species_id_to_index,
        data_config=data_config,
        random_transform=augmentation.get_random_data_augmentation(),
    )

    if not config.merge.enabled:
        model = training.train(
            model=model,
            data_manager=data_manager,
            config=config,
            accelerator=accelerator,
        )
    model = accelerator.prepare(model)

    submission_dataloader = DataLoader(
        dataset=data.MultitileDataset(
            image_folder=os.path.join(
                config.project_path,
                config.data.folder,
                config.data.test_folder,
            ),
            tile_size=model_info.input_size,
            scales=config.prediction.tiling.scales,
            overlaps=config.prediction.tiling.overlaps,
            crop_side_percent=config.prediction.crop_side_percent,
        ),
        batch_size=1,  # config.training.batch_size, TODO: FIX
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    submission_dataloader = accelerator.prepare(submission_dataloader)

    image_predictions = prediction.predict(
        config=config,
        dataloader=submission_dataloader,
        model=model,
        batch_size=config.training.batch_size,
        species_index_to_id=species_index_to_id,
        species_id_to_index=species_id_to_index,
        accelerator=accelerator,
        transform_patch=timm.data.create_transform(**data_config, is_training=False),
    )

    if config.prediction.filter_species_threshold > 0:
        # We recompute rare species, since the filter_species_threshold may differ from the threshold used for training
        rare_species = data._get_rare_classes(
            plant_data_image_info, config.prediction.filter_species_threshold
        )
        image_predictions = {
            k: [i for i in v if i not in rare_species]
            for k, v in image_predictions.items()
        }

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
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        config.project_name,
        config=OmegaConf.to_container(config),
        init_kwargs={
            "wandb": {
                "name": f"{config.models.name}_{'pretrained' if config.models.pretrained else 'from-scratch'}",
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
