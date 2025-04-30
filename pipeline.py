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
from src import prediction, submission, training
from src.data_manager import DataManager
from src.utils import ModelInfo, define_metrics, load_model
from src.vit_multi_head_classifier import ViTMultiHeadClassifier
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
    accelerator.print(model)

    data_config = timm.data.resolve_model_data_config(model)
    model_info = ModelInfo(
        input_size=data_config["input_size"][1],  # Assuming (C, H, W)
        mean=data_config["mean"],
        std=data_config["std"],
    )
    accelerator.print(f"Model info: {model_info}")

    # Initialize data management
    data_manager = DataManager(
        config=config,
        plant_data_image_info=plant_data_image_info,
        plant_data_split=plant_data_split,
        non_plant_data_split=non_plant_data_split,
        df_metadata=df_metadata,
        species_id_to_idx=species_id_to_index,
        data_config=data_config,
    )

    model = training.train(
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
        config=config,
        dataloader=submission_dataloader,
        model=model,
        batch_size=config.training.batch_size,
        top_k_tile=config.training.top_k_tile,
        species_index_to_id=species_index_to_id,
        species_id_to_index=species_id_to_index,
        min_score=config.training.min_score,
        accelerator=accelerator,
        transform_patch=timm.data.create_transform(**data_config, is_training=False),
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
