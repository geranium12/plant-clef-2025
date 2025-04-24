import os

import hydra
import torch
from omegaconf import (
    DictConfig,
    OmegaConf,
)
from torch.utils.data import DataLoader

import src.data as data
import wandb
from src import prediction, submission, training
from src.data_manager import DataManager
from src.utils import load_model, save_model
from src.vit_multi_head_classifier import ViTMultiHeadClassifier
from utils.build_hierarchies import (
    get_organ_number,
    get_plant_tree_number,
    read_plant_taxonomy,
)


def pipeline(
    config: DictConfig,
    device: torch.device,
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

    model = load_model(
        config=config,
        device=device,
        num_classes=len(df_metadata["species_id"].unique()),
    )
    model = ViTMultiHeadClassifier(
        backbone=model,
        num_labels_organ=get_organ_number(df_metadata),
        num_labels_genus=num_labels_genus,
        num_labels_family=num_labels_family,
        num_labels_plant=1,
        num_labels_species=len(species_id_to_index),
        device=device,
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
        device=device,
    )

    save_model(model, config)

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

    image_predictions = prediction.predict(
        dataloader=submission_dataloader,
        model=model,
        model_info=model_info,
        batch_size=config.training.batch_size,
        device=device,
        top_k_tile=config.training.top_k_tile,
        class_map=species_id_to_index,
        min_score=config.training.min_score,
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
    wandb.init(
        project=config.project_name,
        name=f"{config.models.name}_{'pretrained' if config.models.pretrained else 'from-scratch'}",
        config=OmegaConf.to_container(config),  # type: ignore[arg-type]
        reinit=False if config is None else True,
    )

    device = torch.device(config.device)

    pipeline(config, device)

    wandb.finish()


if __name__ == "__main__":
    main()
