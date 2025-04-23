import os

import hydra
from accelerate import Accelerator
from omegaconf import (
    DictConfig,
    OmegaConf,
)
from torch.utils.data import DataLoader

import src.data as data
import wandb
from src import prediction, submission, training
from src.utils import load_model
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
    device = accelerator.device

    (
        df_metadata,
        df_species_ids,
        class_map,
    ) = data.load(config)

    plant_data_split = (
        None
        if config.training.use_all_data
        else data.get_labeled_data_split(
            os.path.join(
                config.project_path,
                config.data.folder,
                config.data.train_folder,
            ),
            config.training.val_size,
            config.training.test_size,
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

    model = load_model(config=config, df_species_ids=df_species_ids)
    model = ViTMultiHeadClassifier(
        backbone=model,
        num_labels_organ=get_organ_number(df_metadata),
        num_labels_genus=num_labels_genus,
        num_labels_family=num_labels_family,
        num_labels_plant=1,
    )
    print(model)

    model, model_info = training.train(
        model=model,
        config=config,
        df_metadata=df_metadata,
        accelerator=accelerator,
        plant_data_split=plant_data_split,
        non_plant_data_split=non_plant_data_split,
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
        class_map=class_map,
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
    accelerator = Accelerator()

    wandb.init(
        project=config.project_name,
        name=f"{config.models.name}_{'pretrained' if config.models.pretrained else 'from-scratch'}",
        config=OmegaConf.to_container(config),  # type: ignore[arg-type]
        reinit=False if config is None else True,
    )

    pipeline(config, accelerator)

    wandb.finish()


if __name__ == "__main__":
    main()
