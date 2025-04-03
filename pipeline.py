import hydra
import torch
from omegaconf import (
    DictConfig,
    OmegaConf,
)
from torch.utils.data import (
    DataLoader,
)

import src.data as data
import src.prediction as prediction
import src.submission as submission
import src.training as training
import wandb
from src.data import (
    TestDataset,
)


def pipeline(
    config: DictConfig,
    device: torch.device,
) -> None:
    (
        df_metadata,
        df_species_ids,
        class_map,
    ) = data.load(config)

    model, model_info = training.train(
        config=config,
        device=device,
        df_species_ids=df_species_ids,
    )

    batch_size = 64
    min_score = 0.1
    top_k_tile = 2

    dataloader = DataLoader(
        dataset=TestDataset(
            image_folder=config.project_path
            + config.data.folder
            + config.data.test_folder
            + config.data.test_images_folder,
            patch_size=model_info.input_size,
            stride=int(model_info.input_size / 2),
            use_pad=True,
        ),
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    )

    image_predictions = prediction.predict(
        dataloader=dataloader,
        model=model,
        model_info=model_info,
        batch_size=batch_size,
        device=device,
        top_k_tile=top_k_tile,
        class_map=class_map,
        min_score=min_score,
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

    device = torch.device("cuda")

    pipeline(config, device)

    wandb.finish()


if __name__ == "__main__":
    main()
