import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

def pipeline(config: DictConfig) -> None:
    # TODO
    pass

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config",
)
def main(config: DictConfig) -> None:
    print("This is the main function of the pipeline module.")
    print(config)
    wandb.init(
        project=config.project_name,
        name="", # TODO
        config=OmegaConf.to_container(config),
        reinit=False if config is None else True,
    )

    pipeline(config)

    wandb.finish()

if __name__ == "__main__":
    main()