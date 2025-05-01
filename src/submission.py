import csv

import pandas as pd
from omegaconf import (
    DictConfig,
)


def submit(
    config: DictConfig,
    image_predictions: dict[str, list[int]],
) -> None:
    if config.data.combine_classes_threshold > 0:
        # Filter all "other species" predictions
        image_predictions = {
            k: [i for i in v if i != 0] for k, v in image_predictions.items()
        }

    df_run = pd.DataFrame(
        list(image_predictions.items()),
        columns=[
            "quadrat_id",
            "species_ids",
        ],
    )
    df_run["species_ids"] = df_run["species_ids"].apply(str)
    df_run.to_csv(
        config.submission_file,
        sep=",",
        index=False,
        quoting=csv.QUOTE_ALL,
    )
