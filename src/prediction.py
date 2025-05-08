import os
import pickle
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as ttransforms
from accelerate import Accelerator
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from torch.amp import autocast
from torch.utils.data import (
    DataLoader,
)

from src.data import PatchDataset
from src.training import (
    ModelInfo,
)
from src.utils import family_name_to_id, genus_name_to_id, species_id_to_name
from utils.build_hierarchies import (
    check_utils_folder,
    get_genus_family_from_species,
    read_plant_taxonomy,
)


class AverageMeter:
    def __init__(
        self,
    ) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(
        self,
        val: float,
        n: int = 1,
    ) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def top_k_tile_prediction(
    tiles_probabilities: torch.Tensor,
    species_index_to_id: dict[int, int],
    top_k_tile: int,
    min_score: float,
) -> dict[int, float]:
    image_results: dict[int, float] = {}

    # Get the top-k indices and probabilities
    (
        top_probs,
        top_indices,
    ) = torch.topk(
        tiles_probabilities,
        top_k_tile,
    )
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    for (
        top_tile_indices,
        top_tile_probs,
    ) in zip(top_indices, top_probs):
        for (
            top_idx,
            top_prob,
        ) in zip(top_tile_indices, top_tile_probs):
            species_id = species_index_to_id[top_idx]
            # Update the results dictionary only if the probability is higher
            if top_prob > min_score:
                if top_idx not in image_results or image_results[species_id] < top_prob:
                    image_results[species_id] = top_prob

    return image_results


def bma_prediction(
    tiles_probabilities: torch.Tensor,
    species_index_to_id: dict[int, int],
    z_score_threshold: float,
) -> dict[int, float]:
    # From "Patch-wise Inference using Pre-trained Vision Transformers: NEUON Submission to PlantCLEF 2024" Figure 8
    d = tiles_probabilities.shape[1]

    ss = (
        1
        / (d - 1)
        * torch.sum(
            (tiles_probabilities - tiles_probabilities.mean(dim=1, keepdim=True)) ** 2,
            dim=1,
        )
    )
    var = torch.abs(torch.log10(ss))
    confidence = torch.sqrt((torch.max(var) + 0.5 - var) / (torch.max(var) + 0.5))
    p_i_d = confidence / torch.sum(confidence, dim=0, keepdim=True)
    weighted_probabilities = tiles_probabilities * p_i_d.unsqueeze(1)
    image_probabilities = torch.sum(weighted_probabilities, dim=0)

    z_scores = (
        image_probabilities - torch.mean(image_probabilities, dim=0)
    ) / torch.std(image_probabilities, dim=0)

    return {
        species_index_to_id[idx]: prob
        for idx, prob in enumerate(image_probabilities)
        if z_scores[idx] > z_score_threshold
    }


def predict(
    config: DictConfig,
    dataloader: DataLoader,
    model: torch.nn.Module,
    model_info: ModelInfo,
    batch_size: int,
    species_index_to_id: dict[int, int],
    species_id_to_index: dict[int, int],
    accelerator: Accelerator,
) -> dict[str, list[int]]:
    image_predictions: dict[str, list[int]] = {}

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

    species_to_other = sorted(
        [
            (
                species_index,
                get_genus_family_from_species(
                    plant_tree, species_id_to_name(species_id, species_mapping)
                ),
            )
            for species_index, species_id in species_index_to_id.items()
            if species_id != 0
        ]
    )

    species_to_genus_list = []
    species_to_family_list = []
    for _, (genus, family) in species_to_other:
        gid = genus_name_to_id(genus, genus_mapping)
        fid = family_name_to_id(family, family_mapping)
        species_to_genus_list.append(gid)
        species_to_family_list.append(fid)

    species_to_genus = torch.tensor(species_to_genus_list, dtype=torch.int64)
    species_to_family = torch.tensor(species_to_family_list, dtype=torch.int64)

    if config.prediction.predict_no_plant:
        with open("./forest.pkl", "rb") as fl:
            noplant_predictor = pickle.load(fl)

    # Initialize batch time tracking
    batch_time = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (
            patches,
            image_path,
        ) in enumerate(dataloader):
            quadrat_id = os.path.splitext(os.path.basename(image_path[0]))[0]

            if config.prediction.predict_no_plant:
                nonplant_threshold = 0.5
                shps = patches[0].shape
                color_counts = np.round((10 * patches[0]).cpu().numpy()).reshape(
                    shps[0], 3, -1
                )
                color_counts[:, 1] *= 11
                color_counts[:, 2] *= 121
                color_counts = color_counts.sum(axis=1).astype("int")
                color_counts = np.apply_along_axis(
                    lambda x: np.bincount(x, minlength=11**3), axis=1, arr=color_counts
                )
                prediction = noplant_predictor.predict_proba(color_counts)[
                    :, noplant_predictor.classes_ == 1
                ].squeeze()
                indices = (prediction > nonplant_threshold) | (
                    np.argsort(-prediction) < 2
                )
                new_patches = patches[0][indices]
                patches = [new_patches]

            transform_patch = ttransforms.Normalize(
                mean=model_info.mean,
                std=model_info.std,
            )
            patch_dataset = PatchDataset(
                patches[0],
                transform=transform_patch,
            )
            patch_loader = DataLoader(
                patch_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

            image_tile_probabilities: torch.Tensor = None

            for batch_patches in patch_loader:
                with autocast("cuda"):
                    outputs = model(batch_patches)  # Perform inference on the batch
                    probabilities_species = torch.nn.functional.softmax(
                        outputs["logits_species"],
                        dim=1,
                    )

                    if config.prediction.use_genus_and_family:
                        probabilities_genus = torch.nn.functional.softmax(
                            outputs["logits_genus"], dim=1
                        )

                        probabilities_family = torch.nn.functional.softmax(
                            outputs["logits_family"], dim=1
                        )

                        species_to_genus = species_to_genus.to(
                            probabilities_species.device
                        )
                        species_to_family = species_to_family.to(
                            probabilities_species.device
                        )

                        genus_probs = probabilities_genus.gather(
                            1,
                            species_to_genus.unsqueeze(0).expand(
                                probabilities_species.shape[0], -1
                            ),
                        )
                        family_probs = probabilities_family.gather(
                            1,
                            species_to_family.unsqueeze(0).expand(
                                probabilities_species.shape[0], -1
                            ),
                        )
                        probabilities = probabilities_species.clone()
                        if config.data.combine_classes_threshold == 0:
                            probabilities *= genus_probs * family_probs
                        else:
                            probabilities[:, 1:] *= genus_probs * family_probs

                    else:
                        probabilities = probabilities_species

                    image_tile_probabilities = (
                        probabilities
                        if image_tile_probabilities is None
                        else torch.cat((image_tile_probabilities, probabilities), dim=0)
                    )

            image_results: dict[int, float] = {}
            match config.prediction.method:
                case "top_k_tile":
                    image_results = top_k_tile_prediction(
                        image_tile_probabilities,
                        species_index_to_id,
                        top_k_tile=config.prediction.top_k_tile.k,
                        min_score=config.prediction.top_k_tile.min_score,
                    )
                case "BMA":
                    image_results = bma_prediction(
                        image_tile_probabilities,
                        species_index_to_id,
                        z_score_threshold=config.prediction.BMA.z_score_threshold,
                    )
                case _:
                    raise ValueError(
                        f"Unknown prediction method: {config.prediction.method}"
                    )

            if config.prediction.filter_genus:
                # Predict only the top species per genus
                filtered_results: dict[int, tuple[int, float]] = {}
                for species_id, prob in image_results.items():
                    species_idx = species_id_to_index[species_id]
                    genus_id = species_to_genus[species_idx].item()
                    if (
                        genus_id not in filtered_results
                        or prob > filtered_results[genus_id][1]
                    ):
                        filtered_results[genus_id] = (species_id, prob)
                image_results = {
                    species_id: prob for species_id, prob in filtered_results.values()
                }

            # store the prediction
            image_predictions[quadrat_id] = list(image_results.keys())

            batch_time.update(time.time() - end)
            end = time.time()

            # Log info at specified frequency
            if batch_idx % 10 == 0:  # You can set your log frequency here
                accelerator.print(
                    f"Predict: [{batch_idx}/{len(dataloader)}] "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})"
                )

    if config.prediction.combine_same_plot_threshold > 0:

        def extract_plot_name(image_path: str) -> str:
            match image_path:
                case t if "CBN-PdlC" in t or "CBN-Pla" in t:
                    plot_name = t[:-9]
                case t if "GUARDEN-CBNMed" in t:
                    plot_name = t[: t.index("-", 15)]
                case t:
                    return t

            if config.prediction.group_same_plot_by_year:
                year = image_path[-8:-4]
                plot_name = f"{plot_name}_{year}"

            return plot_name

        image_plot_predictions: dict[str, list[list[int]]] = {}
        for image_path, prediction in image_predictions.items():
            plot_name = extract_plot_name(image_path)
            if plot_name not in image_plot_predictions:
                image_plot_predictions[plot_name] = []
            image_plot_predictions[plot_name].append(prediction)

        combined_plot_predictions = {}
        for image_path in image_predictions.keys():
            predictions = image_plot_predictions[extract_plot_name(image_path)]
            if len(predictions) == 1:
                combined_plot_predictions[image_path] = predictions[0]
                continue

            prediction_count: dict[int, int] = {}
            for prediction in predictions:
                for predicted_species in prediction:
                    prediction_count[predicted_species] = (
                        prediction_count.get(predicted_species, 0) + 1
                    )

            # Keep prediction if combining would yield empty prediction
            if not any(
                count >= config.prediction.combine_same_plot_threshold
                for count in prediction_count.values()
            ):
                combined_plot_predictions[image_path] = predictions[0]
                continue

            combined_prediction = [
                predicted_species
                for predicted_species, count in prediction_count.items()
                if count >= config.prediction.combine_same_plot_threshold
            ]
            combined_plot_predictions[image_path] = combined_prediction
        image_predictions = combined_plot_predictions

    return image_predictions
