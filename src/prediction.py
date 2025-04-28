import os
import time

import numpy as np
import pandas as pd

from omegaconf import DictConfig

from utils.build_hierarchies import read_plant_taxonomy

import torch
import torchvision.transforms as ttransforms
from accelerate import Accelerator
from torch.amp import autocast
from torch.utils.data import (
    DataLoader,
)

from src.data import (
    PatchDataset,
)
from src.training import (
    ModelInfo,
)

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


def predict(
    dataloader: DataLoader,
    model: torch.nn.Module,
    model_info: ModelInfo,
    batch_size: int,
    top_k_tile: int,
    class_map: dict[int, int],
    min_score: float,
    accelerator: Accelerator,
) -> dict[str, list[int]]:
    image_predictions: dict[str, list[int]] = {}

    # Initialize batch time tracking
    batch_time = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (
            patches,
            image_path,
        ) in enumerate(dataloader):
            image_results: dict[int, float] = {}
            quadrat_id = os.path.splitext(os.path.basename(image_path[0]))[0]
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

            for batch_patches in patch_loader:
                with autocast("cuda"):
                    outputs = model(batch_patches)  # Perform inference on the batch
                    probabilities = torch.nn.functional.softmax(
                        outputs["logits_species"],
                        dim=1,
                    )

                    # Get the top-k indices and probabilities
                    (
                        top_probs,
                        top_indices,
                    ) = torch.topk(
                        probabilities,
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
                            species_id = class_map[top_idx]
                            # Update the results dictionary only if the probability is higher
                            if top_prob > min_score:
                                if (
                                    top_idx not in image_results
                                    or image_results[top_idx] < top_prob
                                ):
                                    image_results[species_id] = top_prob
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

    return image_predictions

def predict_all(
    dataloader: DataLoader,
    model: torch.nn.Module,
    model_info: ModelInfo,
    batch_size: int,
    top_k_tile: int,
    class_map: dict[int, int],
    min_score: float,
    accelerator: Accelerator,
    config: DictConfig
) -> dict[str, list[int]]:
    image_predictions: dict[str, list[int]] = {}
    
    plant_tree = read_plant_taxonomy(config)
    
    folder_path = check_utils_folder(config)
    
    species_mapping = pd.read_csv(
                os.path.join(
            folder_path,
                    'species_ids.csv',
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
    
    # Initialize batch time tracking
    batch_time = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (
            patches,
            image_path,
        ) in enumerate(dataloader):
            image_results: dict[int, float] = {}
            quadrat_id = os.path.splitext(os.path.basename(image_path[0]))[0]
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

            for batch_patches in patch_loader:
                with autocast("cuda"):
                    outputs = model(batch_patches)  # Perform inference on the batch
                    probabilities_species = torch.nn.functional.softmax(
                        outputs["logits_species"],
                        dim=1,
                    )

                    probabilities_genus = torch.nn.functional.softmax(
                        outputs["logits_genus"],
                        dim=1
                    )

                    probabilities_family = torch.nn.functional.softmax(
                        outputs["logits_family"],
                        dim=1
                    )

                    probabilities = probabilities_species \
                        * probabilities_genus[species_to_genus] \
                        * probabilities_family[species_to_family]


                    # Get the top-k indices and probabilities
                    (
                        top_probs,
                        top_indices,
                    ) = torch.topk(
                        probabilities,
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
                            species_id = class_map[top_idx]
                            # Update the results dictionary only if the probability is higher
                            if top_prob > min_score:
                                if (
                                    top_idx not in image_results
                                    or image_results[top_idx] < top_prob
                                ):
                                    image_results[species_id] = top_prob
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

    return image_predictions
