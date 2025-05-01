import math
import os
import random
from dataclasses import astuple, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as ttransforms
from kornia.contrib import (
    compute_padding,
    extract_tensor_patches,
)
from omegaconf import (
    DictConfig,
)
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    Dataset,
)
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PatchDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        patches: torch.Tensor,
        transform: ttransforms.transforms = None,
    ) -> None:
        self.patches = patches.squeeze(0)
        self.transform = transform

    def __len__(self) -> int:
        return self.patches.size(0)  # type: ignore[no-any-return]

    def __getitem__(self, idx: int) -> torch.Tensor:
        patch = self.patches[idx]

        if self.transform:
            patch = self.transform(patch)
        return patch


class MultitileDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        image_folder: str,
        tile_size: int = 518,
        scales: list[float] | None = None,
        overlaps: list[float] | None = None,
    ) -> None:
        self.image_folder = image_folder
        self.image_paths = [
            os.path.join(
                image_folder,
                f,
            )
            for f in os.listdir(image_folder)
        ]
        self.transform = ttransforms.ToTensor()

        self.scales = scales if scales is not None else [1.0]
        self.overlaps = overlaps if overlaps is not None else [0.0]
        assert len(self.scales) == len(self.overlaps), (
            "Same number of scales and overlaps should be provided"
        )
        self.tile_size = tile_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image).unsqueeze(0)

        h, w = image.shape[-2:]
        patches_list = []
        for scale, overlap in zip(self.scales, self.overlaps):
            window_size = (math.ceil(h / scale), math.ceil(w / scale))
            stride = (
                math.ceil(window_size[0] * (1.0 - overlap)),
                math.ceil(window_size[1] * (1.0 - overlap)),
            )
            pad = compute_padding(
                original_size=(h, w),
                window_size=window_size,
                stride=stride,
            )
            patches = extract_tensor_patches(
                image,
                window_size=window_size,
                stride=stride,
                padding=pad,
            )
            patches = patches.squeeze(0)
            patches = torch.nn.functional.interpolate(
                patches,
                size=(self.tile_size, self.tile_size),
                mode="bilinear",
                align_corners=False,
            )
            patches_list.append(patches)

        all_patches = torch.cat(patches_list, dim=0)
        return (all_patches, image_path)


@dataclass
class ImageSampleInfo:
    species_id: int
    image_path: str


def _get_rare_classes(samples: list[ImageSampleInfo], threshold: int) -> set[int]:
    if threshold <= 0:
        return set()

    counts: dict[int, int] = {}
    for sample in samples:
        counts[sample.species_id] = counts.get(sample.species_id, 0) + 1

    return {sid for sid, count in counts.items() if count <= threshold}


def get_plant_data_image_info(
    image_folder: str, combine_classes_threshold: int = 0
) -> tuple[list[ImageSampleInfo], set[int]]:
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    samples: list[ImageSampleInfo] = []  # List of (class_name, image_path) pairs.
    for cls in sorted(os.listdir(image_folder)):
        cls_folder = os.path.join(image_folder, cls)
        if not os.path.isdir(cls_folder):
            continue
        for root, _, files in os.walk(cls_folder):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    path = os.path.join(root, file)
                    samples.append(
                        ImageSampleInfo(species_id=int(cls), image_path=path)
                    )

    rare_classes = _get_rare_classes(samples, combine_classes_threshold)

    return samples, rare_classes


def get_image_paths(image_folder: str) -> list[str]:
    """Walks the directory tree and returns a list of all image paths.

    Args:
        image_folder (str): The folder to walk

    Returns:
        list[str]: A list of image paths.
    """
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    image_paths: list[str] = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


# This types of dataset has species_id and contains only plants
class PlantDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        plant_data_image_info: list[ImageSampleInfo],
        image_size: tuple[int, int] = (400, 400),
        transform: ttransforms.transforms = None,
        indices: list[int] | None = None,
    ) -> None:
        self.image_size = image_size
        self.transform = transform if transform is not None else ttransforms.ToTensor()

        self.samples = plant_data_image_info
        if indices is not None:
            assert all(i < len(self.samples) for i in indices), (
                "All indices must be less than the number of samples."
            )
            self.samples = [self.samples[i] for i in indices if i < len(self.samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        image_path = sample.image_path
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize(self.image_size)
        image = self.transform(image)
        image_name = os.path.basename(image_path)
        return (image, sample.species_id, image_name)


# This types of dataset has no species_id and contains only non-plants
class NonPlantDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        image_folder: str,
        image_size: tuple[int, int] = (400, 400),
        transform: ttransforms.transforms = None,
        indices: list[int] | None = None,
    ) -> None:
        self.image_size = image_size
        self.transform = transform if transform is not None else ttransforms.ToTensor()

        self.samples = get_image_paths(image_folder)
        if indices is not None:
            assert all(i < len(self.samples) for i in indices), (
                "All indices must be less than the number of samples."
            )
            self.samples = [self.samples[i] for i in indices if i < len(self.samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        image_path = self.samples[idx]
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize(self.image_size)
        image = self.transform(image)
        image_name = os.path.basename(image_path)
        return (image, -1, image_name)


class ConcatenatedDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        datasets: list[Dataset],
    ) -> None:
        self.datasets = datasets
        self.end_indices = np.cumsum([len(dataset) for dataset in datasets])

    def __len__(self) -> int:
        return int(self.end_indices[-1])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str | None]:
        idx = (idx % len(self) + len(self)) % len(self)
        for i, end_idx in enumerate(self.end_indices):
            if idx < end_idx:
                dataset = self.datasets[i]
                dataset_idx = idx if i == 0 else idx - self.end_indices[i - 1]
                return dataset[dataset_idx]  # type: ignore[no-any-return]
        raise IndexError("Index out of range")


@dataclass
class DataSplit:
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(astuple(self))


def get_unlabeled_data_split(
    image_folder: str, val_size: float = 0.2, test_size: float = 0.1
) -> DataSplit:
    """Generates split indices for training, validation, and test sets.

    Args:
        image_folder (str): The folder containing the image folders.

    Returns:
        DataSplit: The train, validation, and test indices.
    """
    samples = get_image_paths(image_folder)
    indices = list(range(len(samples)))
    assert val_size + test_size < 1.0, (
        "Validation and test sizes must sum to less than 1.0"
    )

    train_test_indices, val_indices = train_test_split(
        indices,
        test_size=val_size,
        random_state=42,
    )
    train_indices, test_indices = train_test_split(
        train_test_indices,
        test_size=test_size / (1.0 - val_size),
        random_state=42,
    )

    return DataSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


def get_labeled_data_split(
    plant_data_image_info: list[ImageSampleInfo],
    val_size: float = 0.2,
    test_size: float = 0.1,
) -> DataSplit:
    """Generates split indices for training, validation, and test sets.

    Args:
        image_folder (str): The folder containing the images.

    Returns:
        DataSplit: The train, validation, and test indices.
    """
    indices = list(range(len(plant_data_image_info)))
    assert val_size + test_size < 1.0, (
        "Validation and test sizes must sum to less than 1.0"
    )

    # A lot of classes have only one image, so we need to treat those differently
    # We also treat those with 2 images differently, since we want three splits
    class_counts: dict[int, int] = {}
    for info in plant_data_image_info:
        class_counts[info.species_id] = class_counts.get(info.species_id, 0) + 1

    few_image_indices = [
        i for i in indices if class_counts[plant_data_image_info[i].species_id] < 3
    ]
    multi_image_indices = [
        i for i in indices if class_counts[plant_data_image_info[i].species_id] >= 3
    ]

    # FIXME: There is a small chance, that for a class with 3 images, 2 are put into val and 1 remains in train_test. In this case an exception will be thrown. I don't want to implement stratified though...
    train_test_indices, val_indices = train_test_split(
        multi_image_indices,
        test_size=val_size,
        random_state=42,
        stratify=[plant_data_image_info[i].species_id for i in multi_image_indices],
    )
    train_indices, test_indices = train_test_split(
        train_test_indices,
        test_size=test_size / (1.0 - val_size),
        random_state=42,
        stratify=[plant_data_image_info[i].species_id for i in train_test_indices],
    )

    for i in few_image_indices:
        r = random.random()
        if r < val_size:
            val_indices.append(i)
        elif r < val_size + test_size:
            test_indices.append(i)
        else:
            train_indices.append(i)

    return DataSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


def read_csv_in_chunks(path: str, **read_params: Any) -> pd.DataFrame:
    if "chunksize" not in read_params or read_params["chunksize"] < 1:
        read_params["chunksize"] = 10000
    chunks = []
    for _, chunk in enumerate(
        tqdm(
            pd.read_csv(path, **read_params),
            desc="Reading CSV",
        )
    ):
        chunks.append(chunk)
    concat_df = pd.concat(chunks, axis=0)
    del chunks
    return concat_df


def load_metadata(
    config: DictConfig,
) -> pd.DataFrame:
    metadata_path = os.path.join(
        config.project_path, config.data.folder, config.data.metadata.folder
    )

    df_metadata = read_csv_in_chunks(
        path=os.path.join(metadata_path, config.data.metadata.training),
        sep=";",
        dtype={"partner": str},
    )

    return df_metadata
