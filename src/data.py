import os
import random
from dataclasses import astuple, dataclass
from typing import Optional

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
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    Dataset,
)


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


class TestDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        image_folder: str,
        patch_size: int = 518,
        stride: int = 259,
        use_pad: bool = False,
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
        self.use_pad = use_pad
        self.patch_size = patch_size
        self.stride = stride

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image).unsqueeze(0)

        h, w = image.shape[-2:]

        if self.use_pad:
            pad = compute_padding(
                original_size=(
                    h,
                    w,
                ),
                window_size=self.patch_size,
                stride=self.stride,
            )
            patches = extract_tensor_patches(
                image,
                self.patch_size,
                self.stride,
                padding=pad,
            )
        else:
            patches = extract_tensor_patches(
                image,
                self.patch_size,
                self.stride,
            )

        return (
            patches,
            image_path,
        )


@dataclass
class ImageSampleInfo:
    class_name: str
    image_path: str

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(astuple(self))


def get_samples(image_folder: str) -> list[ImageSampleInfo]:
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
                    samples.append(ImageSampleInfo(class_name=cls, image_path=path))
    return samples


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


class TrainDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        image_folder: str,
        image_size: tuple[int, int] = (400, 400),
        transform: ttransforms.transforms = None,
        indices: Optional[list[int]] = None,
    ) -> None:
        self.image_size = image_size
        self.transform = transform if transform is not None else ttransforms.ToTensor()

        self.samples = get_samples(image_folder)
        if indices is not None:
            assert all(i < len(self.samples) for i in indices), (
                "All indices must be less than the number of samples."
            )
            self.samples = [self.samples[i] for i in indices if i < len(self.samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        cls_name, image_path = self.samples[idx]
        image = Image.open(image_path)
        image = image.resize(self.image_size)
        image = self.transform(image)
        return image, cls_name


class UnlabeledDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        image_folder: str,
        image_size: tuple[int, int] = (400, 400),
        transform: ttransforms.transforms = None,
        indices: Optional[list[int]] = None,
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.samples[idx]
        image = Image.open(image_path)
        image = image.resize(self.image_size)
        image = self.transform(image)
        return image


def get_data_split(
    image_folder: str, val_size: float = 0.2, test_size: float = 0.1
) -> tuple[list[int], list[int], list[int]]:
    """Generates split indices for training, validation, and test sets.

    Args:
        image_folder (str): The folder containing the image folders.

    Returns:
        dict[str, list[int]]: A dict with 'train', 'val', and 'test' indices.
    """
    samples = get_samples(image_folder)
    indices = list(range(len(samples)))
    assert val_size + test_size < 1.0, (
        "Validation and test sizes must sum to less than 1.0"
    )

    # A lot of classes have only one image, so we need to treat those differently
    # We also treat those with 2 images differently, since we want three splits
    class_counts: dict[str, int] = {}
    for class_name, _ in samples:
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    few_image_indices = [i for i in indices if class_counts[samples[i].class_name] < 3]
    multi_image_indices = [
        i for i in indices if class_counts[samples[i].class_name] >= 3
    ]

    # FIXME: There is a small chance, that for a class with 3 images, 2 are put into val and 1 remains in train_test. In this case an exception will be thrown. I don't want to implement stratified though...
    train_test_indices, val_indices = train_test_split(
        multi_image_indices,
        test_size=val_size,
        random_state=42,
        stratify=[samples[i].class_name for i in multi_image_indices],
    )
    train_indices, test_indices = train_test_split(
        train_test_indices,
        test_size=test_size / (1.0 - val_size),
        random_state=42,
        stratify=[samples[i].class_name for i in train_test_indices],
    )

    for i in few_image_indices:
        r = random.random()
        if r < val_size:
            val_indices.append(i)
        elif r < val_size + test_size:
            test_indices.append(i)
        else:
            train_indices.append(i)

    return train_indices, val_indices, test_indices


def load(
    config: DictConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, int],
]:
    metadata_path = os.path.join(
        config.project_path, config.data.folder, config.data.metadata.folder
    )

    df_metadata = pd.read_csv(
        os.path.join(metadata_path, config.data.metadata.training),
        sep=";",
        dtype={"partner": str},
    )

    df_species_ids = pd.read_csv(
        os.path.join(metadata_path, config.data.metadata.labels)
    )

    class_map = df_species_ids[
        "species_id"
    ].to_dict()  # dictionary to map the species model Id with the species Id

    return (
        df_metadata,
        df_species_ids,
        class_map,
    )
