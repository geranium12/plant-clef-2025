import os

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
from torch.utils.data import (
    Dataset,
)


class PatchDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        patches: torch.Tensor,
        transform: ttransforms.Normalize = None,
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


def load(
    config: DictConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, int],
]:
    test_data_path = os.path.join(config.project_path, config.data.folder, config.data.test_folder)

    df_metadata = pd.read_csv(
        os.path.join(test_data_path, config.data.metadata_file),
        sep=";",
        dtype={"partner": str},
    )

    df_species_ids = pd.read_csv(os.path.join(test_data_path, config.data.species_file))

    class_map = df_species_ids[
        "species_id"
    ].to_dict()  # dictionary to map the species model Id with the species Id

    return (
        df_metadata,
        df_species_ids,
        class_map,
    )
