import os
import time

import torch
import torchvision.transforms as ttransforms
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
    device: torch.device,
    top_k_tile: int,
    class_map: dict[int, int],
    min_score: float,
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
                batch_patches = batch_patches.to(device)

                with autocast("cuda"):
                    outputs = model(batch_patches)  # Perform inference on the batch
                    probabilities = torch.nn.functional.softmax(
                        outputs,
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
                    ) in zip(
                        top_indices,
                        top_probs,
                        strict=False,
                    ):
                        for (
                            top_idx,
                            top_prob,
                        ) in zip(
                            top_tile_indices,
                            top_tile_probs,
                            strict=False,
                        ):
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
                print(
                    f"Predict: [{batch_idx}/{len(dataloader)}] "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})"
                )

    return image_predictions
