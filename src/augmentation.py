import torchvision.transforms as ttransforms
from omegaconf import DictConfig


def get_random_data_augmentation() -> ttransforms.transforms:
    return ttransforms.Compose(
        [
            ttransforms.RandomResizedCrop(size=(518, 518)),
            ttransforms.RandomHorizontalFlip(),
            ttransforms.RandomVerticalFlip(),
            ttransforms.RandomPerspective(distortion_scale=0.2),
            ttransforms.RandomRotation(20),
            ttransforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            ttransforms.ToTensor(),
            ttransforms.Normalize(
                mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
            ),
        ]
    )


def get_data_augmentation(config: DictConfig, name: str) -> ttransforms.transforms:
    match name:
        case "Identity":
            return ttransforms.ToTensor()
        case "FlipHorizontal":
            return ttransforms.Compose(
                [
                    ttransforms.ToTensor(),
                    ttransforms.RandomHorizontalFlip(),
                ]
            )
        case "FlipVertical":
            return ttransforms.Compose(
                [
                    ttransforms.ToTensor(),
                    ttransforms.RandomVerticalFlip(),
                ]
            )
        case "Rotate":
            return ttransforms.Compose(
                [
                    ttransforms.ToTensor(),
                    ttransforms.RandomRotation(30),
                ]
            )
        case "Jitter":
            return ttransforms.Compose(
                [
                    ttransforms.ToTensor(),
                    ttransforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                ]
            )
        case "ResizeCrop":
            return ttransforms.Compose(
                [
                    ttransforms.ToTensor(),
                    ttransforms.RandomResizedCrop(
                        size=(config.image_width, config.image_height)
                    ),
                ]
            )
        case "Perspective":
            return ttransforms.Compose(
                [
                    ttransforms.ToTensor(),
                    ttransforms.RandomPerspective(distortion_scale=0.5),
                ]
            )
        case "Sharpness":
            return ttransforms.Compose(
                [
                    ttransforms.ToTensor(),
                    ttransforms.RandomAdjustSharpness(sharpness_factor=2),
                ]
            )
        case "GaussianBlur":
            return ttransforms.Compose(
                [
                    ttransforms.ToTensor(),
                    ttransforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                ]
            )
        case _:
            raise TypeError(f"Unknown augmentation name: {name}")
