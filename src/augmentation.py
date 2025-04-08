import torchvision.transforms as ttransforms
from omegaconf import DictConfig


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
