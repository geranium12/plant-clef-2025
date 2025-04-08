import torchvision.transforms as ttransforms
from omegaconf import DictConfig


def create_augmentations(config: DictConfig) -> dict[str, ttransforms.transforms]:
    return {
        "Identity": ttransforms.ToTensor(),
        "FlipHorizontal": ttransforms.Compose(
            [
                ttransforms.ToTensor(),
                ttransforms.RandomHorizontalFlip(),
            ]
        ),
        "FlipVertical": ttransforms.Compose(
            [
                ttransforms.ToTensor(),
                ttransforms.RandomVerticalFlip(),
            ]
        ),
        "Rotate": ttransforms.Compose(
            [
                ttransforms.ToTensor(),
                ttransforms.RandomRotation(30),
            ]
        ),
        "Jitter": ttransforms.Compose(
            [
                ttransforms.ToTensor(),
                ttransforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        ),
        "ResizeCrop": ttransforms.Compose(
            [
                ttransforms.ToTensor(),
                ttransforms.RandomResizedCrop(
                    size=(config.image_width, config.image_height)
                ),
            ]
        ),
        "Perspective": ttransforms.Compose(
            [
                ttransforms.ToTensor(),
                ttransforms.RandomPerspective(distortion_scale=0.5),
            ]
        ),
        "Sharpness": ttransforms.Compose(
            [
                ttransforms.ToTensor(),
                ttransforms.RandomAdjustSharpness(sharpness_factor=2),
            ]
        ),
        "GaussianBlur": ttransforms.Compose(
            [
                ttransforms.ToTensor(),
                ttransforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            ]
        ),
    }
