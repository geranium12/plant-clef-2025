import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def flip(
    image: Image.Image, horizontal: bool = True, vertical: bool = False
) -> Image.Image:
    if horizontal:
        image = ImageOps.mirror(image)
    if vertical:
        image = ImageOps.flip(image)
    return image


def rotate(image: Image.Image, angle: int) -> Image.Image:
    return image.rotate(angle)


def scale(image: Image.Image, scale_factor: float) -> Image.Image:
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return image.resize((new_width, new_height))


def crop(
    image: Image.Image, crop_size: tuple[int, int], offset: tuple[int, int] = (0, 0)
) -> Image.Image:
    width, height = image.size
    crop_width, crop_height = crop_size
    left, upper = offset
    right = left + crop_width
    lower = upper + crop_height
    assert right <= width and lower <= height, "Crop size exceeds image dimensions."
    return image.crop((left, upper, right, lower))


def add_gaussian_noise(
    image: Image.Image, mean: float = 0, stddev: float = 10, seed: int = 0
) -> Image.Image:
    np_image = np.array(image)
    noise = np.random.default_rng(seed).normal(mean, stddev, size=np_image.shape)
    np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_image)


def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def zoom(image: Image.Image, factor: float) -> Image.Image:
    assert factor >= 1, "Zoom factor must be greater than or equal to 1."
    width, height = image.size
    if factor == 1:
        return image.copy()
    new_width = int(width / factor)
    new_height = int(height / factor)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    cropped = image.crop((left, top, left + new_width, top + new_height))
    return cropped.resize((width, height))


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def adjust_saturation(image: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def blur(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def mix(image1: Image.Image, image2: Image.Image, alpha: float) -> Image.Image:
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1."
    return Image.blend(image1, image2, alpha)
