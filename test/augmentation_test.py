from PIL import Image, ImageChops

import src.augmentation as augmentation

def create_test_image(width=100, height=100):
    img = Image.new('RGB', (width, height))
    for x in range(width):
        for y in range(height):
            img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
    return img

def images_equal(img1, img2):
    return ImageChops.difference(img1, img2).getbbox() is None

def test_image():
    img = create_test_image()
    img.save("outputs/test/test_image.png")

def test_flip_horizontal():
    img = create_test_image()
    flipped = augmentation.flip(img, horizontal=True, vertical=False)
    flipped.save("outputs/test/flipped_horizontal.png")
    re_flipped = augmentation.flip(flipped, horizontal=True, vertical=False)
    assert images_equal(img, re_flipped)

def test_flip_vertical():
    img = create_test_image()
    flipped = augmentation.flip(img, horizontal=False, vertical=True)
    flipped.save("outputs/test/flipped_vertical.png")
    re_flipped = augmentation.flip(flipped, horizontal=False, vertical=True)
    assert images_equal(img, re_flipped)

def test_rotate():
    img = create_test_image()
    rotated = augmentation.rotate(img, angle=45)
    rotated.save("outputs/test/rotated.png")
    assert rotated.size == img.size
    assert not images_equal(img, rotated)

def test_scale():
    img = create_test_image()
    scaled = augmentation.scale(img, scale_factor=1.5)
    scaled.save("outputs/test/scaled.png")
    assert scaled.size == (int(img.size[0] * 1.5), int(img.size[1] * 1.5))
    assert not images_equal(img, scaled)

def test_crop():
    img = create_test_image()
    cropped = augmentation.crop(img, crop_size=img.size, offset=(0, 0))
    assert images_equal(img, cropped)

    cropped = augmentation.crop(img, crop_size=(50, 50), offset=(50, 50))
    cropped.save("outputs/test/cropped.png")
    assert cropped.size == (50, 50)
    assert not images_equal(img, cropped)

def test_add_gaussian_noise():
    img = create_test_image()
    noisy = augmentation.add_gaussian_noise(img, mean=0, stddev=5, seed=42)
    noisy.save("outputs/test/noisy.png")
    assert not images_equal(img, noisy)

def test_adjust_brightness():
    img = create_test_image()
    brightened = augmentation.adjust_brightness(img, factor=1.5)
    brightened.save("outputs/test/brightened.png")
    assert not images_equal(img, brightened)

def test_zoom():
    img = create_test_image()
    zoomed = augmentation.zoom(img, factor=2)
    zoomed.save("outputs/test/zoomed.png")
    assert zoomed.size == img.size
    assert not images_equal(img, zoomed)

def test_adjust_contrast():
    img = create_test_image()
    contrasted = augmentation.adjust_contrast(img, factor=1.5)
    contrasted.save("outputs/test/contrasted.png")
    assert contrasted.size == img.size
    assert not images_equal(img, contrasted)

def test_adjust_saturation():
    img = create_test_image()
    saturated = augmentation.adjust_saturation(img, factor=1.5)
    saturated.save("outputs/test/saturated.png")
    assert saturated.size == img.size
    assert not images_equal(img, saturated)

def test_blur():
    img = create_test_image()
    blurred = augmentation.blur(img, radius=10)
    blurred.save("outputs/test/blurred.png")
    assert blurred.size == img.size
    assert not images_equal(img, blurred)

def test_mix():
    img1 = create_test_image()
    img2 = create_test_image()
    mixed = augmentation.mix(img1, img2, alpha=0.5)
    assert mixed.size == img1.size
    assert images_equal(mixed, img1)
    assert images_equal(mixed, img2)

    img2 = augmentation.flip(img1, horizontal=True, vertical=False)
    mixed = augmentation.mix(img1, img2, alpha=0.5)
    mixed.save("outputs/test/mixed.png")
    assert mixed.size == img1.size
    assert not images_equal(mixed, img1)
    assert not images_equal(mixed, img2)