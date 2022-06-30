from typing import List

import numpy
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

__all__ = ["apply_op"]


def gaussian_blur(
    image: Image.Image, sigma: float = 0.0, truncate: float = 3.5
) -> Image.Image:
    """Apply gaussian blur to an image.

    Args:
        image: Input image.
        sigma: Sigma, used to calculate kernel_size. Defaults to 0.0.
        truncate: Truncate kernel. Defaults to 3.5.

    Returns:
        Blurred image.
    """
    # Check image.
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image not {}".format(type(image)))
    # Define kernel size.
    kernel_size = round(float(sigma) * truncate)
    kernel_size = max(3, kernel_size // 2 * 2 + 1)
    if sigma > 0:
        image = F.gaussian_blur(
            image, kernel_size=(kernel_size, kernel_size), sigma=sigma
        )
    return image


def adjust_channel(image: Image.Image, factor: float, channel: int) -> Image.Image:
    """Adjust image channel values by a factor.

    Args:
        image: Input image.
        factor: Multiplier factor.
        channel: Channel number.

    Returns:
        Adjusted image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image not {}".format(type(image)))
    # Convert to array.
    arr = numpy.array(image).astype(numpy.float32) / 255
    # Multiply
    if arr.ndim > 2:
        arr[..., channel] *= factor
    else:
        arr *= factor
    # Clip values and convert back to uint8.
    image = (numpy.clip(arr, 0.0, 1.0) * 255).astype(numpy.uint8)
    return Image.fromarray(image)


def apply_op(
    image: Image.Image,
    name: str,
    magnitude: float,
    interpolation: T.InterpolationMode = T.InterpolationMode.NEAREST,
    fill: List[int] = [128, 128, 128],
) -> Image.Image:
    """Apply operation to an image with magnitude.

    Args:
        image: Input image.
        name: Operation name.
        magnitude: Magnitude for the operation
        interpolation: Interpolation mode. Defaults to NEAREST.
        fill: Fill value. Defaults to [128, 128, 128].

    Returns:
        Transformed image.
    """
    # Check image.
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image not {}".format(type(image)))
    if name.lower() == "shearx":
        image = F.affine(
            image,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[magnitude, 0.0],
            interpolation=interpolation,
            fill=fill,
        )
    elif name.lower() == "sheary":
        image = F.affine(
            image,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, magnitude],
            interpolation=interpolation,
            fill=fill,
        )
    elif name.lower() == "translatex":
        image = F.affine(
            image,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif name.lower() == "translatey":
        image = F.affine(
            image,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif name.lower() == "rotate":
        image = F.rotate(image, int(magnitude), interpolation=interpolation, fill=fill)
    elif name.lower() == "brightness":
        image = F.adjust_brightness(image, magnitude)
    elif name.lower() == "saturation":
        image = F.adjust_saturation(image, magnitude)
    elif name.lower() == "grayscale":
        image = F.adjust_saturation(image, 0.0)
    elif name.lower() == "hue":
        # HSV conversion causes changes.
        if magnitude != 0:
            image = F.adjust_hue(image, magnitude)
    elif name.lower() == "gamma":
        image = F.adjust_gamma(image, magnitude)
    elif name.lower() == "contrast":
        image = F.adjust_contrast(image, magnitude)
    elif name.lower() == "red":
        image = adjust_channel(image, magnitude, channel=0)
    elif name.lower() == "green":
        image = adjust_channel(image, magnitude, channel=1)
    elif name.lower() == "blue":
        image = adjust_channel(image, magnitude, channel=2)
    elif name.lower() == "sharpness":
        image = F.adjust_sharpness(image, magnitude)
    elif name.lower() == "blur" or name.lower() == "gaussian_blur":
        image = gaussian_blur(image, magnitude)
    elif name.lower() == "posterize":
        image = F.posterize(image, int(magnitude))
    elif name.lower() == "solarize":
        if int(magnitude) <= 255:
            image = F.solarize(image, int(magnitude))
    elif name.lower() == "autocontrast":
        image = F.autocontrast(image)
    elif name.lower() == "equalize":
        image = F.equalize(image)
    elif name.lower() == "invert":
        image = F.invert(image)
    elif name.lower() == "identity":
        pass
    else:
        raise ValueError(f"Operation '{name}' not recognized.")
    return image
