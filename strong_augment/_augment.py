import random
from typing import Callable, Dict, List, Tuple, Union

import numpy
from PIL import Image
from torchvision.transforms import InterpolationMode

from ._operations import apply_op

AFFINE_TRANSFORMS = {
    "shearx",
    "sheary",
    "translatex",
    "translatey",
    "perspective",
    "rotate",
}
AUGMENTATION_SPACE = {
    "identity": (True, False),
    "red": (0.0, 2.0),
    "green": (0.0, 2.0),
    "blue": (0.0, 2.0),
    "hue": (-0.5, 0.5),
    "saturation": (0.0, 2.0),
    "brightness": (0.1, 2.0),
    "contrast": (0.1, 2.0),
    "gamma": (0.1, 2.0),
    "sharpness": (1.0, 4.0),
    "blur": (0.0, 2.0),
    "solarize": (0, 256),
    "posterize": (1, 8),
    "autocontrast": (True, False),
    "equalize": (True, False),
    "grayscale": (True, False),
    "shearx": (-45.0, 45.0),
    "sheary": (-45.0, 45.0),
    "translatex": (-32.0, 32.0),
    "translatey": (-32.0, 32.0),
    "rotate": (-135.0, 135.0),
}

__all__ = ["StrongAugment", "AUGMENTATION_SPACE", "augmentation_collage"]


class StrongAugment:
    def __init__(
        self,
        p: float = 0.4,
        min_ops: int = 2,
        max_ops: int = 5,
        max_affine: int = 1,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        augmentation_space: Dict[
            str, List[Union[int, float, bool]]
        ] = AUGMENTATION_SPACE,
        fill: List[int] = [128, 128, 128],
    ):
        """Strong automatic augmentation.

        Args:
            p: Probability consecutive ops after `min_ops`. Defaults  to 0.4.
            min_ops: Minimum number of operations. Defaults to 2.
            max_ops: Maximum number of operations. Defaults to 5.
            max_distortive: Maximum number of distortive operations. Defaults to 1.
            interpolation: Interpolation method. Defaults to InterpolationMode.NEAREST.
            augmentation_space: Augmentation space from where transforms and magntitudes
                are sampled. Defaults to the one used in the original paper.
            fill: Fill for distortive operations. Defaults to [128, 128, 128].
        """

        self.__p = p
        self.__min_ops = min_ops
        self.__max_ops = max_ops
        self.__max_affine = max_affine
        self.__interpolation = interpolation
        self.__fill = fill
        self.__augmentation_space = augmentation_space
        self.__last_operations = []

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL image not {}".format(type(image)))
        # Define possible operations.
        operation_pool = list(self.__augmentation_space.keys())
        # Start transforming images.
        num_distortive = 0
        num_operations = 0
        self.__last_operations = []
        while len(operation_pool) > 0:
            # Select random operation and remove from the pool.
            name = random.choice(operation_pool)
            operation_pool.remove(name)
            # Check if operation is allowed.
            if name in AFFINE_TRANSFORMS:
                if num_distortive >= self.__max_affine:
                    # Operation not allowed.
                    continue
                else:
                    num_distortive += 1
            # Get magnitude.
            magnitude = get_magntiude(*self.__augmentation_space[name])
            # Apply transformation.
            image = apply_op(
                image,
                name,
                magnitude,
                interpolation=self.__interpolation,
                fill=self.__fill,
            )
            # Save transform.
            self.__last_operations.append((name, magnitude))
            # Increase operation counter.
            num_operations += 1
            # Check if we continue.
            if num_operations >= self.__min_ops:
                if num_operations >= self.__max_ops:
                    break
                elif random.random() > self.__p:
                    break
        return image

    def repeat(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL image not {}".format(type(image)))
        for name, magnitude in self.__last_operations:
            image = apply_op(
                image,
                name,
                magnitude,
                interpolation=self.__interpolation,
                fill=self.__fill,
            )
        return image

    def __repr__(self):
        return "{}(p={}, min_ops={}, max_ops={}, max_distortive={})".format(
            self.__class__.__name__,
            self.__p,
            self.__min_ops,
            self.__max_ops,
            self.__max_affine,
        )


def get_magntiude(
    low: Union[float, int, bool], high: Union[float, int, bool]
) -> Union[float, int, bool]:
    if isinstance(low, float):
        return random.uniform(low, high)
    elif isinstance(low, bool):
        return random.choice([True, False])
    elif isinstance(low, int):
        return random.choice(range(low, high + 1))
    else:
        raise ValueError(
            "Augmentation space should contain int/float/bool, not  {}".format(
                type(low)
            )
        )


def augmentation_collage(
    image: Image.Image,
    augmentation: Callable,
    nrows: int = 4,
    ncols: int = 16,
    shape: Tuple[int, int] = (64, 64),
):
    """Generate a collage image with the passed augmentation strategy.

    Args:
        image: Input image.
        auto_augment: AutoAugement strategy
        nrows: Number of collage rows. Defaults to 4.
        ncols: Number of collage columns. Defaults to 16.
        shape: Shape of each image. Defaults to (64, 64).

    Returns:
        Collage image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image not {}".format(type(image)))
    collage = []
    row = []
    for i in range(nrows * ncols):
        transformed = augmentation(image)
        row.append(transformed.resize(shape))
        if len(row) == ncols:
            collage.append(numpy.hstack([numpy.array(x) for x in row]))
            row = []
    collage = numpy.vstack(collage)
    return Image.fromarray(collage)
