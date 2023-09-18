__all__ = ["StrongAugment", "get_augment_space", "ALLOWED_OPERATIONS"]

import functools
from typing import Optional, Union

import cv2
import numpy as np
from numpy.random import RandomState
from PIL import Image, ImageOps

# Types.
ImageType = Union[Image.Image, np.ndarray]
MagnitudeType = Union[float, int, bool]

# Error messages.
ERROR_LENGTHS_DIFFER = "Operation length does not match probabilities length."
ERROR_OPERATION_NOT_SUPPORTED = "Operation '{}' not supported. Please, select from: {}."
ERROR_AUGMENT_SPACE_WRONG_TYPE = "Augment space should be a dictionary, not {}."
ERROR_AUGMENT_SPACE_WRONG_BOUNDS_TYPE = (
    "Augment space bounds should be a tuple of two elements (low, high)."
)
ERRPR_AUGMENT_SPACE_BOUND_TYPES_DIFFER = "Bound types should be the same ({} != {})"
ERROR_AUGMENT_SPACE_WRONG_BOUND_TYPE = (
    "Augment space bound should be int/float/bool, not {}."
)
ERROR_WRONG_OPERATION_BOUND_TYPE = "Bounds for operation '{}' should be {}, not {}."
ERROR_WRONG_OPERATION_BOUND_RANGE = "Bounds for operation '{}' should be between [{}]."
ERROR_NEGATIVE_OPERATION_BOUND = "Negative values are not allowed for operation '{}'"

# Constants.
BOUND_LENGTH = 2
GRAYSCALE_NDIM = 2
OPERATIONS_WITH_INT_BOUND = {"solarize", "posterize", "jpeg"}
OPERATIONS_WITH_BOOL_BOUND = {"autocontrast", "equalize", "grayscale"}
OPERATIONS_WITH_NON_NEGATIVE_BOUND = {
    "red",
    "green",
    "blue",
    "saturation",
    "brightness",
    "contrast",
    "gamma",
    "solarize",
    "sharpen",
    "emboss",
    "blur",
    "noise",
    "jpeg",
    "tone",
}
HUE_MIN = -0.5
HUE_MAX = 0.5
SOLARIZE_MAX = 256
POSTERIZE_MIN = 1
POSTERIZE_MAX = 8
JPEG_MAX = 100
TONE_MAX = 1.0


def get_augment_space() -> dict[str, tuple[float, float]]:
    """Create a default augmentation space."""
    return {
        "red": (0.0, 2.0),
        "green": (0.0, 2.0),
        "blue": (0.0, 2.0),
        "hue": (-0.5, 0.5),
        "saturation": (0.0, 2.0),
        "brightness": (0.1, 2.0),
        "contrast": (0.1, 2.0),
        "gamma": (0.1, 2.0),
        "solarize": (0, 255),
        "posterize": (1, 8),
        "sharpen": (0.0, 1.0),
        "emboss": (0.0, 1.0),
        "blur": (0.0, 3.0),
        "noise": (0.0, 0.2),
        "jpeg": (0, 100),
        "tone": (0.0, 1.0),
        "autocontrast": (True, True),
        "equalize": (True, True),
        "grayscale": (True, True),
    }


DEFAULT_AUGMENT_SPACE = get_augment_space()


class StrongAugment:
    def __init__(
        self,
        operations: tuple[int, ...] = (2, 3, 4),
        probabilities: tuple[float, ...] = (0.5, 0.3, 0.2),
        augment_space: dict[str, tuple] = DEFAULT_AUGMENT_SPACE,
        seed: Optional[int] = None,
    ) -> None:
        """Augment like there's no tomorrow!

        Args:
            operations: Number of operations. Defaults to (2, 3, 4).
            probabilities: Probabilities for each operation. Defaults to (0.5, 0.3, 0.2).
            augment_space: Augmentation space where operations and magnitudes
                are sampled from. Should contain a tuple with (low, high)
                values for operation defined by the key. Defaults to
                `get_augment_space()`.
            seed: For `numpy.random.RandomState`.
        """  # noqa: D400
        super().__init__()
        _check_augment_space(augment_space)
        if len(operations) != len(probabilities):
            raise ValueError(ERROR_LENGTHS_DIFFER)
        self.rng = np.random.RandomState(seed=seed)
        self.augment_space = augment_space
        self.operations = operations
        self.probabilities = probabilities
        self.last_operations = {}

    def __call__(self, image: ImageType) -> ImageType:
        """Augment image."""
        # Copy image.
        image = image.copy()
        # Convert to a numpy array.
        to_pil = False
        if isinstance(image, Image.Image):
            to_pil = True
            image = np.array(image)
        # Convert to an RGB image.
        to_gray = False
        if image.ndim == GRAYSCALE_NDIM:
            to_gray = True
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Select operations.
        num_ops = np.random.choice(self.operations, p=self.probabilities)
        operations = np.random.choice(
            list(self.augment_space), size=num_ops, replace=False
        )
        # Transform image.
        for name in operations:
            # Define kwargs.
            kwargs = dict(
                operation_name=name,
                **_magnitude_kwargs(
                    name, bounds=self.augment_space[name], rng=self.rng
                ),
            )
            # Apply augmentation.
            image = _apply_operation(image, **kwargs)
            # Save last operations.
            self.last_operations = kwargs
        # Convert back to grayscale.
        if to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Convert back to Pillow image.
        if to_pil:
            image = Image.fromarray(image)
        return image

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"operations={self.operations}, "
            f"probabilities={self.probabilities}, "
            f"augment_space={self.augment_space})"
        )


def adjust_channel(image: np.ndarray, magnitude: float, channel: int) -> np.ndarray:
    image[..., channel] = cv2.addWeighted(
        image[..., channel],
        magnitude,
        np.zeros_like(image[..., channel]),
        1 - magnitude,
        gamma=0,
    )
    return image


def adjust_hue(image: np.ndarray, magnitude: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * magnitude, 180).astype(np.uint8)
    hsv[..., 0] = cv2.LUT(hsv[..., 0], lut)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def adjust_saturation(image: np.ndarray, magnitude: float) -> np.ndarray:
    gray = grayscale(image)
    if magnitude == 0:
        return gray
    return cv2.addWeighted(image, magnitude, gray, 1 - magnitude, gamma=0)


def adjust_brightness(image: np.ndarray, magnitude: float) -> np.ndarray:
    return cv2.addWeighted(
        image, magnitude, np.zeros_like(image), 1 - magnitude, gamma=0
    )


def adjust_contrast(image: np.ndarray, magnitude: float) -> np.ndarray:
    mean = np.full_like(
        image,
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean(),
        dtype=image.dtype,
    )
    return cv2.addWeighted(image, magnitude, mean, 1 - magnitude, gamma=0)


def adjust_gamma(image: np.ndarray, magnitude: float) -> np.ndarray:
    table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** magnitude) * 255
    return cv2.LUT(image, table.astype(np.uint8))


def solarize(image: np.ndarray, magnitude: int) -> np.ndarray:
    lut = [(i if i < int(round(magnitude)) else 255 - i) for i in range(256)]
    return cv2.LUT(image, np.array(lut, dtype=np.uint8))


def posterize(image: np.ndarray, magnitude: int) -> np.ndarray:
    return (image & -int(2 ** (8 - int(round(magnitude))))).astype(np.uint8)


def autocontrast(image: np.ndarray, **__) -> np.ndarray:
    # histogram function is ffffast as fuck in PIL.
    return np.array(ImageOps.autocontrast(Image.fromarray(image)))


def equalize(image: np.ndarray, **__) -> np.ndarray:
    output = np.empty_like(image)
    for c in range(image.shape[-1]):
        output[..., c] = cv2.equalizeHist(image[..., c])
    return output


def grayscale(image: np.ndarray, **__) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)


def gaussian_blur(image: np.ndarray, magnitude: float) -> np.ndarray:
    if magnitude <= 0:
        return image
    # Define kernel size.
    kernel_size = round(float(magnitude) * 3.5)
    kernel_size = max(3, kernel_size // 2 * 2 + 1)
    return cv2.GaussianBlur(image, ksize=(kernel_size, kernel_size), sigmaX=magnitude)


def sharpen(image: np.ndarray, magnitude: float) -> np.ndarray:
    kernel_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    kernel_sharpen = np.array(
        [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
        dtype=np.float32,
    )
    kernel = (1 - magnitude) * kernel_nochange + magnitude * kernel_sharpen
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def emboss(image: np.ndarray, magnitude: float) -> np.ndarray:
    kernel_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    kernel = (1 - magnitude) * kernel_nochange + magnitude * kernel_emboss
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def jpeg(image: np.ndarray, magnitude: int) -> np.ndarray:
    return cv2.imdecode(
        cv2.imencode(".jpeg", image, (cv2.IMWRITE_JPEG_QUALITY, int(round(magnitude))))[
            1
        ],
        cv2.IMREAD_UNCHANGED,
    )


def tone_shift(image: np.ndarray, magnitude_0: float, magnitude_1: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 256)
    evaluate_bez = np.vectorize(
        lambda t: 3 * (1 - t) ** 2 * t * magnitude_0
        + 3 * (1 - t) * t**2 * magnitude_1
        + t**3
    )
    remapping = np.rint(evaluate_bez(t) * 255).astype(np.uint8)
    return cv2.LUT(image, lut=remapping)


def add_noise(image: np.ndarray, magnitude: float) -> np.ndarray:
    noise = np.random.randint(0, 255, size=image.shape[:2], dtype=np.uint8)
    for c in range(3):
        image[..., c] = cv2.addWeighted(
            image[..., c], 1 - magnitude, noise, magnitude, gamma=0.0
        )
    return image


NAME_TO_OPERATION = {
    "red": functools.partial(adjust_channel, channel=0),
    "green": functools.partial(adjust_channel, channel=1),
    "blue": functools.partial(adjust_channel, channel=2),
    "hue": adjust_hue,
    "saturation": adjust_saturation,
    "brightness": adjust_brightness,
    "contrast": adjust_contrast,
    "gamma": adjust_gamma,
    "solarize": solarize,
    "posterize": posterize,
    "sharpen": sharpen,
    "emboss": emboss,
    "blur": gaussian_blur,
    "noise": add_noise,
    "jpeg": jpeg,
    "tone": tone_shift,
    "autocontrast": autocontrast,
    "equalize": equalize,
    "grayscale": grayscale,
}
ALLOWED_OPERATIONS = list(NAME_TO_OPERATION.keys())


def _apply_operation(image: np.ndarray, operation_name: str, **kwargs) -> np.ndarray:
    operation_fn = NAME_TO_OPERATION.get(operation_name.lower())
    if operation_fn is None:
        raise ValueError(
            ERROR_OPERATION_NOT_SUPPORTED.format(
                operation_name.lower(), ALLOWED_OPERATIONS
            )
        )
    return operation_fn(image, **kwargs)


def _check_augment_space(space: dict[str, tuple[MagnitudeType, MagnitudeType]]) -> None:
    """Check that passed augmentation space is valid."""
    if not isinstance(space, dict):
        raise TypeError(ERROR_AUGMENT_SPACE_WRONG_TYPE.format(type(space)))
    for key, val in space.items():
        if key not in ALLOWED_OPERATIONS:
            raise ValueError(
                ERROR_OPERATION_NOT_SUPPORTED.format(key.lower(), ALLOWED_OPERATIONS)
            )
        if not isinstance(val, tuple) or len(val) != BOUND_LENGTH:
            raise TypeError(ERROR_AUGMENT_SPACE_WRONG_BOUNDS_TYPE)
        # Check bounds.
        low, high = val
        if type(low) != type(high):
            raise TypeError(
                ERRPR_AUGMENT_SPACE_BOUND_TYPES_DIFFER.format(type(low), type(high))
            )
        if not isinstance(low, (int, float, bool)):
            raise TypeError(ERROR_AUGMENT_SPACE_WRONG_BOUND_TYPE.format(type(low)))
        _check_operation_bounds(key, low, high)


def _check_operation_bounds(name: str, low: MagnitudeType, high: MagnitudeType) -> None:
    # Check operation types.
    if name in OPERATIONS_WITH_BOOL_BOUND and not isinstance(low, bool):
        raise TypeError(ERROR_WRONG_OPERATION_BOUND_TYPE.format(name, bool, type(low)))
    if name in OPERATIONS_WITH_INT_BOUND and (
        not isinstance(low, int) or isinstance(low, bool)
    ):
        raise TypeError(ERROR_WRONG_OPERATION_BOUND_TYPE.format(name, int, type(low)))
    if name not in OPERATIONS_WITH_BOOL_BOUND.union(OPERATIONS_WITH_INT_BOUND) and (
        not isinstance(low, (int, float)) or isinstance(low, bool)
    ):
        raise TypeError(ERROR_WRONG_OPERATION_BOUND_TYPE.format(name, float, type(low)))
    # Check operation bounds.
    if low < 0 and name in OPERATIONS_WITH_NON_NEGATIVE_BOUND:
        raise ValueError(ERROR_NEGATIVE_OPERATION_BOUND.format(name))
    if name == "hue" and (low < HUE_MIN or high > HUE_MAX):
        raise ValueError(ERROR_WRONG_OPERATION_BOUND_RANGE.format(name, (-0.5, 0.5)))
    if name == "solarize" and high > SOLARIZE_MAX:
        raise ValueError(ERROR_WRONG_OPERATION_BOUND_RANGE.format(name, (0, 256)))
    if name == "posterize" and (high > POSTERIZE_MAX or low < POSTERIZE_MIN):
        raise ValueError(ERROR_WRONG_OPERATION_BOUND_RANGE.format(name, (1, 8)))
    if name == "jpeg" and high > JPEG_MAX:
        raise ValueError(ERROR_WRONG_OPERATION_BOUND_RANGE.format(name, (0, 100)))
    if name == "tone" and high > TONE_MAX:
        raise ValueError(ERROR_WRONG_OPERATION_BOUND_RANGE.format(name, (0, 1.0)))


def _magnitude_kwargs(
    operation_name: str, bounds: tuple[MagnitudeType, MagnitudeType], rng: RandomState
) -> Optional[dict[str, MagnitudeType]]:
    """Generate magnitude kwargs for apply_operations."""
    if operation_name == "tone":
        return {
            "magnitude_0": _sample_magnitude(*bounds, rng),
            "magnitude_1": _sample_magnitude(*bounds, rng),
        }
    magnitude = _sample_magnitude(*bounds, rng)
    if magnitude is None:
        return {}
    return {"magnitude": magnitude}


def _sample_magnitude(
    low: MagnitudeType, high: MagnitudeType, rng: RandomState
) -> MagnitudeType:
    """Sample magnitude value."""
    if isinstance(low, float):
        return rng.uniform(low, high)
    if isinstance(low, int):
        return rng.choice(range(low, high + 1))
    # Boolean does not require arguments.
    return None
