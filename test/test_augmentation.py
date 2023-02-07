import functools

import numpy as np
import pytest
from PIL import Image

from strong_augment import StrongAugment
from strong_augment._augment import _apply_operation

from .utils import EXAMPLE_IMAGE

IMAGE = np.array(EXAMPLE_IMAGE)
apply_op = functools.partial(_apply_operation, image=IMAGE)


def test_strong_augment_rgb() -> None:
    augment = StrongAugment()
    img_pil = EXAMPLE_IMAGE
    img_arr = np.array(EXAMPLE_IMAGE)
    # Pillow image.
    for __ in range(256):
        assert isinstance(augment(img_pil), Image.Image)
    # Array image.
    for __ in range(256):
        img_aug = augment(img_arr)
        assert isinstance(img_aug, np.ndarray)
        assert img_aug.ndim == 3


def test_strong_augment_grayscale() -> None:
    augment = StrongAugment()
    img_pil = EXAMPLE_IMAGE.convert("L")
    img_arr = np.array(EXAMPLE_IMAGE.convert("L"))
    # Pillow image.
    for __ in range(256):
        assert isinstance(augment(img_pil), Image.Image)
    # Array image.
    assert img_arr.ndim == 2
    for __ in range(256):
        img_aug = augment(img_arr)
        assert isinstance(img_aug, np.ndarray)
        assert img_aug.ndim == 2


def test_zero_magnitude():
    assert (apply_op(operation_name="red", magnitude=0.0) == 0)[..., 0].all()
    assert (apply_op(operation_name="green", magnitude=0.0) == 0)[..., 1].all()
    assert (apply_op(operation_name="blue", magnitude=0.0) == 0)[..., 2].all()
    assert (apply_op(operation_name="brightness", magnitude=0.0) == 0).all()
    assert (apply_op(operation_name="gamma", magnitude=0.0) == 255).all()


def test_no_op():  # no op.
    assert (apply_op(operation_name="red", magnitude=1.0) == IMAGE).all()
    assert (apply_op(operation_name="green", magnitude=1.0) == IMAGE).all()
    assert (apply_op(operation_name="blue", magnitude=1.0) == IMAGE).all()
    assert (apply_op(operation_name="hue", magnitude=0.0) == IMAGE).all()
    assert (apply_op(operation_name="brightness", magnitude=1.0) == IMAGE).all()
    assert (apply_op(operation_name="saturation", magnitude=1.0) == IMAGE).all()
    assert (apply_op(operation_name="gamma", magnitude=1.0) == IMAGE).all()
    assert (apply_op(operation_name="sharpen", magnitude=0.0) == IMAGE).all()
    assert (apply_op(operation_name="blur", magnitude=0.0) == IMAGE).all()
    assert (apply_op(operation_name="posterize", magnitude=8) == IMAGE).all()
    assert (apply_op(operation_name="solarize", magnitude=256) == IMAGE).all()


def test_augment_space():
    # full should pass.
    StrongAugment()
    # int should be fine
    StrongAugment(augment_space={"red": (1, 2)})
    with pytest.raises(TypeError):
        StrongAugment(augment_space={"red": (True, False)})
    with pytest.raises(TypeError):
        StrongAugment(augment_space={"posterize": (True, False)})
    with pytest.raises(TypeError):
        StrongAugment(augment_space={"autocontrast": (1, 2)})
    with pytest.raises(TypeError):
        StrongAugment(augment_space={"autocontrast": (True, 2)})
    with pytest.raises(ValueError):
        StrongAugment(augment_space={"posterize": (0, 2)})
    with pytest.raises(ValueError):
        StrongAugment(augment_space={"posterizzzzzzzzze": (1, 2)})
