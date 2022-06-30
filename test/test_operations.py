import numpy
from strong_augment import EXAMPLE_IMAGE, apply_op
from strong_augment._operations import adjust_channel, gaussian_blur

IMAGE_ARRAY = numpy.array(EXAMPLE_IMAGE)


def test_adjust_channels():
    # magnitude 0.
    assert (
        numpy.array(adjust_channel(EXAMPLE_IMAGE.convert("L"), 0, channel=4235)) == 0
    ).all()
    assert (numpy.array(adjust_channel(EXAMPLE_IMAGE, 0, channel=0)) == 0)[..., 0].all()
    assert (numpy.array(adjust_channel(EXAMPLE_IMAGE, 0, channel=1)) == 0)[..., 1].all()
    assert (numpy.array(adjust_channel(EXAMPLE_IMAGE, 0, channel=2)) == 0)[..., 2].all()
    # no operation.
    assert (
        numpy.array(adjust_channel(EXAMPLE_IMAGE, 1.0, channel=0)) == IMAGE_ARRAY
    ).all()
    assert (
        numpy.array(adjust_channel(EXAMPLE_IMAGE, 1.0, channel=1)) == IMAGE_ARRAY
    ).all()
    assert (
        numpy.array(adjust_channel(EXAMPLE_IMAGE, 1.0, channel=2)) == IMAGE_ARRAY
    ).all()
    # halve.
    assert (
        numpy.array(adjust_channel(EXAMPLE_IMAGE, 0.5, channel=2))[..., 2]
        == (IMAGE_ARRAY[..., 2] * 0.5).astype(numpy.uint8)
    ).all()


def test_gaussian_blur_no_operation():
    assert (
        numpy.array(gaussian_blur(EXAMPLE_IMAGE)) == numpy.array(EXAMPLE_IMAGE)
    ).all()


def test_brightness_contrast_gamma_zero():
    assert numpy.array(apply_op(EXAMPLE_IMAGE, "brightness", 0)).mean() == 0
    assert numpy.array(apply_op(EXAMPLE_IMAGE, "contrast", 0)).mean() == 128
    assert numpy.array(apply_op(EXAMPLE_IMAGE, "gamma", 0)).mean() == 255


def test_no_op():
    assert (numpy.array(apply_op(EXAMPLE_IMAGE, "red", 1.0)) == IMAGE_ARRAY).all()
    assert (numpy.array(apply_op(EXAMPLE_IMAGE, "green", 1.0)) == IMAGE_ARRAY).all()
    assert (numpy.array(apply_op(EXAMPLE_IMAGE, "blue", 1.0)) == IMAGE_ARRAY).all()
    assert (numpy.array(apply_op(EXAMPLE_IMAGE, "hue", 0.0)) == IMAGE_ARRAY).all()
    assert (numpy.array(apply_op(EXAMPLE_IMAGE, "gamma", 1.0)) == IMAGE_ARRAY).all()
    assert (numpy.array(apply_op(EXAMPLE_IMAGE, "contrast", 1.0)) == IMAGE_ARRAY).all()
    assert (numpy.array(apply_op(EXAMPLE_IMAGE, "sharpness", 1)) == IMAGE_ARRAY).all()
    assert (numpy.array(apply_op(EXAMPLE_IMAGE, "blur", 0)) == IMAGE_ARRAY).all()
    assert (numpy.array(apply_op(EXAMPLE_IMAGE, "posterize", 8)) == IMAGE_ARRAY).all()
    assert (numpy.array(apply_op(EXAMPLE_IMAGE, "solarize", 256)) == IMAGE_ARRAY).all()
    assert (
        numpy.array(apply_op(EXAMPLE_IMAGE, "saturation", 1.0)) == IMAGE_ARRAY
    ).all()
    assert (
        numpy.array(apply_op(EXAMPLE_IMAGE, "brightness", 1.0)) == IMAGE_ARRAY
    ).all()
