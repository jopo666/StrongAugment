from os.path import dirname, join

from PIL import Image

__all__ = ["EXAMPLE_IMAGE"]

EXAMPLE_IMAGE = Image.open(join(dirname(__file__), "image.jpeg"))
