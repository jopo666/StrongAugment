import os

from PIL import Image

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "image.jpeg")
EXAMPLE_IMAGE = Image.open(IMAGE_PATH)
