import os
import shutil
from functools import partial

import torchvision.transforms.functional as F
from strong_augment import shift_dataset

IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "strong_augment", "image.jpeg"
)
TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")


def test_distribution_shift():
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    # Shift function.
    output = shift_dataset(
        paths=[IMAGE_PATH] * 100,
        output_dir=TMP_DIR,
        function=partial(F.adjust_gamma, gamma=0.2),
        num_workers=1,
    )
    assert output == []
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
