import os
import shutil

from strong_augment import shift_dataset

from .utils import IMAGE_PATH

TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")


def augment(x):
    return x.convert("L")


def test_distribution_shift():
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    # Shift function.
    shift_dataset(
        paths=[IMAGE_PATH] * 100,
        output_dir=TMP_DIR,
        function=augment,
        max_workers=1,
    )
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
