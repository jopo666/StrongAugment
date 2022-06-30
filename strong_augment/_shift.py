import os
from typing import Callable, List, Tuple

from PIL import Image

from ._multiprocess import multiprocess_loop
from ._progress import progress_bar

__all__ = ["shift_dataset"]


def process_image(path: str, output_dir: str, function: Callable):
    """Helper function to load, shift and save an image."""
    try:
        image = Image.open(path)
        shifted = function(image)
        shifted.save(os.path.join(output_dir, os.path.basename(path)))
    except Exception as e:
        return path, e
    return path, None


def shift_dataset(
    paths: List[str], output_dir: str, function: Callable, num_workers: int = 1
) -> List[Tuple[str, Exception]]:
    """Shift the data distribution of dataset to a known direction with a function.

    Args:
        paths: Paths to images in the dataset.
        output_dir: Output directory for the new dataset images.
        function: Function used to shift the data distribution.
        num_processes: Number worker processes.

    Returns:
        List of paths, which could not be processed with the associated Expection.
    """
    # Check output_dir.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not os.path.isdir(output_dir):
        raise NotADirectoryError("{} exists but isn't a directory.")
    # Shift images.
    exceptions = []
    for logs, (path, exception) in progress_bar(
        multiprocess_loop(
            func=process_image,
            iterable=paths,
            output_dir=output_dir,
            function=function,
            num_workers=num_workers,
        ),
        total=len(paths),
        desc="Processing images",
        log_values=True,
    ):
        if exception is not None:
            exceptions.append((path, exception))
            logs["failures"] = len(exceptions)
    return exceptions
