import functools
from collections.abc import Callable
from pathlib import Path
from typing import Union

from PIL import Image
from tqdm.contrib.concurrent import process_map

__all__ = ["shift_dataset"]


def process_image(
    path: Path, output_dir: Path, function: Callable
) -> tuple[str, Exception]:
    """Helper function to load, shift and save an image."""
    function(Image.open(path)).save(output_dir / path.name)


def shift_dataset(
    paths: list[Union[str, Path]],
    output_dir: Union[str, Path],
    function: Callable[[Image.Image], Image.Image],
    max_workers: int = 1,
) -> list[tuple[str, Exception]]:
    """Shift the data distribution of dataset to a known direction with a function.

    Args:
        paths: Paths to images in the dataset.
        output_dir: Output directory for the new dataset images.
        function: Function used to shift the data distribution (shoud input and output a
            PIL image).
        max_workers: Maximum number of worker processes. Defaults to 1.

    Returns:
        List of paths, which could not be processed.
    """
    # Check output dir.
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Define iterable of Paths.
    iterable = (Path(x) if not isinstance(x, Path) else x for x in paths)
    # Map with num workers.
    process_map(
        functools.partial(process_image, output_dir=output_dir, function=function),
        iterable,
        desc="Shifting dataset",
        total=len(paths),
        max_workers=max_workers,
    )
