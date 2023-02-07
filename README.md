# Augment like there's no tomorrow: Consistently performing neural networks for medical imaging [[`arXiv`]](https://arxiv.org/abs/2206.15274)

This repository contains implementations for `StrongAugment` and creating
**_distribution-shifted_** datasets.

## Installation

```bash
pip3 install strong-augment
```

## Training with strong augmentation.

To train your neural networks with strong augmentatiom simply include `StrongAugment` to your image transformation pipeline!

```python
import torchvision.transforms as T
from strong_augment import StrongAugment

trnsf = T.Compose(
    T.RandomResizedCrop(224),
    T.RandomVerticalFlip(0.5),
    T.RandomHorizontalFlip(0.5),
    StrongAugment(operations=[2, 3, 4], probabilities=[0.5, 0.3, 0.2]), # Just one line!
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    T.RandomErase(0.2)
)
```
## Creating shifted datasets.

Function `shift_dataset` can be used create the distribution-shifted datasets for shifted evaluation.

```python
from functools import partial
import torchvision.transforms.functional as F
from strong_augment import shift_dataset

# Let's define the distribution shift function.
shift_fn = partial(F.adjust_gamma, gamma=0.2)

# Now we can shift the dataset!
shift_dataset(
    paths=paths_to_dataset_images,
    output_dir="/data/shifted_datasets/gamma_02",
    function=shift_fn,
    num_workers=20,
)
```

    Processing images |##########| 100000/100000 [00:49<00:00]

## Citation

If you use `StrongAugment` or **_shifted evaluation_**, please cite us!

```bibtex
@paper{strong_augment2022,
    title = {Augment like there's no tomorrow: Consistently performing neural networks for medical imaging},
    author = {Pohjonen, Joona and Stürenberg, Carolin and Föhr, Atte and Randen-Brady, Reija and Luomala, Lassi and Lohi, Jouni and Pitkänen, Esa and Rannikko, Antti and Mirtti, Tuomas},
    url = {https://arxiv.org/abs/2206.15274},
    publisher = {arXiv},
    year = {2022},
}
```
