"""Dataset utilities for MNIST and EMNIST (auto-downloaded by torchvision).

Real-world handwriting augmentation pipeline
--------------------------------------------
Training uses a heavy augmentation stack designed to make the model robust
to the kinds of variation seen in actual handwritten text:

  PIL-space (before ToTensor):
    RandomAffine       – rotation, translation, scale, shear
    RandomPerspective  – simulate tilted paper / camera angle
    ElasticTransform   – simulate pen pressure & style variation
    GaussianBlur       – simulate out-of-focus / low-res scans

  Tensor-space (after ToTensor):
    RandomErasing      – randomly black out small patches (ink smudges/holes)
    AddGaussianNoise   – paper grain, scanner noise

These transforms cover the main failure modes of a model trained only on
clean EMNIST data when applied to real photographs of handwriting.
"""
import platform
import string

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Windows uses 'spawn' for multiprocessing; num_workers > 0 hangs unless
# the entry-point is guarded by `if __name__ == '__main__'`. Default to 0.
_DEFAULT_WORKERS = 0 if platform.system() == "Windows" else 2

MEAN, STD = 0.1307, 0.3081

_BALANCED = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
CLASSES = {
    "mnist":   list(string.digits),
    "digits":  list(string.digits),                # EMNIST digits
    "letters": list(string.ascii_uppercase),        # EMNIST letters (case merged)
    "balanced": _BALANCED,                          # 47 classes
    "bymerge": _BALANCED,                           # 47 classes
    "byclass": list(string.digits                   # 62 classes
                    + string.ascii_uppercase
                    + string.ascii_lowercase),
}


# ──────────────────────────────────────────────────────────────────────────────
# Custom transforms
# ──────────────────────────────────────────────────────────────────────────────

class TransposeImage:
    """EMNIST images are stored transposed; this fixes the orientation."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.transpose(Image.Transpose.TRANSPOSE)


class ShiftLabel:
    """EMNIST 'letters' labels are 1..26; shift to 0..25."""

    def __call__(self, y: int) -> int:
        return y - 1


class AddGaussianNoise:
    """Add small Gaussian noise to a tensor to simulate paper grain / scanner noise."""

    def __init__(self, std: float = 0.03):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Transform builders
# ──────────────────────────────────────────────────────────────────────────────

def build_transform(dataset: str, augment: bool, normalize: bool) -> transforms.Compose:
    """Build the full preprocessing pipeline.

    When ``augment=True`` a heavy stack of real-world transforms is applied
    (see module docstring).  ``normalize`` applies MNIST-style mean/std.
    """
    t = []

    # 1. EMNIST orientation fix (must come first)
    if dataset != "mnist":
        t.append(TransposeImage())

    # 2. PIL-space augmentations ── simulate real-world pen/paper variation
    if augment:
        t.extend([
            transforms.RandomAffine(
                degrees=14,
                translate=(0.13, 0.13),
                scale=(0.85, 1.15),
                shear=8,
                fill=0,                                 # fill with black (background)
            ),
            transforms.RandomPerspective(
                distortion_scale=0.30, p=0.45, fill=0
            ),
            transforms.ElasticTransform(
                alpha=35.0, sigma=4.5
            ),                                          # wavy strokes / pen flex
            transforms.GaussianBlur(
                kernel_size=3, sigma=(0.1, 1.0)
            ),                                          # blur from scan/camera
            transforms.RandomAdjustSharpness(
                sharpness_factor=3, p=0.3
            ),
        ])

    # 3. Rasterise to float tensor in [0, 1]
    t.append(transforms.ToTensor())

    # 4. Tensor-space augmentations
    if augment:
        t.extend([
            transforms.RandomErasing(
                p=0.25,
                scale=(0.01, 0.12),
                ratio=(0.3, 3.3),
                value=0,                                # erase = black
            ),
            AddGaussianNoise(std=0.025),
        ])

    # 5. Normalise
    if normalize:
        t.append(transforms.Normalize((MEAN,), (STD,)))

    return transforms.Compose(t)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset / loader factories
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(dataset: str, root: str, train: bool,
                  augment: bool = False, normalize: bool = True):
    if dataset not in CLASSES:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(CLASSES)}")
    tf = build_transform(dataset, augment, normalize)
    if dataset == "mnist":
        return datasets.MNIST(root, train=train, download=True, transform=tf)
    target_tf = ShiftLabel() if dataset == "letters" else None
    return datasets.EMNIST(root, split=dataset, train=train, download=True,
                           transform=tf, target_transform=target_tf)


def get_loaders(dataset: str, root: str = "data", batch_size: int = 128,
                augment: bool = True, val_split: float = 0.1,
                num_workers: int = _DEFAULT_WORKERS, seed: int = 42):
    """Return (train_loader, val_loader, test_loader, classes)."""
    train_aug   = build_dataset(dataset, root, train=True,  augment=augment)
    train_plain = build_dataset(dataset, root, train=True,  augment=False)   # for val
    test_set    = build_dataset(dataset, root, train=False, augment=False)

    n      = len(train_aug)
    n_val  = int(n * val_split)
    g      = torch.Generator().manual_seed(seed)
    perm   = torch.randperm(n, generator=g).tolist()
    train_set = Subset(train_aug,   perm[n_val:])
    val_set   = Subset(train_plain, perm[:n_val])

    pin = torch.cuda.is_available()
    kw  = dict(num_workers=num_workers, pin_memory=pin)
    train_loader = DataLoader(train_set, batch_size=batch_size,      shuffle=True,  **kw)
    val_loader   = DataLoader(val_set,   batch_size=batch_size * 2,  shuffle=False, **kw)
    test_loader  = DataLoader(test_set,  batch_size=batch_size * 2,  shuffle=False, **kw)
    return train_loader, val_loader, test_loader, CLASSES[dataset]


def get_test_loader(dataset: str, root: str = "data", batch_size: int = 256,
                    num_workers: int = _DEFAULT_WORKERS):
    test_set = build_dataset(dataset, root, train=False, augment=False)
    loader   = DataLoader(test_set, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
    return loader, CLASSES[dataset]
