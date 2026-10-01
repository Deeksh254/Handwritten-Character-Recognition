"""Dataset utilities for MNIST and EMNIST (auto-downloaded by torchvision)."""
import string

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = 0.1307, 0.3081

_BALANCED = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
CLASSES = {
    "mnist": list(string.digits),
    "digits": list(string.digits),                 # EMNIST digits
    "letters": list(string.ascii_uppercase),       # EMNIST letters (case merged)
    "balanced": _BALANCED,                         # 47 classes
    "bymerge": _BALANCED,                          # 47 classes
    "byclass": list(string.digits + string.ascii_uppercase + string.ascii_lowercase),  # 62
}


class TransposeImage:
    """EMNIST images are stored transposed; this fixes the orientation."""

    def __call__(self, img):
        return img.transpose(Image.Transpose.TRANSPOSE)


class ShiftLabel:
    """EMNIST 'letters' labels are 1..26; shift to 0..25."""

    def __call__(self, y):
        return y - 1


def build_transform(dataset: str, augment: bool, normalize: bool):
    t = []
    if dataset != "mnist":
        t.append(TransposeImage())
    if augment:
        t.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)))
    t.append(transforms.ToTensor())
    if normalize:
        t.append(transforms.Normalize((MEAN,), (STD,)))
    return transforms.Compose(t)


def build_dataset(dataset: str, root: str, train: bool, augment: bool = False, normalize: bool = True):
    if dataset not in CLASSES:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(CLASSES)}")
    tf = build_transform(dataset, augment, normalize)
    if dataset == "mnist":
        return datasets.MNIST(root, train=train, download=True, transform=tf)
    target_tf = ShiftLabel() if dataset == "letters" else None
    return datasets.EMNIST(root, split=dataset, train=train, download=True,
                           transform=tf, target_transform=target_tf)


def get_loaders(dataset: str, root: str = "data", batch_size: int = 128, augment: bool = True,
                val_split: float = 0.1, num_workers: int = 2, seed: int = 42):
    train_aug = build_dataset(dataset, root, train=True, augment=augment)
    train_plain = build_dataset(dataset, root, train=True, augment=False)  # for validation
    test_set = build_dataset(dataset, root, train=False)

    n = len(train_aug)
    n_val = int(n * val_split)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    train_set = Subset(train_aug, perm[n_val:])
    val_set = Subset(train_plain, perm[:n_val])

    pin = torch.cuda.is_available()
    kw = dict(num_workers=num_workers, pin_memory=pin)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False, **kw)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, **kw)
    return train_loader, val_loader, test_loader, CLASSES[dataset]


def get_test_loader(dataset: str, root: str = "data", batch_size: int = 256, num_workers: int = 2):
    test_set = build_dataset(dataset, root, train=False)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, CLASSES[dataset]
