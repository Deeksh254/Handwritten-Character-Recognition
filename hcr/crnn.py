"""CRNN (CNN + BiLSTM + CTC) extension: recognise whole words / sequences.

There is no handwritten *word* dataset bundled with torchvision, so this script builds
synthetic "words" by stitching random EMNIST characters side by side. It demonstrates the
sequence-modelling pipeline; swap SyntheticWords for IAM (or any word dataset) for real text.

Train:    python -m hcr.crnn --dataset balanced --epochs 10
Predict:  python -m hcr.crnn --predict word.png --checkpoint runs/crnn/best.pt
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import CLASSES, build_dataset
from .preprocess import load_gray, to_binary
from .utils import get_device, set_seed


def trim_columns(img: torch.Tensor) -> torch.Tensor:
    """Remove empty columns left/right of a (1, H, W) character."""
    cols = (img[0].max(dim=0).values > 0.1).nonzero()
    if len(cols) == 0:
        return img[:, :, :4]
    return img[:, :, int(cols.min()):int(cols.max()) + 1]


class SyntheticWords(Dataset):
    """Random 'words' built from EMNIST characters. Labels are class index + 1 (0 = CTC blank)."""

    def __init__(self, base, size, min_len=3, max_len=6, max_gap=4, deterministic=False, seed=0):
        self.base, self.size = base, size
        self.min_len, self.max_len, self.max_gap = min_len, max_len, max_gap
        self.width = max_len * (28 + max_gap)  # 192 for defaults, divisible by 4
        self.deterministic, self.seed = deterministic, seed

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i if self.deterministic else None)
        n = int(rng.integers(self.min_len, self.max_len + 1))
        pieces, labels = [], []
        for k in rng.integers(0, len(self.base), size=n):
            img, y = self.base[int(k)]
            pieces.append(trim_columns(img))
            labels.append(int(y) + 1)
            pieces.append(torch.zeros(1, 28, int(rng.integers(1, self.max_gap + 1))))
        canvas = torch.cat(pieces[:-1], dim=2)[:, :, :self.width]
        pad = self.width - canvas.shape[2]
        canvas = nn.functional.pad(canvas, (0, pad))
        target = torch.tensor(labels, dtype=torch.long)
        return canvas, target, len(labels)


def collate(batch):
    imgs, targets, lengths = zip(*batch)
    return torch.stack(imgs), torch.cat(targets), torch.tensor(lengths, dtype=torch.long)


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),        # 14 x W/2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),      # 7 x W/4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1)),                                                     # 3 x W/4
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((3, 1)),                                                     # 1 x W/4
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden * 2, num_classes + 1)  # +1 for CTC blank at index 0

    def forward(self, x):                       # x: (B, 1, 28, W)
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)   # (B, T, 256), T = W/4
        out, _ = self.rnn(f)
        return self.fc(out)                     # (B, T, num_classes + 1) logits


def greedy_decode(logits: torch.Tensor, classes):
    """Collapse repeats and drop blanks."""
    results = []
    for seq in logits.argmax(2).cpu().numpy():
        chars, prev = [], 0
        for t in seq:
            if t != prev and t != 0:
                chars.append(classes[t - 1])
            prev = t
        results.append("".join(chars))
    return results


def split_targets(targets, lengths, classes):
    refs, offset = [], 0
    for l in lengths.tolist():
        refs.append("".join(classes[j - 1] for j in targets[offset:offset + l].tolist()))
        offset += l
    return refs


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    correct = total = 0
    for x, targets, lengths in loader:
        preds = greedy_decode(model(x.to(device)), classes)
        refs = split_targets(targets, lengths, classes)
        correct += sum(p == r for p, r in zip(preds, refs))
        total += len(refs)
    return correct / total


def word_image_to_tensor(path: str, width: int, invert: bool = False) -> torch.Tensor:
    """Rough preprocessing of a real word image into the CRNN input format (1, 1, 28, W)."""
    binary = to_binary(load_gray(path), invert=invert)
    ys, xs = np.nonzero(binary)
    binary = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = binary.shape
    new_h = 20
    new_w = max(1, min(width - 8, round(w * new_h / h)))
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, width), np.uint8)
    canvas[4:24, 4:4 + new_w] = resized
    return torch.from_numpy(canvas).float().div(255).unsqueeze(0).unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="balanced", choices=["balanced", "byclass", "bymerge", "letters", "digits"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-size", type=int, default=50000, help="synthetic words per epoch")
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--out-dir", default="runs/crnn")
    p.add_argument("--predict", default=None, help="path to a word image to recognise")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(allow_mps=False)  # CTC loss is not reliably supported on MPS

    # ---------- inference ----------
    if args.predict:
        ckpt = torch.load(args.checkpoint or f"{args.out_dir}/best.pt", map_location=device)
        classes = ckpt["classes"]
        model = CRNN(len(classes)).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        x = word_image_to_tensor(args.predict, ckpt["width"], args.invert).to(device)
        with torch.no_grad():
            print("Prediction:", greedy_decode(model(x), classes)[0])
        return

    # ---------- training ----------
    classes = CLASSES[args.dataset]
    base_train = build_dataset(args.dataset, args.data_root, train=True, normalize=False)
    base_test = build_dataset(args.dataset, args.data_root, train=False, normalize=False)
    train_ds = SyntheticWords(base_train, args.train_size)
    val_ds = SyntheticWords(base_test, args.val_size, deterministic=True, seed=12345)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    model = CRNN(len(classes)).to(device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    T = train_ds.width // 4
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, batches = 0.0, 0
        for x, targets, lengths in train_loader:
            x = x.to(device)
            log_probs = model(x).log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full((x.size(0),), T, dtype=torch.long)
            loss = ctc(log_probs, targets.to(device), input_lengths, lengths)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item()
            batches += 1
        acc = evaluate(model, val_loader, device, classes)
        print(f"Epoch {epoch:02d}/{args.epochs} | CTC loss {running / batches:.4f} | val word acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "width": train_ds.width, "dataset": args.dataset}, out_dir / "best.pt")

    print(f"Best validation exact-match accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
