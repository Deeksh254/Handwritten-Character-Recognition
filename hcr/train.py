"""Train the CharCNN on MNIST / EMNIST with real-world augmentation.

Quick-start examples
--------------------
# Letters + Digits + common symbols (recommended for real handwriting):
python -m hcr.train --dataset byclass --epochs 25 --out-dir runs/byclass

# Letters only (A-Z, fast smoke test):
python -m hcr.train --dataset letters --epochs 15

# After training, predict on a real image:
python -m hcr.predict --checkpoint runs/byclass/best.pt --image photo.png --mode paragraph
"""
import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .data import CLASSES, get_loaders
from .evaluate import evaluate_model
from .model import CharCNN
from .utils import get_device, set_seed

# Windows DataLoader workers deadlock when spawned from a module entry-point.
_DEFAULT_WORKERS = 0 if platform.system() == "Windows" else 2


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train handwritten character CNN with real-world augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="byclass", choices=list(CLASSES),
                   help="EMNIST split to train on. 'byclass' gives 62 classes "
                        "(0-9 + A-Z + a-z) for best real-world coverage.")
    p.add_argument("--data-root",    default="data")
    p.add_argument("--epochs",       type=int,   default=25)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--mixup-alpha",  type=float, default=0.3,
                   help="Mixup interpolation strength. Set 0 to disable.")
    p.add_argument("--label-smooth", type=float, default=0.10,
                   help="Label smoothing coefficient.")
    p.add_argument("--no-augment",   action="store_true",
                   help="Disable real-world data augmentation.")
    p.add_argument("--num-workers",  type=int,   default=_DEFAULT_WORKERS)
    p.add_argument("--out-dir",      default=None,
                   help="Output directory. Defaults to runs/<dataset>.")
    p.add_argument("--patience",     type=int,   default=7,
                   help="Early-stopping patience (epochs without val improvement).")
    p.add_argument("--log-every",    type=int,   default=50,
                   help="Print a batch-level status line every N batches.")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Mixup
# ──────────────────────────────────────────────────────────────────────────────

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Beta-distributed linear interpolation of samples and labels."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────────────────────────────────────────
# Training / validation loops
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, device, optimizer, scheduler,
                mixup_alpha: float, log_every: int, epoch: int):
    model.train()
    total_loss, correct, n, t0 = 0.0, 0, 0, time.time()

    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        x_mix, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)

        pred = model(x_mix)
        loss = mixup_loss(criterion, pred, y_a, y_b, lam)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * x.size(0)
        # Accuracy measured on clean (unmixed) labels only
        correct += (model(x).argmax(1) == y).sum().item()
        n += x.size(0)

        if log_every > 0 and i % log_every == 0:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [{epoch}] batch {i:4d}/{len(loader)} | "
                  f"loss {total_loss/n:.4f} | acc {correct/n:.4f} | "
                  f"lr {lr_now:.2e} | {elapsed:.0f}s", flush=True)

    return total_loss / n, correct / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item() * x.size(0)
        correct += (pred.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_history(history: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ["loss", "acc"], ["Loss", "Accuracy"]):
        ax.plot(history[f"train_{key}"], label="train")
        ax.plot(history[f"val_{key}"],   label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    set_seed(args.seed)
    device  = get_device()
    out_dir = Path(args.out_dir or f"runs/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Dataset    : {args.dataset}  ({len(CLASSES[args.dataset])} classes)")
    print(f"Epochs     : {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}")
    print(f"Mixup α    : {args.mixup_alpha}  |  Label smooth: {args.label_smooth}")
    print(f"Out dir    : {out_dir}")
    print("=" * 60)

    train_loader, val_loader, test_loader, classes = get_loaders(
        args.dataset, args.data_root, args.batch_size,
        augment=not args.no_augment,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model     = CharCNN(len(classes)).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params : {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # OneCycleLR: warm-up + cosine decay in one schedule
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.10,         # 10 % warm-up
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

    history   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc  = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, criterion, device,
                             optimizer, scheduler, args.mixup_alpha,
                             args.log_every, epoch)
        vl, va = val_epoch(model, val_loader, criterion, device)

        for k, v in zip(history, (tl, ta, vl, va)):
            history[k].append(v)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch:02d}/{args.epochs} | "
              f"train loss {tl:.4f}  acc {ta:.4f} | "
              f"val loss {vl:.4f}  acc {va:.4f} | "
              f"{elapsed:.0f}s", flush=True)

        if va > best_acc:
            best_acc   = va
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "dataset":     args.dataset,
                "classes":     classes,
                "val_acc":     va,
                "epoch":       epoch,
            }, out_dir / "best.pt")
            print(f"  ✓ New best saved  ({best_acc:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{args.patience} epochs.")
            if no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, out_dir / "training_curves.png")

    print(f"\nBest val acc : {best_acc:.4f}")
    print("Evaluating best checkpoint on the test set ...")
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    evaluate_model(model, test_loader, device, classes, out_dir)


if __name__ == "__main__":
    main()
