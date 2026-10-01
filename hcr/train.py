"""Train the CNN on MNIST / EMNIST.

Example:
    python -m hcr.train --dataset balanced --epochs 15
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .data import CLASSES, get_loaders
from .evaluate import evaluate_model
from .model import CharCNN
from .utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train handwritten character CNN")
    p.add_argument("--dataset", default="mnist", choices=list(CLASSES))
    p.add_argument("--data-root", default="data")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-augment", action="store_true", help="disable data augmentation")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", default=None, help="default: runs/<dataset>")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir or f"runs/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | dataset: {args.dataset}")

    train_loader, val_loader, test_loader, classes = get_loaders(
        args.dataset, args.data_root, args.batch_size,
        augment=not args.no_augment, num_workers=args.num_workers, seed=args.seed)

    model = CharCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        vl, va = run_epoch(model, val_loader, criterion, device)
        for k, v in zip(history, (tl, ta, vl, va)):
            history[k].append(v)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tl:.4f} acc {ta:.4f} "
              f"| val loss {vl:.4f} acc {va:.4f}")
        if va > best_acc:
            best_acc = va
            torch.save({"model_state": model.state_dict(), "dataset": args.dataset,
                        "classes": classes, "val_acc": va}, out_dir / "best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, out_dir / "training_curves.png")

    print(f"\nBest val acc: {best_acc:.4f}. Evaluating best checkpoint on the test set...")
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    evaluate_model(model, test_loader, device, classes, out_dir)


if __name__ == "__main__":
    main()
