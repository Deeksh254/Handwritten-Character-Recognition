"""Evaluate a trained checkpoint: accuracy, classification report, confusion matrix."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .data import get_test_loader
from .model import CharCNN
from .utils import get_device


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def plot_confusion_matrix(cm, classes, path):
    cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(6, len(classes) * 0.25)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8 if len(classes) < 30 else 6)
    ax.set_yticklabels(classes, fontsize=8 if len(classes) < 30 else 6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized confusion matrix")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_model(model, loader, device, classes, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = predict_loader(model, loader, device)
    acc = float((y_true == y_pred).mean())
    labels = list(range(len(classes)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                   digits=4, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(cm, classes, out_dir / "confusion_matrix.png")
    print(report)
    print(f"Test accuracy: {acc:.4f}")
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    loader, classes = get_test_loader(ckpt["dataset"], args.data_root)
    model = CharCNN(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    out_dir = args.out_dir or str(Path(args.checkpoint).parent)
    evaluate_model(model, loader, device, classes, out_dir)


if __name__ == "__main__":
    main()
