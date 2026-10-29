"""Train and evaluate a HOG + linear SVM on MNIST or EMNIST."""
import argparse
import json
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from skimage.feature import hog
from torch.utils.data import DataLoader

from .data import CLASSES, build_dataset
from .evaluate import plot_confusion_matrix
from .metrics import classification_report, confusion_matrix
from .utils import set_seed

HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (7, 7),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}
FEATURE_COUNT = 324


def describe_batch(images: torch.Tensor) -> np.ndarray:
    gray_images = images[:, 0].cpu().numpy()
    return np.stack([
        hog(image, **HOG_CONFIG).astype(np.float32, copy=False)
        for image in gray_images
    ])


def extract_features(dataset, output: Path, batch_size: int, num_workers: int) -> tuple[np.memmap, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    features = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32,
        shape=(len(dataset), FEATURE_COUNT),
    )
    labels = np.empty(len(dataset), dtype=np.int64)
    offset = 0
    started = time.time()
    for batch_index, (images, targets) in enumerate(loader, 1):
        batch_features = describe_batch(images)
        end = offset + len(targets)
        features[offset:end] = batch_features
        labels[offset:end] = targets.numpy()
        offset = end
        if batch_index % 100 == 0 or offset == len(dataset):
            print(f"  HOG {offset:,}/{len(dataset):,} images ({time.time() - started:.0f}s)", flush=True)
    features.flush()
    return features, labels


def predict_in_batches(model, features: np.ndarray, batch_size: int) -> np.ndarray:
    predictions = np.empty(len(features), dtype=np.int64)
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        predictions[start:end] = model.predict(features[start:end])
        if end % 10_000 < batch_size or end == len(features):
            print(f"  SVM predicted {end:,}/{len(features):,} images", flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(CLASSES), default="byclass")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/benchmarks/hog_svm_byclass"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.feature_batch_size < 1:
        parser.error("epochs and batch sizes must be positive")

    set_seed(args.seed)
    classes = CLASSES[args.dataset]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_set = build_dataset(args.dataset, args.data_root, train=True, normalize=False)
    test_set = build_dataset(args.dataset, args.data_root, train=False, normalize=False)
    print(f"Dataset: EMNIST/MNIST {args.dataset} | classes={len(classes)}")
    print(f"Train: {len(train_set):,} | Test: {len(test_set):,} | HOG features: {FEATURE_COUNT}")
    print(f"SVM: hinge loss, alpha={args.alpha}, epochs={args.epochs}")

    with tempfile.TemporaryDirectory(prefix="hcr_hog_svm_") as temporary:
        temporary = Path(temporary)
        train_features, train_labels = extract_features(
            train_set, temporary / "train_features.npy",
            args.feature_batch_size, args.num_workers,
        )
        test_features, test_labels = extract_features(
            test_set, temporary / "test_features.npy",
            args.feature_batch_size, args.num_workers,
        )

        model = SGDClassifier(
            loss="hinge",
            alpha=args.alpha,
            random_state=args.seed,
            average=True,
            learning_rate="optimal",
        )
        all_classes = np.arange(len(classes), dtype=np.int64)
        rng = np.random.default_rng(args.seed)
        training_history = []
        for epoch in range(1, args.epochs + 1):
            permutation = rng.permutation(len(train_labels))
            started = time.time()
            for start in range(0, len(permutation), args.batch_size):
                indices = permutation[start:start + args.batch_size]
                if epoch == 1 and start == 0:
                    model.partial_fit(train_features[indices], train_labels[indices],
                                      classes=all_classes)
                else:
                    model.partial_fit(train_features[indices], train_labels[indices])
            row = {"epoch": epoch, "seconds": round(time.time() - started, 1)}
            training_history.append(row)
            print(f"Epoch {epoch}/{args.epochs} complete in {row['seconds']}s", flush=True)

        predictions = predict_in_batches(model, test_features, args.batch_size)
        accuracy = float(np.mean(predictions == test_labels))
        matrix = confusion_matrix(test_labels, predictions, len(classes))
        train_features._mmap.close()
        test_features._mmap.close()
        del train_features, test_features
        report = classification_report(matrix, classes)
        (args.out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        plot_confusion_matrix(matrix, classes, args.out_dir / "confusion_matrix.png")
        metadata = {
            "algorithm": "HOG + SGDClassifier(loss='hinge') linear SVM",
            "dataset": args.dataset,
            "classes": classes,
            "train_examples": len(train_set),
            "test_examples": len(test_set),
            "accuracy": accuracy,
            "hog_config": HOG_CONFIG,
            "feature_count": FEATURE_COUNT,
            "epochs": args.epochs,
            "alpha": args.alpha,
            "seed": args.seed,
            "training_history": training_history,
        }
        joblib.dump({"model": model, "metadata": metadata}, args.out_dir / "model.joblib")
        (args.out_dir / "metrics.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(report)
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Artifacts saved to: {args.out_dir}")


if __name__ == "__main__":
    main()