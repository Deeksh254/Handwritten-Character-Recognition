"""Fit a temperature for a CharCNN using separate labeled character crops."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from .predict import load_model
from .preprocess import box_to_mnist, single_box, to_binary
from .data import MEAN, STD
from .utils import get_device


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float]:
    logits = logits.detach()
    labels = labels.to(device=logits.device, dtype=torch.long)
    before = float(F.cross_entropy(logits, labels))
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
    after = float(F.cross_entropy(logits / temperature, labels))
    return temperature, before, after


def fit_review_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    target_accuracy: float = 0.90,
) -> tuple[float, float, float]:
    probabilities = torch.softmax(logits / temperature, dim=1)
    scores, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels.to(device=predictions.device, dtype=torch.long))
    candidates = torch.unique(scores, sorted=True)
    threshold = 1.01
    for candidate in candidates:
        accepted = scores >= candidate
        if bool(accepted.any()) and float(correct[accepted].float().mean()) >= target_accuracy:
            threshold = float(candidate)
            break
    accepted = scores >= threshold
    coverage = float(accepted.float().mean())
    accuracy = float(correct[accepted].float().mean()) if bool(accepted.any()) else 0.0
    return threshold, coverage, accuracy


def load_calibration_logits(manifest: Path, model, classes: list[str], device, split: str):
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"image", "label", "split"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    allowed_splits = {"temperature", "threshold"}
    invalid_splits = sorted({row["split"].strip().lower() for row in rows} - allowed_splits)
    if invalid_splits:
        raise ValueError("Calibration split must be temperature or threshold")
    rows = [row for row in rows if row["split"].strip().lower() == split]
    if len(rows) < 30:
        raise ValueError(f"Use at least 30 held-out {split} crops; found {len(rows)}")

    logits, labels = [], []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            label = row["label"].strip()
            if label not in classes:
                raise ValueError(f"Label {label!r} is absent from this checkpoint's classes")
            path = (manifest.parent / row["image"].strip()).resolve()
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(f"Could not read calibration image: {path}")
            binary = to_binary(gray, dilate=0)
            boxes = single_box(binary)
            if not boxes:
                raise ValueError(f"No character found in calibration image: {path}")
            char28 = box_to_mnist(binary, boxes[0])
            x = torch.from_numpy(char28).float().div(255).unsqueeze(0).unsqueeze(0)
            x = ((x - MEAN) / STD).to(device)
            logits.append(model(x)[0].cpu())
            labels.append(classes.index(label))
    return torch.stack(logits), torch.tensor(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    device = get_device()
    model, classes = load_model(str(args.checkpoint), device)
    temperature_logits, temperature_labels = load_calibration_logits(
        args.manifest, model, classes, device, "temperature"
    )
    threshold_logits, threshold_labels = load_calibration_logits(
        args.manifest, model, classes, device, "threshold"
    )
    temperature, before, after = fit_temperature(temperature_logits, temperature_labels)
    review_threshold, coverage, accepted_accuracy = fit_review_threshold(
        threshold_logits, threshold_labels, temperature
    )
    report = {
        "temperature": temperature,
        "review_threshold": review_threshold,
        "review_target_accuracy": 0.90,
        "review_coverage": coverage,
        "review_accepted_accuracy": accepted_accuracy,
        "temperature_examples": len(temperature_labels),
        "threshold_examples": len(threshold_labels),
        "nll_before": before,
        "nll_after": after,
        "manifest": str(args.manifest),
        "splits": ["temperature", "threshold"],
    }
    output = args.checkpoint.with_suffix(".calibration.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Calibration saved to {output}")


if __name__ == "__main__":
    main()