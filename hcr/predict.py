"""Predict characters from an image.

Supported modes
---------------
single    – the whole image is a single character
word      – single-line handwritten word / short phrase
paragraph – multi-line handwritten text (letters + digits if model supports them)

Examples
--------
# Single character:
python -m hcr.predict --checkpoint runs/balanced/best.pt --image char.png --mode single

# Word (dark ink on white paper):
python -m hcr.predict --checkpoint runs/letters/best.pt --image word.png --mode word --save-vis out.png

# Full sentence / paragraph (use byclass model for digits + letters):
python -m hcr.predict --checkpoint runs/byclass/best.pt --image page.png --mode paragraph --save-vis out.png

Notes
-----
* For **letters only** use a model trained with ``--dataset letters`` (26 classes A-Z).
* For **letters + digits** train with ``--dataset byclass`` (62 classes 0-9 A-Z a-z).
* For **mixed alphanum balanced** use ``--dataset balanced`` (47 classes).
* Accuracy depends heavily on which checkpoint you load; the mode does NOT need to match.
"""
import argparse
import json
import math
import string
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import MEAN, STD
from .htr import DEFAULT_MODEL_ID, recognize_handwriting
from .model import build_model_from_state_dict
from .preprocess import (box_to_mnist, find_boxes, find_lines,
                         load_gray, single_box, to_binary)
from .utils import get_device


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, device):
    """Load a CharCNN checkpoint.  Works for any dataset (letters/balanced/byclass)."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = build_model_from_state_dict(len(classes), ckpt["model_state"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    calibration_path = Path(checkpoint).with_suffix(".calibration.json")
    if calibration_path.exists():
        temperature = float(
            json.loads(calibration_path.read_text(encoding="utf-8"))["temperature"]
        )
        if not math.isfinite(temperature) or not 0.05 <= temperature <= 20.0:
            raise ValueError(f"Invalid calibration temperature in {calibration_path}")
        review_threshold = float(
            json.loads(calibration_path.read_text(encoding="utf-8"))["review_threshold"]
        )
        if not math.isfinite(review_threshold) or not 0.0 <= review_threshold <= 1.01:
            raise ValueError(f"Invalid review threshold in {calibration_path}")
        model.calibration_temperature = temperature
        model.review_threshold = review_threshold
        model.is_temperature_calibrated = True
    else:
        model.calibration_temperature = 1.0
        model.review_threshold = 1.01
        model.is_temperature_calibrated = False
    return model, classes


# ──────────────────────────────────────────────────────────────────────────────
# Character classification
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def classify(model, char28: np.ndarray, device):
    """Return (class_index, confidence) for a single 28x28 uint8 crop."""
    x = torch.from_numpy(char28).float().div(255).unsqueeze(0).unsqueeze(0)
    x = ((x - MEAN) / STD).to(device)
    temperature = float(getattr(model, "calibration_temperature", 1.0))
    probs = torch.softmax(model(x)[0] / temperature, dim=0)
    conf, idx = probs.max(0)
    return int(idx), float(conf)


@torch.no_grad()
def classify_topk(
    model,
    char28: np.ndarray,
    device,
    classes: list[str],
    allowed_labels: set[str] | None = None,
    k: int = 3,
):
    """Return top-k softmax values, optionally adjusted by a held-out temperature."""
    x = torch.from_numpy(char28).float().div(255).unsqueeze(0).unsqueeze(0)
    x = ((x - MEAN) / STD).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    if allowed_labels is not None:
        indices = torch.tensor(
            [index for index, label in enumerate(classes) if label in allowed_labels],
            device=probs.device,
            dtype=torch.long,
        )
        if indices.numel() == 0:
            raise ValueError("The selected model has no classes in the allowed alphabet")
        probs = probs[indices]
        probs = probs / probs.sum()
        selected_indices = indices
    else:
        selected_indices = torch.arange(probs.numel(), device=probs.device)
    scores, positions = probs.topk(min(k, probs.numel()))
    indices = selected_indices[positions]
    return [(int(index), float(score)) for index, score in zip(indices, scores)]


# ──────────────────────────────────────────────────────────────────────────────
# Box → text helpers
# ──────────────────────────────────────────────────────────────────────────────

def _word_gap_threshold(boxes: list) -> float:
    """Adaptive gap threshold: 55% of the median character width."""
    if not boxes:
        return 0.0
    widths = sorted(b[2] for b in boxes)
    median_w = widths[len(widths) // 2]
    return 0.55 * median_w


def boxes_to_text(binary: np.ndarray, boxes: list, classes: list,
                  model, device, offset_y: int = 0) -> tuple:
    """Classify each box and assemble text, inserting spaces at word gaps.

    ``offset_y`` is added to y-coordinates so bounding boxes refer to the
    original (full-image) coordinate system when the boxes come from a strip.

    Returns ``(text_string, details_list)`` where each detail is
    ``(char, confidence, (x, y, w, h))`` in full-image coordinates.
    """
    if not boxes:
        return "", []

    gap_thresh = _word_gap_threshold(boxes)
    text, details = "", []

    for i, box in enumerate(boxes):
        x, y, w, h = box
        idx, conf = classify(model, box_to_mnist(binary, box), device)
        ch = classes[idx]
        full_box = (x, y + offset_y, w, h)

        # Insert a space when there is a visible gap between two characters
        if i > 0:
            prev = boxes[i - 1]
            gap = x - (prev[0] + prev[2])
            if gap > gap_thresh:
                text += " "

        text += ch
        details.append((ch, conf, full_box))

    return text, details


# ──────────────────────────────────────────────────────────────────────────────
# Per-mode predictors
# ──────────────────────────────────────────────────────────────────────────────

def predict_single(binary, classes, model, device):
    """Whole image = one character."""
    boxes = single_box(binary)
    if not boxes:
        return "", []
    idx, conf = classify(model, box_to_mnist(binary, boxes[0]), device)
    ch = classes[idx]
    return ch, [(ch, conf, boxes[0])]


def predict_word(binary, classes, model, device):
    """Single-line word / phrase."""
    boxes = find_boxes(binary)
    return boxes_to_text(binary, boxes, classes, model, device)


def predict_paragraph(binary, classes, model, device):
    """Multi-line text.

    Returns a list of ``(line_text, details)`` — one entry per detected line.
    """
    line_segs = find_lines(binary)
    results = []
    for y0, y1 in line_segs:
        strip = binary[y0:y1, :]
        boxes = find_boxes(strip)
        if not boxes:
            continue
        line_text, details = boxes_to_text(strip, boxes, classes, model, device,
                                           offset_y=y0)
        results.append((line_text, details))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ──────────────────────────────────────────────────────────────────────────────

def draw_details(vis: np.ndarray, details: list, font_scale: float = 0.7):
    """Draw character boxes and labels without implying calibrated confidence."""
    for ch, _, (x, y, w, h) in details:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 210, 0), 2)
        label = ch
        cv2.putText(vis, label, (x, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (60, 60, 220), 2,
                    cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Predict handwritten text from an image.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--checkpoint", default=None,
                   help="CharCNN checkpoint; required only for --mode single.")
    p.add_argument("--htr-model", default=DEFAULT_MODEL_ID,
                   help="Hugging Face line-recognition model ID.")
    p.add_argument("--charset", choices=["any", "digits", "uppercase", "alphanumeric"], default="any",
                   help="Restrict single-character predictions to a known alphabet.")
    p.add_argument("--image", required=True,
                   help="Path to the input image.")
    p.add_argument("--mode", choices=["single", "word", "paragraph"], default="word",
                   help=(
                       "single    - whole image is one character\n"
                       "word      - single-line word or phrase (default)\n"
                       "paragraph - multi-line text / full page"
                   ))
    p.add_argument("--invert", action="store_true",
                   help="Use when ink is light on a dark background.")
    p.add_argument("--dilate", type=int, default=1,
                   help="Dilation iterations to thicken strokes (default 1). "
                        "Use 0 for crisp printed text.")
    p.add_argument("--save-vis", default=None,
                   help="Save an annotated copy of the image here.")
    args = p.parse_args()

    device = get_device()
    print(f"Mode          : {args.mode}")
    print()

    gray = load_gray(args.image)
    if args.mode == "single":
        if not args.checkpoint:
            p.error("--checkpoint is required when --mode single")
        model, classes = load_model(args.checkpoint, device)
        print(f"Model loaded  : {len(classes)} classes  "
              f"({''.join(classes[:8])}{'...' if len(classes) > 8 else ''})")
        binary = to_binary(gray, invert=args.invert, dilate=args.dilate)
        text, details = predict_single(binary, classes, model, device)
        if not details:
            print("No character found.")
            return
        ch, _, _ = details[0]
        print(f"Prediction : {ch}")
        allowed_labels = {
            "digits": set("0123456789"),
            "uppercase": set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            "alphanumeric": set(string.ascii_letters + string.digits),
        }.get(args.charset)
        top = classify_topk(
            model, box_to_mnist(binary, details[0][2]), device, classes,
            allowed_labels=allowed_labels,
        )
        alternatives = ", ".join(f"{classes[index]} {score:.0%}" for index, score in top)
        scores_are_calibrated = model.is_temperature_calibrated and allowed_labels is None
        score_label = (
            "Top guesses (temperature-scaled probabilities)"
            if scores_are_calibrated
            else "Top guesses (raw, uncalibrated scores)"
        )
        print(f"{score_label}: {alternatives}")
        if not scores_are_calibrated:
            print("Scores are not calibrated confidence estimates.")
        elif top[0][1] < model.review_threshold:
            print("Review needed: top calibrated probability is below the held-out threshold.")
    else:
        page, line_results = recognize_handwriting(
            gray, invert=args.invert, model_id=args.htr_model
        )
        if not line_results:
            print("No text lines found.")
            return
        text = "\n".join(primary for primary, _, _ in line_results)
        for index, (primary, hypotheses, _) in enumerate(line_results, 1):
            print(f"Line {index}: {primary}")
            print(f"  Beam alternatives: {' | '.join(hypotheses[:3])}")
        print(f"\nFull text:\n{text}")

    # ── Optional visualisation ────────────────────────────────────────────────
    if args.save_vis:
        if args.mode == "single":
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            draw_details(vis, details)
        else:
            vis = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
            for index, (_, _, (x, y, width, height)) in enumerate(line_results, 1):
                cv2.rectangle(vis, (x, y), (x + width, y + height), (0, 200, 0), 2)
                cv2.putText(vis, f"Line {index}", (x, max(16, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2,
                            cv2.LINE_AA)
        cv2.imwrite(args.save_vis, vis)
        print(f"\nVisualization saved to: {args.save_vis}")


if __name__ == "__main__":
    main()
