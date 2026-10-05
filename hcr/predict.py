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

import cv2
import numpy as np
import torch

from .data import MEAN, STD
from .model import CharCNN
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
    model = CharCNN(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes


# ──────────────────────────────────────────────────────────────────────────────
# Character classification
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def classify(model, char28: np.ndarray, device):
    """Return (class_index, confidence) for a single 28×28 uint8 crop."""
    x = torch.from_numpy(char28).float().div(255).unsqueeze(0).unsqueeze(0)
    x = ((x - MEAN) / STD).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(0)
    return int(idx), float(conf)


# ──────────────────────────────────────────────────────────────────────────────
# Box → text helpers
# ──────────────────────────────────────────────────────────────────────────────

def _word_gap_threshold(boxes: list) -> float:
    """Adaptive gap threshold: 55 % of the median character width."""
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
    """Draw green bounding boxes and red predicted labels on *vis* (in-place)."""
    for ch, conf, (x, y, w, h) in details:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"{ch}({conf:.0%})"
        cv2.putText(vis, label, (x, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 220), 2,
                    cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Predict handwritten text from an image.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to a trained checkpoint (.pt file).")
    p.add_argument("--image", required=True,
                   help="Path to the input image.")
    p.add_argument("--mode", choices=["single", "word", "paragraph"], default="word",
                   help=(
                       "single    – whole image is one character\n"
                       "word      – single-line word or phrase (default)\n"
                       "paragraph – multi-line text / full page"
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
    model, classes = load_model(args.checkpoint, device)

    print(f"Model loaded  : {len(classes)} classes  "
          f"({''.join(classes[:8])}{'...' if len(classes) > 8 else ''})")
    print(f"Mode          : {args.mode}")
    print()

    gray = load_gray(args.image)
    binary = to_binary(gray, invert=args.invert, dilate=args.dilate)

    # ── Dispatch to mode ──────────────────────────────────────────────────────
    if args.mode == "single":
        text, details = predict_single(binary, classes, model, device)
        if not details:
            print("No character found.")
            return
        ch, conf, _ = details[0]
        print(f"Prediction : {ch}")
        print(f"Confidence : {conf:.2%}")

    elif args.mode == "word":
        text, details = predict_word(binary, classes, model, device)
        if not details:
            print("No characters found.")
            return
        print(f"Prediction : {text}")
        print()
        for ch, conf, _ in details:
            print(f"  {ch!r}: {conf:.2%}")

    else:  # paragraph
        line_results = predict_paragraph(binary, classes, model, device)
        if not line_results:
            print("No text found.")
            return

        all_details = []
        print("Paragraph prediction:")
        print("-" * 40)
        for i, (line_text, details) in enumerate(line_results, 1):
            print(f"  Line {i:2d}: {line_text}")
            all_details.extend(details)
        print("-" * 40)
        full_text = "\n".join(lt for lt, _ in line_results)
        print(f"\nFull text:\n{full_text}")

        details = all_details  # for visualisation below

    # ── Optional visualisation ────────────────────────────────────────────────
    if args.save_vis:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        draw_details(vis, details)
        cv2.imwrite(args.save_vis, vis)
        print(f"\nVisualization saved to: {args.save_vis}")


if __name__ == "__main__":
    main()
