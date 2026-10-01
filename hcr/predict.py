"""Predict characters from an image.

Examples:
    python -m hcr.predict --checkpoint runs/balanced/best.pt --image my_char.png --mode single
    python -m hcr.predict --checkpoint runs/balanced/best.pt --image my_word.png --mode word --save-vis out.png
"""
import argparse

import cv2
import numpy as np
import torch

from .data import MEAN, STD
from .model import CharCNN
from .preprocess import box_to_mnist, find_boxes, load_gray, single_box, to_binary
from .utils import get_device


def load_model(checkpoint: str, device):
    ckpt = torch.load(checkpoint, map_location=device)
    classes = ckpt["classes"]
    model = CharCNN(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes


@torch.no_grad()
def classify(model, char28: np.ndarray, device):
    x = torch.from_numpy(char28).float().div(255).unsqueeze(0).unsqueeze(0)
    x = ((x - MEAN) / STD).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(0)
    return int(idx), float(conf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--mode", choices=["single", "word"], default="single")
    p.add_argument("--invert", action="store_true", help="light ink on dark background")
    p.add_argument("--save-vis", default=None, help="save an annotated image here")
    args = p.parse_args()

    device = get_device()
    model, classes = load_model(args.checkpoint, device)
    gray = load_gray(args.image)
    binary = to_binary(gray, invert=args.invert)
    boxes = single_box(binary) if args.mode == "single" else find_boxes(binary)
    if not boxes:
        print("No characters found.")
        return

    text, details = "", []
    heights = [b[3] for b in boxes]
    for i, box in enumerate(boxes):
        idx, conf = classify(model, box_to_mnist(binary, box), device)
        if i > 0:
            prev = boxes[i - 1]
            if box[0] - (prev[0] + prev[2]) > 0.8 * float(np.mean(heights)):
                text += " "
        text += classes[idx]
        details.append((classes[idx], conf))

    print(f"Prediction: {text}")
    for ch, conf in details:
        print(f"  {ch!r}: {conf:.2%}")

    if args.save_vis:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h), (ch, _) in zip(boxes, details):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(vis, ch, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(args.save_vis, vis)
        print(f"Saved visualization to {args.save_vis}")


if __name__ == "__main__":
    main()
