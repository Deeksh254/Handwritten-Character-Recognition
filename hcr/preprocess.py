"""Classic image-processing pipeline: turn a photo/scan into MNIST-style 28x28 characters."""
import cv2
import numpy as np


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_binary(gray: np.ndarray, invert: bool = False, dilate: int = 1) -> np.ndarray:
    """Return a binary image with WHITE strokes on a BLACK background (like MNIST).

    Assumes dark ink on light paper. Pass invert=True for light ink on a dark background.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    flag = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    _, binary = cv2.threshold(blur, 0, 255, flag + cv2.THRESH_OTSU)
    if dilate > 0:  # thicken thin pen strokes to resemble EMNIST/MNIST
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=dilate)
    return binary


def single_box(binary: np.ndarray):
    """Bounding box (x, y, w, h) around all ink: the whole image is one character."""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return []
    x, y = int(xs.min()), int(ys.min())
    return [(x, y, int(xs.max()) - x + 1, int(ys.max()) - y + 1)]


def find_boxes(binary: np.ndarray, min_rel_height: float = 0.25) -> list:
    """Segment characters via external contours, sorted left-to-right.

    Boxes shorter than min_rel_height * tallest box are treated as noise.
    After contour detection, overly-wide boxes (likely merged/touching characters)
    are split using a column-projection valley search.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    if not boxes:
        return []
    tallest = max(b[3] for b in boxes)
    boxes = [b for b in boxes if b[3] >= min_rel_height * tallest]
    boxes = sorted(boxes, key=lambda b: b[0])
    boxes = split_merged_boxes(binary, boxes)
    return boxes


def split_merged_boxes(binary: np.ndarray, boxes: list, split_ratio: float = 1.6,
                       min_char_w: int = 6) -> list:
    """Break boxes that are too wide (likely merged touching characters).

    For each box wider than ``split_ratio × median_width``, we look for
    ink-free (or near-empty) vertical valleys in the column-projection
    profile and cut there.  Falls back to an equal split if no valley found.
    """
    if len(boxes) < 2:
        return boxes

    widths = sorted(b[2] for b in boxes)
    median_w = widths[len(widths) // 2]
    thresh_w = max(min_char_w * 2, split_ratio * median_w)

    result = []
    for box in boxes:
        x, y, w, h = box
        if w < thresh_w:
            result.append(box)
            continue

        # Estimate how many chars are merged
        n_chars = max(2, round(w / median_w))
        strip = binary[y: y + h, x: x + w].astype(np.float32)
        col_proj = strip.sum(axis=0)          # (W,) – ink mass per column

        # Find n_chars-1 valley positions
        splits = _valley_splits(col_proj, n_chars, min_char_w)

        prev = 0
        for sp in splits:
            seg_w = sp - prev
            if seg_w >= min_char_w:
                result.append((x + prev, y, seg_w, h))
            prev = sp
        tail_w = w - prev
        if tail_w >= min_char_w:
            result.append((x + prev, y, tail_w, h))

    return sorted(result, key=lambda b: b[0])


def _valley_splits(col_proj: np.ndarray, n_chars: int, min_char_w: int) -> list:
    """Return n_chars-1 column indices that are ink valleys (best split points)."""
    w = len(col_proj)
    splits = []
    segment_w = w // n_chars

    for k in range(1, n_chars):
        center = k * w // n_chars
        half = max(min_char_w, segment_w // 3)
        lo = max(0, center - half)
        hi = min(w, center + half)
        local = col_proj[lo:hi]
        # Pick the column with the least ink in the local window
        valley = lo + int(np.argmin(local))
        splits.append(valley)

    return splits


def find_lines(binary: np.ndarray, min_gap_px: int = 4) -> list:
    """Split a binary image into horizontal text-line strips.

    Uses a horizontal projection profile (row-wise ink sum) to detect
    text lines separated by blank rows.

    Returns a list of ``(y_start, y_end)`` tuples (row indices into *binary*).
    """
    row_sums = binary.sum(axis=1).astype(np.float32)       # shape: (H,)
    ink_threshold = max(1.0, float(row_sums.max()) * 0.02)
    in_line = row_sums > ink_threshold

    segments, start = [], None
    for y, active in enumerate(in_line):
        if active and start is None:
            start = y
        elif not active and start is not None:
            segments.append([start, y])
            start = None
    if start is not None:
        segments.append([start, int(binary.shape[0])])

    # Merge segments separated by very small gaps (ascenders/descenders)
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= min_gap_px:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    return [(s, e) for s, e in merged]


def box_to_mnist(binary: np.ndarray, box) -> np.ndarray:
    """Crop a box, scale the longer side to 20px, center it on a 28x28 canvas."""
    x, y, w, h = box
    crop = binary[y:y + h, x:x + w]
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.uint8)
    y0, x0 = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas
