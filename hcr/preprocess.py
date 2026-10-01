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


def find_boxes(binary: np.ndarray, min_rel_height: float = 0.25):
    """Segment characters via external contours, sorted left-to-right.

    Boxes shorter than min_rel_height * tallest box are treated as noise.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    if not boxes:
        return []
    tallest = max(b[3] for b in boxes)
    boxes = [b for b in boxes if b[3] >= min_rel_height * tallest]
    return sorted(boxes, key=lambda b: b[0])


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
