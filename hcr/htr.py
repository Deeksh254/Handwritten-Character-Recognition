"""Line-level handwritten text recognition with pretrained TrOCR."""
from functools import lru_cache
import os

import cv2
import numpy as np
import torch
from PIL import Image

from .utils import get_device

DEFAULT_MODEL_ID = os.environ.get("HCR_TROCR_MODEL", "microsoft/trocr-base-handwritten")


def _gray_image(image: np.ndarray, invert: bool = False) -> np.ndarray:
    gray = np.asarray(image)
    if gray.ndim == 3:
        if gray.shape[2] == 4:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)
    return gray.astype(np.uint8, copy=False)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def rectify_page(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Perspective-correct a clearly visible page quadrilateral when present."""
    gray = _gray_image(image, invert=invert)
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < height * width * 0.45:
            break
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(corners) != 4 or not cv2.isContourConvex(corners):
            continue
        points = corners.reshape(4, 2).astype(np.float32)
        sums = points.sum(axis=1)
        differences = np.diff(points, axis=1).reshape(-1)
        ordered = np.float32([
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ])
        top_left, top_right, bottom_right, bottom_left = ordered
        target_width = int(max(np.linalg.norm(bottom_right - bottom_left),
                               np.linalg.norm(top_right - top_left)))
        target_height = int(max(np.linalg.norm(top_right - bottom_right),
                                np.linalg.norm(top_left - bottom_left)))
        if min(target_width, target_height) < 32:
            continue
        target = np.float32([[0, 0], [target_width - 1, 0],
                             [target_width - 1, target_height - 1], [0, target_height - 1]])
        transform = cv2.getPerspectiveTransform(ordered, target)
        return cv2.warpPerspective(gray, transform, (target_width, target_height),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return gray


def image_quality_warnings(image: np.ndarray) -> list[str]:
    """Return conservative focus/exposure hints; never alter recognition output."""
    gray = _gray_image(image)
    warnings = []
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 35:
        warnings.append("Image may be out of focus; hold steady and retake if possible.")
    low, high = np.percentile(gray, [5, 95])
    if high < 110:
        warnings.append("Image appears underexposed; improve lighting and retake if possible.")
    elif low > 238:
        warnings.append("Image may be overexposed; reduce glare and retake if possible.")
    return warnings


def deskew_page(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Correct small page rotations while retaining grayscale for recognition."""
    gray = rectify_page(image, invert=invert)
    mask = _ink_mask(gray)
    points = np.column_stack(np.where(mask > 0))
    if len(points) < 32:
        return gray

    raw_angle = cv2.minAreaRect(points.astype(np.float32))[-1]
    correction = -(raw_angle + 90) if raw_angle < -45 else -raw_angle
    if abs(correction) < 0.35 or abs(correction) > 12:
        return gray

    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), correction, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def detect_text_lines(image: np.ndarray, invert: bool = False) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Deskew and find line crops using row projections, not character contours."""
    page = deskew_page(image, invert=invert)
    mask = _ink_mask(page)
    row_ink = np.count_nonzero(mask, axis=1)
    if not row_ink.size or row_ink.max() == 0:
        return page, []

    threshold = max(2, int(row_ink.max() * 0.018))
    active = (row_ink >= threshold).astype(np.uint8).reshape(-1, 1)
    gap = max(1, min(9, round(page.shape[0] * 0.002)))
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((gap, 1), np.uint8))[:, 0] > 0

    spans = []
    start = None
    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            spans.append((start, y))
            start = None
    if start is not None:
        spans.append((start, page.shape[0]))

    boxes = []
    min_height = max(5, round(page.shape[0] * 0.003))
    for y0, y1 in spans:
        if y1 - y0 < min_height:
            continue
        line_mask = mask[y0:y1]
        columns = np.flatnonzero(np.count_nonzero(line_mask, axis=0))
        if not columns.size:
            continue
        line_height = y1 - y0
        pad_y = max(2, round(line_height * 0.12))
        pad_x = max(3, round(line_height * 0.18))
        x0 = max(0, int(columns[0]) - pad_x)
        x1 = min(page.shape[1], int(columns[-1]) + 1 + pad_x)
        boxes.append((x0, max(0, y0 - pad_y), x1 - x0, min(page.shape[0], y1 + pad_y) - max(0, y0 - pad_y)))

    return page, boxes


@lru_cache(maxsize=2)
def _load_trocr(model_id: str, device_name: str):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    device = torch.device(device_name)
    model.to(device).eval()
    return processor, model, device


@torch.inference_mode()
def recognize_line_crops(
    page: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    model_id: str = DEFAULT_MODEL_ID,
    num_hypotheses: int = 3,
) -> list[tuple[str, list[str], tuple[int, int, int, int]]]:
    """Recognize each detected line and return beam alternatives with its box."""
    if not boxes:
        return []

    processor, model, device = _load_trocr(model_id, str(get_device()))
    predictions = []
    for box in boxes:
        x, y, width, height = box
        crop = page[y:y + height, x:x + width]
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        pixel_values = processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device, dtype=model.dtype)
        output_ids = model.generate(
            pixel_values,
            num_beams=max(4, num_hypotheses),
            num_return_sequences=num_hypotheses,
            max_new_tokens=128,
            early_stopping=True,
        )
        hypotheses = [text.strip() for text in processor.batch_decode(output_ids, skip_special_tokens=True)]
        primary = hypotheses[0] if hypotheses else ""
        predictions.append((primary, hypotheses, box))
    return predictions


def recognize_handwriting(
    image: np.ndarray,
    invert: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
) -> tuple[np.ndarray, list[tuple[str, list[str], tuple[int, int, int, int]]]]:
    page, boxes = detect_text_lines(image, invert=invert)
    predictions = recognize_line_crops(page, boxes, model_id=model_id)
    return page, predictions