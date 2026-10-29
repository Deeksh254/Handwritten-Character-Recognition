"""Quick pipeline smoke test — runs 3 batches through CNN and CRNN, exits non-zero on failure."""
import sys
import platform
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

print(f"Python {sys.version}")
print(f"PyTorch {torch.__version__}")
print(f"OS: {platform.system()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print()

from hcr.utils import get_device, set_seed
from hcr.data import build_dataset, CLASSES
from hcr.model import CharCNN, LegacyCharCNN, build_model_from_state_dict
from hcr.predict import classify_topk
from hcr.crnn import CRNN, SyntheticWords, collate, greedy_decode
from hcr.preprocess import to_binary, find_boxes, split_merged_boxes, box_to_mnist
from hcr.htr import detect_text_lines, image_quality_warnings
from hcr.calibrate_char import fit_temperature, fit_review_threshold
from hcr.train_hog_svm import describe_batch, FEATURE_COUNT
from app import coerce_image

set_seed(42)
device = get_device()

# ── Checkpoint and Gradio image compatibility ────────────────────────────────
print("=== Compatibility pipeline ===")
legacy = LegacyCharCNN(len(CLASSES["balanced"]))
compatible = build_model_from_state_dict(len(CLASSES["balanced"]), legacy.state_dict())
assert compatible(torch.zeros(2, 1, 28, 28)).shape == (2, len(CLASSES["balanced"]))
legacy.eval()
digit_guesses = classify_topk(
    legacy, np.zeros((28, 28), dtype=np.uint8), device,
    CLASSES["balanced"], allowed_labels=set("0123456789"),
)
assert digit_guesses and all(CLASSES["balanced"][index].isdigit() for index, _ in digit_guesses)
temperature, nll_before, nll_after = fit_temperature(
    torch.tensor([[4.0, 1.0], [1.0, 4.0], [2.0, 1.0], [1.0, 2.0]]),
    torch.tensor([0, 1, 0, 1]),
)
assert 0.05 <= temperature <= 20.0 and nll_after <= nll_before
threshold, coverage, accepted_accuracy = fit_review_threshold(
    torch.tensor([[6.0, 0.0], [0.0, 6.0], [0.8, 0.0], [0.0, 0.8]]),
    torch.tensor([0, 1, 1, 0]),
    temperature=1.0,
)
assert 0.0 <= threshold <= 1.01 and 0.0 <= coverage <= 1.0 and 0.0 <= accepted_accuracy <= 1.0
composite = np.zeros((8, 8, 3), dtype=np.uint8)
assert coerce_image({"composite": composite}) is composite
assert coerce_image({"composite": None, "image": composite}) is composite
print("  legacy checkpoint architecture and Gradio image dict: passed")
print("COMPATIBILITY PASSED\n")

# ── HOG descriptor ───────────────────────────────────────────────────────────
print("=== HOG feature pipeline ===")
hog_features = describe_batch(torch.zeros(2, 1, 28, 28))
assert hog_features.shape == (2, FEATURE_COUNT), hog_features.shape
assert np.isfinite(hog_features).all(), "HOG descriptor contains non-finite values"
print(f"  feature shape={hog_features.shape}")
print("HOG FEATURES PASSED\n")

# ── CNN ───────────────────────────────────────────────────────────────────────
print("=== CNN CharCNN pipeline ===")
base = build_dataset("mnist", "data", train=True, normalize=True)
loader = DataLoader(base, batch_size=64, shuffle=True, num_workers=0)
classes = CLASSES["mnist"]

model = CharCNN(len(classes)).to(device)
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for i, (x, y) in enumerate(loader):
    x, y = x.to(device), y.to(device)
    loss = crit(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"  train batch {i+1}: loss={loss.item():.4f}")
    if i >= 2:
        break

model.eval()
with torch.no_grad():
    x, y = next(iter(loader))
    acc = (model(x.to(device)).argmax(1).cpu() == y).float().mean().item()
    print(f"  val   acc={acc:.4f}")

print("CNN PASSED\n")

# ── CRNN ──────────────────────────────────────────────────────────────────────
print("=== CRNN+CTC pipeline ===")
base_tr = build_dataset("balanced", "data", train=True, normalize=False)
base_va = build_dataset("balanced", "data", train=False, normalize=False)
cc = CLASSES["balanced"]

ds_tr = SyntheticWords(base_tr, 256)
ds_va = SyntheticWords(base_va, 64, deterministic=True, seed=99)
tl = DataLoader(ds_tr, 32, shuffle=True,  collate_fn=collate, num_workers=0)
vl = DataLoader(ds_va, 32, shuffle=False, collate_fn=collate, num_workers=0)

crnn = CRNN(len(cc)).to(device)
ctc  = nn.CTCLoss(blank=0, zero_infinity=True)
opt2 = torch.optim.Adam(crnn.parameters(), lr=1e-3)
T    = ds_tr.width // 4

crnn.train()
for i, (x, tgt, lengths) in enumerate(tl):
    x = x.to(device)
    lp = crnn(x).log_softmax(2).permute(1, 0, 2)
    il = torch.full((x.size(0),), T, dtype=torch.long)
    loss = ctc(lp, tgt.to(device), il, lengths)
    opt2.zero_grad(); loss.backward(); opt2.step()
    print(f"  train batch {i+1}: ctc={loss.item():.4f}")
    if i >= 2:
        break

crnn.eval()
with torch.no_grad():
    x, tgt, lengths = next(iter(vl))
    preds = greedy_decode(crnn(x.to(device)), cc)
    print(f"  sample preds: {preds[:3]}")

print("CRNN PASSED\n")

# ── Preprocess / segmentation ─────────────────────────────────────────────────
print("=== OpenCV preprocess pipeline ===")

canvas = np.full((80, 220), 255, dtype=np.uint8)
# Two well-separated black letters
cv2.putText(canvas, "AB", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5)
binary = to_binary(canvas, invert=False, dilate=0)
boxes = find_boxes(binary)
assert boxes, "expected at least one character box"
char28 = box_to_mnist(binary, boxes[0])
assert char28.shape == (28, 28), char28.shape
assert char28.max() > 0, "crop should contain ink"

# A single wide rectangle should still be split using height as the width proxy
blob = np.zeros((40, 160), np.uint8)
blob[5:35, 5:155] = 255
split = split_merged_boxes(blob, [(0, 0, 160, 40)])
assert len(split) >= 2, f"expected merged blob to split, got {split}"
print(f"  separated boxes={len(boxes)}  merged-split={len(split)}")
print("PREPROCESS PASSED\n")

word_image = cv2.imread("samples/word.png", cv2.IMREAD_GRAYSCALE)
assert word_image is not None, "expected bundled word sample"
word_boxes = find_boxes(to_binary(word_image, dilate=1))
assert len(word_boxes) == 3, f"expected three sample letters, got {word_boxes}"
print("BUNDLED WORD SEGMENTATION PASSED\n")

page = np.full((240, 640, 3), 255, dtype=np.uint8)
cv2.putText(page, "first line", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
cv2.putText(page, "second line", (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
_, line_boxes = detect_text_lines(page)
assert len(line_boxes) == 2, f"expected two text lines, got {line_boxes}"
print("LINE-LEVEL HTR PREPROCESS PASSED\n")

photo = np.zeros((480, 640, 3), dtype=np.uint8)
corners = np.array([[100, 40], [550, 65], [500, 440], [75, 410]], dtype=np.int32)
cv2.fillConvexPoly(photo, corners, (245, 245, 245))
cv2.putText(photo, "first line", (140, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
cv2.putText(photo, "second line", (130, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
rectified, photo_boxes = detect_text_lines(photo)
assert rectified.shape != photo.shape[:2], "expected visible page quadrilateral to be rectified"
assert len(photo_boxes) == 2, f"expected two rectified lines, got {photo_boxes}"
assert image_quality_warnings(np.zeros((32, 32), dtype=np.uint8))
print("PAGE RECTIFICATION AND CAPTURE WARNINGS PASSED\n")

print("=" * 40)
print("All smoke tests PASSED")
