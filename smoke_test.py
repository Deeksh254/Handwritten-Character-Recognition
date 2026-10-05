"""Quick pipeline smoke test — runs 3 batches through CNN and CRNN, exits non-zero on failure."""
import sys
import platform
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
from hcr.model import CharCNN
from hcr.crnn import CRNN, SyntheticWords, collate, greedy_decode

set_seed(42)
device = get_device()

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
print("=" * 40)
print("All smoke tests PASSED")
