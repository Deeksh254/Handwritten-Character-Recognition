"""Improved CharCNN with Squeeze-Excitation residual blocks.

Architecture: SE-ResNet  (28×28 → 14 → 7 → 3) + Global-Average-Pool + MLP head.

Key upgrades over the original conv_block model
-----------------------------------------------
* Residual skip connections → easier gradient flow on deeper nets
* Squeeze-Excitation (SE) attention → channel-wise feature re-calibration
* Global Average Pooling → removes spatial bias, more robust to small shifts
* GELU activation → smoother than ReLU, common in modern nets
* Wider channels (64-128-256) → more capacity for 62-class byclass model
"""
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel-wise attention.

    Learns to emphasise informative channels (e.g. curved strokes for 'C')
    and suppress uninformative ones, improving accuracy on real handwriting.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        bottleneck = max(4, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).view(x.shape[0], -1, 1, 1)
        return x * w


class ResBlock(nn.Module):
    """Residual block: two 3×3 convs + SE + MaxPool, with a matching skip.

    Spatial halving (÷2) via MaxPool on both the main path and the skip.
    """

    def __init__(self, ch_in: int, ch_out: int, dropout: float = 0.1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self.se = SEBlock(ch_out)

        # Skip connection — must also halve spatial dims
        if ch_in != ch_out:
            self.skip = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.MaxPool2d(2),
            )
        else:
            self.skip = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.main(x)) + self.skip(x)


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class CharCNN(nn.Module):
    """SE-ResNet for 28×28 handwritten character recognition.

    Spatial flow:  (1,28,28) → (64,14,14) → (128,7,7) → (256,3,3) → GAP → (256,)
    Then:          Linear(256→512) → GELU → Dropout → Linear(512→num_classes)

    Works for any number of classes (10 digits / 26 letters / 47 balanced / 62 byclass).
    """

    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ResBlock(1,   64,  dropout=0.05),   # 28 → 14
            ResBlock(64,  128, dropout=0.10),   # 14 →  7
            ResBlock(128, 256, dropout=0.15),   #  7 →  3
        )
        self.pool = nn.AdaptiveAvgPool2d(1)     # (256,3,3) → (256,1,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # → (256,)
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
