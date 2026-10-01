# Handwritten Character Recognition

Recognise handwritten digits and letters using image processing + deep learning.

- **Datasets:** MNIST (digits) and EMNIST (digits, letters, balanced, byclass, bymerge), downloaded automatically by torchvision
- **Model:** Convolutional Neural Network (3 conv blocks with BatchNorm, Dropout, and an MLP head)
- **Image processing:** OpenCV pipeline (Gaussian blur, Otsu threshold, contour segmentation, 28x28 normalisation) so the model can read your own photos and scans
- **Extension:** CRNN (CNN + BiLSTM + CTC) for word / sequence recognition

## Project structure

```
handwritten-char-recognition/
├── hcr/
│   ├── data.py         # MNIST/EMNIST loading, orientation fix, train/val/test split
│   ├── model.py        # CharCNN
│   ├── train.py        # training loop (AdamW + OneCycleLR + augmentation)
│   ├── evaluate.py     # accuracy, classification report, confusion matrix
│   ├── preprocess.py   # OpenCV thresholding + character segmentation
│   ├── predict.py      # inference on your own images (single char or word)
│   ├── crnn.py         # CRNN + CTC sequence-recognition extension
│   └── utils.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run all commands from the repo root.

**1. Train**

```bash
python -m hcr.train --dataset mnist --epochs 10        # digits only (10 classes)
python -m hcr.train --dataset balanced --epochs 15     # digits + letters (47 classes)
python -m hcr.train --dataset letters --epochs 15      # letters A-Z (26 classes)
python -m hcr.train --dataset byclass --epochs 15      # 62 classes (case-sensitive)
```

Outputs go to `runs/<dataset>/`: `best.pt`, `history.json`, `training_curves.png`,
`classification_report.txt`, `confusion_matrix.png`.

**2. Evaluate a checkpoint**

```bash
python -m hcr.evaluate --checkpoint runs/balanced/best.pt
```

**3. Predict on your own image** (dark ink on light paper; add `--invert` otherwise)

```bash
python -m hcr.predict --checkpoint runs/balanced/best.pt --image samples/a.png --mode single
python -m hcr.predict --checkpoint runs/balanced/best.pt --image samples/hello.png --mode word --save-vis out.png
```

`word` mode segments characters with contours and classifies each one. It works for
well-spaced, non-touching characters.

**4. Sequence modelling (CRNN)**

```bash
python -m hcr.crnn --dataset balanced --epochs 10
python -m hcr.crnn --predict samples/word.png --checkpoint runs/crnn/best.pt
```

The CRNN is trained on synthetic words made by stitching EMNIST characters together.
To recognise real handwritten text (cursive, touching letters), replace `SyntheticWords`
with a real word-level dataset such as IAM.

## Typical results

Approximate ranges you can expect after 10 to 15 epochs (exact numbers vary by seed and hardware):

| Dataset | Classes | Test accuracy |
|---|---|---|
| MNIST | 10 | ~99.5% |
| EMNIST digits | 10 | ~99.7% |
| EMNIST letters | 26 | ~94 to 95% |
| EMNIST balanced | 47 | ~89 to 91% |
| EMNIST byclass | 62 | ~86 to 88% |

Letters like `O`/`0`, `I`/`l`/`1`, and `S`/`5` are inherently ambiguous in EMNIST, so the
confusion matrix will show errors there. That is expected.

## Ideas to extend

- Add a Streamlit or Gradio drawing-canvas demo
- Try ResNet-style blocks or a Vision Transformer and compare
- Add spell-correction or a language model on top of the CRNN output
- Train the CRNN on IAM Handwriting Database for real words

## License

MIT (add a LICENSE file when you create the repo on GitHub).
