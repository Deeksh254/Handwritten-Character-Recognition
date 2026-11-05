# Handwritten Character Recognition

Recognise handwritten digits and letters using image processing + deep learning.

- **Datasets:** MNIST (digits) and EMNIST (digits, letters, balanced, byclass, bymerge), downloaded automatically by torchvision
- **App model:** One CRNN (CNN + bidirectional GRU + CTC) handles single characters, words/lines, and paragraph line crops
- **Character baseline:** SE-ResNet CharCNN remains available for isolated-character CLI prediction and comparison
- **Text baseline:** TrOCR `base-handwritten` remains available for CLI line recognition and comparison
- **Comparison baseline:** HOG (Histogram of Oriented Gradients) features with a multiclass linear SVM trained on the same MNIST/EMNIST splits
- **Web app:** Gradio UI (`app.py`) uses one CRNN checkpoint for every recognition mode
- **Evaluation:** CER/WER evaluation and a manifest-driven TrOCR fine-tuning workflow for labeled line images

## Project structure

```
handwritten-char-recognition/
├── app.py              # Gradio web demo
├── smoke_test.py       # CNN, CRNN, HOG, and preprocessing smoke checks
├── hcr/
│   ├── data.py         # MNIST/EMNIST loading, orientation fix, train/val/test split
│   ├── model.py        # CharCNN (SE-ResNet)
│   ├── train.py        # training loop (AdamW + OneCycleLR + augmentation)
│   ├── evaluate.py     # accuracy, classification report, confusion matrix
│   ├── train_hog_svm.py # HOG feature extraction and linear SVM benchmark
│   ├── metrics.py      # NumPy classification report (no scikit-learn)
│   ├── preprocess.py   # OpenCV thresholding + character segmentation
│   ├── predict.py      # CharCNN single-character and TrOCR text CLI
│   ├── htr.py          # TrOCR line crops, page deskew, and line detection
│   ├── train_htr.py    # fine-tune TrOCR on labeled line images
│   ├── evaluate_htr.py # real-image CER/WER evaluation
│   ├── crnn.py         # CNN + selectable BiLSTM/BiGRU + CTC model
│   ├── compare_crnn.py # seeded synthetic BiLSTM/BiGRU comparison and real-word examples
│   └── utils.py
├── samples/runs/       # example training histories and reports
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

**Train the unified app CRNN**

```bash
python -m hcr.crnn --dataset balanced --rnn gru --epochs 10 --train-size 5000 --val-size 1000 --out-dir runs/crnn
```

The app loads `runs/crnn/best.pt` by default; set `HCR_CRNN_CHECKPOINT` to use another
CRNN checkpoint. The current checkpoint uses a bidirectional GRU and was trained for
10 epochs on 5,000 synthetic words per epoch. It reached 39.9% exact-match on 1,000
validation words. On a fixed 512-word synthetic test set it scored 17.78% CER, 58.59%
WER, and 41.41% exact match. These results are synthetic: the sequences are stitched
from EMNIST characters, not natural word-level handwriting.

**Compare HOG + linear SVM against CharCNN**

```bash
python -m hcr.train_hog_svm --dataset byclass --out-dir runs/benchmarks/hog_svm_byclass
```

HOG summarizes local stroke directions into a fixed-length vector for each character;
the SVM then separates the 62 character classes using a linear decision boundary. This
baseline uses 324-dimensional HOG descriptors (9 orientations, 7x7 pixel cells, 2x2-cell
blocks) and an incremental multiclass linear SVM with hinge loss. It trains on the same
torchvision EMNIST ByClass training split as CharCNN, evaluates on the official test
split, and saves a joblib model, classification report, confusion matrix, and JSON metrics
in the specified output directory. With the current five-epoch settings, measured test
accuracy is 78.83% for HOG + SVM versus 87.33% for the existing CharCNN checkpoint, both
on the same 116,323 test images. This is a standalone benchmark; the Gradio app uses the
unified CRNN instead. These are isolated-character EMNIST results, not
real-photo accuracy.

**3. Recognize an isolated character** with the CharCNN CLI baseline (dark ink on light paper; add `--invert` otherwise)

```bash
python -m hcr.predict --checkpoint samples/runs/byclass/best.pt --image samples/letter.png --mode single --charset uppercase
```

Supported single-character alphabets are `any`, `digits`, `uppercase`, and `alphanumeric`.
The current classifiers have no punctuation classes, so use TrOCR for punctuation-bearing
text such as email addresses.

**4. Recognize words, lines, or paragraphs** with the separate pretrained TrOCR CLI baseline. The first use
downloads the model and processor from Hugging Face; model weights are about 1.33 GB.

```bash
python -m hcr.predict --image samples/word.png --mode word --save-vis out_word.png
python -m hcr.predict --image samples/sentence.png --mode paragraph --save-vis out_page.png
```

Text modes detect line crops and pass intact grayscale lines to TrOCR; they do not split
natural handwriting into character contours. Beam alternatives are ranked sequences,
not confidence estimates. `--checkpoint` is needed only for `--mode single`.

**Compare synthetic BiLSTM-CTC and BiGRU-CTC**

```bash
python -m hcr.compare_crnn --dataset balanced --train-size 1024 --validation-size 256 --test-size 512 --epochs 3 --seed 42
```

The runner creates deterministic synthetic words once and trains both recurrent types
with the same seed, data order, and epoch count. It reports synthetic-test CER, WER, and
exact match, then compares predictions on `samples/word.png` (`Dee`) and
`samples/mixed.png` (`B8Cd7`) against contour+CNN and TrOCR. Results from the 3-epoch
CPU run:

Each synthetic sample is one space-free random character string, so synthetic WER is
effectively a whole-string exact-match error rate; it is not natural-language sentence WER.

| Model | Synthetic test CER | Synthetic test WER | Exact match |
|---|---:|---:|---:|
| BiLSTM-CTC | 100.00% | 100.00% | 0.00% |
| BiGRU-CTC | 97.31% | 100.00% | 0.00% |

Real-image outputs (two examples only; qualitative, not a representative accuracy set):

| Image (reference) | BiLSTM-CTC | BiGRU-CTC | Contour + CNN | TrOCR |
|---|---|---|---|---|
| `word.png` (`Dee`) | (empty) | (empty) | `Dee` | `De e` |
| `mixed.png` (`B8Cd7`) | (empty) | (empty) | `B8Cd7` | `B8cd7` |

At this small training budget neither CRNN learned useful word decoding; treat this as
an architecture smoke comparison, not a competitive accuracy claim. Outputs, separate
CRNN checkpoints, and row-level real predictions go to `runs/benchmarks/crnn_comparison/`.

**5. Web app**

```bash
python app.py              # http://127.0.0.1:7860
python app.py --share      # also create a temporary public link
```

All app modes use the same CRNN checkpoint: single-character input is treated as a
one-character sequence, and text input is detected as lines before CTC decoding. The
CharCNN and TrOCR paths remain available only through their separate CLI/benchmark tools.

**6. Evaluate real handwriting**

```bash
python -m hcr.evaluate_htr --manifest samples/evaluation/manifest.csv
```

Create a labeled CSV using `samples/evaluation/manifest.template.csv` as its header:
`image,reference_text,source,stratum,split,error_stage`. Image paths are relative to the
CSV. Include at least 50 real examples from each stratum: `clean_scan`, `phone_photo`,
`low_light`, `cursive`, `paragraph`, and `hard_case` (300 examples total). Evaluation rows
must use `split=eval`. Attribute observed errors to `capture`, `line_detection`,
`recognition`, `ordering`, or `post_correction`. Reports include CER, WER, exact match,
and per-stratum/per-stage results. Do not use training examples for evaluation.

To temperature-calibrate isolated-character scores, prepare a separate CSV with
`image,label,split`, at least 30 held-out crops with `split=temperature` and another 30
with `split=threshold`. The sets are kept separate: one fits the temperature and the
other selects the review threshold. Do not reuse training or text-evaluation images.
Then run:

```bash
python -m hcr.calibrate_char --checkpoint samples/runs/byclass/best.pt --manifest data/calibration/characters.csv
```

This writes a checkpoint sidecar that CharCNN CLI prediction loads automatically; the
CRNN-only web app does not use this calibration. Its threshold targets 90% accuracy among
accepted calibration examples; this is not a guarantee of future accuracy. Without a
sidecar, top-three values are reported as uncalibrated raw scores. Use the header in
`samples/evaluation/char_calibration.template.csv`.

**7. Fine-tune on IAM or consented app samples**

Prepare line images and a CSV containing `image,reference_text,split`, with rows assigned
to `train` or `val`; paths are relative to the CSV. IAM access and data preparation are
not bundled. Then run:

```bash
python -m hcr.train_htr --manifest data/iam/lines.csv --epochs 3 --batch-size 1
```

Training applies mild rotation, perspective, brightness/contrast, and blur augmentation,
selects the best checkpoint by validation CER, and records validation CER/WER. This is
the optional TrOCR CLI fine-tuning path; it does not change the CRNN-only app. To select a
fine-tuned TrOCR model in the CLI, set `HCR_TROCR_MODEL` before running prediction. To
select a different CRNN checkpoint in the app, set `HCR_CRNN_CHECKPOINT`. In PowerShell:

```powershell
$env:HCR_TROCR_MODEL = "runs/htr/fine-tuned"
$env:HCR_CRNN_CHECKPOINT = "runs/crnn/best.pt"
python -m hcr.predict --image samples/word.png --mode word
python app.py --port 7863
```

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

- Try a Vision Transformer and compare against the SE-ResNet CharCNN
- Add an optional domain lexicon after line-level decoding for constrained fields
- Expand the consented real-image evaluation set and fine-tune TrOCR on its error cases

## License

MIT. See `LICENSE`.
