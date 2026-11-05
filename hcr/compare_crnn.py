"""Compare BiLSTM-CTC and BiGRU-CTC on fixed synthetic words and real samples."""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from jiwer import cer, wer
from torch.utils.data import DataLoader

from .crnn import CRNN, SyntheticWords, collate, greedy_decode, split_targets, word_image_to_tensor
from .data import CLASSES, build_dataset
from .htr import recognize_handwriting
from .predict import load_model, predict_word
from .preprocess import load_gray, to_binary
from .utils import get_device, set_seed

REAL_WORDS = (
    ("samples/word.png", "Dee"),
    ("samples/mixed.png", "B8Cd7"),
)


def build_loader(dataset, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=0, generator=generator)


@torch.inference_mode()
def evaluate_synthetic(model, loader, device, classes) -> dict:
    model.eval()
    references, predictions = [], []
    for images, targets, lengths in loader:
        predictions.extend(greedy_decode(model(images.to(device)), classes))
        references.extend(split_targets(targets, lengths, classes))
    return {
        "examples": len(references),
        "cer": float(cer(references, predictions)),
        "wer": float(wer(references, predictions)),
        "exact_match": float(np.mean([ref == pred for ref, pred in zip(references, predictions)])),
    }


def train_variant(rnn_type: str, args, train_dataset, val_loader, classes, device) -> tuple[CRNN, dict]:
    set_seed(args.seed)
    model = CRNN(len(classes), hidden=args.hidden, rnn_type=rnn_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    train_loader = build_loader(
        train_dataset, args.batch_size, shuffle=True, seed=args.seed,
    )
    input_steps = train_dataset.width // 4
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_samples = 0.0, 0
        started = time.time()
        for images, targets, target_lengths in train_loader:
            images = images.to(device)
            log_probs = model(images).log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((len(images),), input_steps, dtype=torch.long)
            loss = criterion(log_probs, targets.to(device), input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)
            total_samples += len(images)

        row = {
            "epoch": epoch,
            "train_ctc_loss": total_loss / total_samples,
            "elapsed_seconds": round(time.time() - started, 2),
            "validation": evaluate_synthetic(model, val_loader, device, classes),
        }
        history.append(row)
        print(f"{rnn_type.upper()} {json.dumps(row)}", flush=True)

    return model, {"history": history, "epochs": args.epochs}


@torch.inference_mode()
def predict_crnn_word(model, image_path: str, width: int, classes, device) -> str:
    tensor = word_image_to_tensor(image_path, width).to(device)
    return greedy_decode(model(tensor), classes)[0]


def evaluate_real_words(models: dict, classes, device, output_dir: Path) -> list[dict]:
    cnn, cnn_classes = load_model("samples/runs/byclass/best.pt", device)
    records = []
    for image_path, reference in REAL_WORDS:
        gray = load_gray(image_path)
        cnn_binary = to_binary(gray, dilate=1)
        cnn_prediction, _ = predict_word(cnn_binary, cnn_classes, cnn, device)

        rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        _, htr_lines = recognize_handwriting(rgb)
        htr_prediction = " ".join(text for text, _, _ in htr_lines)

        record = {
            "image": image_path,
            "reference": reference,
            "contour_cnn": cnn_prediction,
            "trocr": htr_prediction,
        }
        for rnn_type, model in models.items():
            record[f"crnn_{rnn_type}"] = predict_crnn_word(
                model, image_path, model.input_width, classes, device
            )
        records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "real_word_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def real_word_metrics(records: list[dict], prediction_key: str) -> dict:
    references = [row["reference"] for row in records]
    predictions = [row[prediction_key] for row in records]
    return {
        "cer": float(cer(references, predictions)),
        "wer": float(wer(references, predictions)),
        "exact_match": float(np.mean([ref == pred for ref, pred in zip(references, predictions)])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["balanced", "byclass", "letters", "digits"], default="balanced")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmarks/crnn_comparison"))
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-real", action="store_true",
                        help="Skip the two bundled real-word samples and CNN/TrOCR comparisons")
    args = parser.parse_args()
    if min(args.train_size, args.validation_size, args.test_size, args.epochs,
           args.batch_size, args.hidden) < 1:
        parser.error("dataset sizes, epochs, batch size, and hidden size must be positive")

    device = get_device(allow_mps=False)
    classes = CLASSES[args.dataset]
    base_train = build_dataset(args.dataset, args.data_root, train=True, normalize=False)
    base_test = build_dataset(args.dataset, args.data_root, train=False, normalize=False)

    train_dataset = SyntheticWords(
        base_train, args.train_size, deterministic=True, seed=args.seed,
    )
    validation_dataset = SyntheticWords(
        base_test, args.validation_size, deterministic=True, seed=args.seed + 10_000,
    )
    test_dataset = SyntheticWords(
        base_test, args.test_size, deterministic=True, seed=args.seed + 20_000,
    )
    validation_loader = build_loader(validation_dataset, args.batch_size, False, args.seed + 1)
    test_loader = build_loader(test_dataset, args.batch_size, False, args.seed + 2)

    print(f"Device={device}; dataset={args.dataset}; classes={len(classes)}")
    print(f"Fixed synthetic words: train={args.train_size}, validation={args.validation_size}, test={args.test_size}; seed={args.seed}")
    models, results = {}, {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for rnn_type in ("lstm", "gru"):
        model, training = train_variant(rnn_type, args, train_dataset,
                                        validation_loader, classes, device)
        model.input_width = train_dataset.width
        test_metrics = evaluate_synthetic(model, test_loader, device, classes)
        result = {**training, "synthetic_test": test_metrics}
        models[rnn_type] = model
        results[rnn_type] = result
        checkpoint_path = args.output_dir / f"{rnn_type}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "classes": classes,
            "rnn_type": rnn_type,
            "width": train_dataset.width,
            "dataset": args.dataset,
            "seed": args.seed,
            "epochs": args.epochs,
            "train_size": args.train_size,
            "validation_size": args.validation_size,
            "test_size": args.test_size,
            "test_metrics": test_metrics,
        }, checkpoint_path)
        print(f"{rnn_type.upper()} synthetic test: {json.dumps(test_metrics)}")

    report = {
        "dataset": args.dataset,
        "seed": args.seed,
        "train_size": args.train_size,
        "validation_size": args.validation_size,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden,
        "synthetic": results,
    }
    if not args.skip_real:
        real_records = evaluate_real_words(models, classes, device, args.output_dir)
        report["real_word_samples"] = {
            "count": len(real_records),
            "predictions": real_records,
            "metrics": {
                key: real_word_metrics(real_records, key)
                for key in ["contour_cnn", "trocr", "crnn_lstm", "crnn_gru"]
            },
            "note": "Two bundled images only; qualitative sanity check, not a representative accuracy estimate.",
        }

    report_path = args.output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Comparison report saved to {report_path}")


if __name__ == "__main__":
    main()