"""Fine-tune pretrained TrOCR on labeled line-image CSV manifests."""
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from transformers import Adafactor, TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer, wer

from .htr import DEFAULT_MODEL_ID
from .utils import get_device, set_seed


class LineManifestDataset(Dataset):
    def __init__(self, manifest: Path, split: str, augment: bool = False):
        self.root = manifest.parent
        self.augment = augment
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"image", "reference_text", "split"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
            self.rows = [row for row in reader if row["split"].strip().lower() == split]
        if not self.rows:
            raise ValueError(f"No {split!r} rows found in {manifest}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["image"].strip()).resolve()
        image = Image.open(path).convert("RGB")
        if self.augment:
            image = augment_line(image)
        return image, row["reference_text"]


def augment_line(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        image = image.rotate(random.uniform(-2.5, 2.5), fillcolor="white", resample=Image.Resampling.BICUBIC)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.3))
    if random.random() < 0.2:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 0.8)))
    if random.random() < 0.25:
        pixels = np.asarray(image)
        height, width = pixels.shape[:2]
        margin_x, margin_y = max(1, width // 30), max(1, height // 30)
        source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        target = source + np.float32([
            [random.randint(-margin_x, margin_x), random.randint(-margin_y, margin_y)]
            for _ in range(4)
        ])
        matrix = cv2.getPerspectiveTransform(source, target)
        pixels = cv2.warpPerspective(pixels, matrix, (width, height),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        image = Image.fromarray(pixels)
    return image


def make_collate(processor: TrOCRProcessor, max_length: int):
    def collate(batch):
        images, texts = zip(*batch)
        pixel_values = processor(images=list(images), return_tensors="pt").pixel_values
        tokens = processor.tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        ).input_ids
        tokens[tokens == processor.tokenizer.pad_token_id] = -100
        return pixel_values, tokens
    return collate


@torch.inference_mode()
def validation_metrics(model, processor, loader, device):
    model.eval()
    references, hypotheses = [], []
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device=device, dtype=model.dtype)
        generated = model.generate(pixel_values, num_beams=4, max_new_tokens=128)
        hypotheses.extend(processor.batch_decode(generated, skip_special_tokens=True))
        labels[labels == -100] = processor.tokenizer.pad_token_id
        references.extend(processor.batch_decode(labels, skip_special_tokens=True))
    return float(cer(references, hypotheses)), float(wer(references, hypotheses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/htr/fine-tuned"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(allow_mps=False)
    model_dtype = torch.bfloat16 if device.type == "cpu" else torch.float32
    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(
        args.base_model,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)

    train_data = LineManifestDataset(args.manifest, "train", augment=True)
    val_data = LineManifestDataset(args.manifest, "val")
    collate = make_collate(processor, args.max_length)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)
    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, (pixel_values, labels) in enumerate(train_loader, 1):
            pixel_values = pixel_values.to(device=device, dtype=model.dtype)
            labels = labels.to(device)
            loss = model(pixel_values=pixel_values, labels=labels).loss
            (loss / args.gradient_accumulation).backward()
            total_loss += float(loss.detach())
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        val_cer, val_wer = validation_metrics(model, processor, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / len(train_loader),
               "val_cer": val_cer, "val_wer": val_wer}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_cer < best_cer:
            best_cer = val_cer
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            (args.output_dir / "best_metrics.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()