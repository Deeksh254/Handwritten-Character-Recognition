"""Evaluate line-level HTR on a labeled, stratified real-image manifest."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from jiwer import cer, wer

from .htr import DEFAULT_MODEL_ID, recognize_handwriting

STRATA = ("clean_scan", "phone_photo", "low_light", "cursive", "paragraph", "hard_case")
ERROR_STAGES = ("capture", "line_detection", "recognition", "ordering", "post_correction")
REQUIRED_FIELDS = ("image", "reference_text", "source", "stratum", "split", "error_stage")


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not 300 <= len(rows) <= 500:
        raise ValueError(f"Evaluation manifest must contain 300-500 examples; found {len(rows)}")
    counts = Counter(row["stratum"].strip() for row in rows)
    shortages = [f"{name}={counts[name]}" for name in STRATA if counts[name] < 50]
    if shortages:
        raise ValueError("Need at least 50 examples in each stratum: " + ", ".join(shortages))
    unknown_strata = sorted(set(counts) - set(STRATA))
    if unknown_strata:
        raise ValueError("Unknown evaluation strata: " + ", ".join(unknown_strata))
    if any(row["split"].strip().lower() != "eval" for row in rows):
        raise ValueError("Evaluation manifest rows must all have split=eval")
    valid_stages = set(ERROR_STAGES) | {""}
    unknown_stages = sorted({row["error_stage"].strip() for row in rows} - valid_stages)
    if unknown_stages:
        raise ValueError("Unknown error stages: " + ", ".join(unknown_stages))
    return rows


def score_rows(rows: list[dict], manifest_path: Path, model_id: str) -> dict:
    predictions = []
    for row in rows:
        image_path = (manifest_path.parent / row["image"].strip()).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read evaluation image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, lines = recognize_handwriting(image, model_id=model_id)
        predicted = "\n".join(text for text, _, _ in lines)
        predictions.append({**row, "predicted_text": predicted})

    def metrics(items: list[dict]) -> dict:
        references = [item["reference_text"] for item in items]
        hypotheses = [item["predicted_text"] for item in items]
        return {
            "examples": len(items),
            "cer": float(cer(references, hypotheses)),
            "wer": float(wer(references, hypotheses)),
            "exact_match": sum(ref == hyp for ref, hyp in zip(references, hypotheses)) / len(items),
        }

    by_stratum = {
        stratum: metrics([item for item in predictions if item["stratum"].strip() == stratum])
        for stratum in STRATA
    }
    by_stage = defaultdict(list)
    for item in predictions:
        by_stage[item.get("error_stage", "").strip() or "unattributed"].append(item)

    return {
        "model_id": model_id,
        "overall": metrics(predictions),
        "by_stratum": by_stratum,
        "by_error_stage": {stage: metrics(items) for stage, items in sorted(by_stage.items())},
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, default=Path("runs/htr/evaluation.json"))
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    report = score_rows(rows, args.manifest, args.model_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))
    print(f"Full predictions saved to {args.output}")


if __name__ == "__main__":
    main()