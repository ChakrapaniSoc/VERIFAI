"""Measure VERIFAI checkpoint sensitivity to common image transformations.

Run from the repository root:
    python -m training.robustness --data data/test \
        --checkpoint artifacts/efficientnet_b0_best.pt

The report is diagnostic: it measures this checkpoint on the supplied test
folder and transformed copies. It does not establish performance on all
real-world images or prove image authenticity.
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torchvision import datasets

from backend.model import CLASS_NAMES, Detector


def build_variants() -> dict[str, Callable[[Image.Image], Image.Image]]:
    """Return deterministic transformations applied before model preprocessing."""

    def jpeg_variant(quality: int) -> Callable[[Image.Image], Image.Image]:
        def transform(image: Image.Image) -> Image.Image:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                return compressed.convert("RGB").copy()
        return transform

    def resize_variant(scale: float) -> Callable[[Image.Image], Image.Image]:
        def transform(image: Image.Image) -> Image.Image:
            width = max(1, round(image.width * scale))
            height = max(1, round(image.height * scale))
            resized = image.resize((width, height), Image.Resampling.BILINEAR)
            return resized.resize(image.size, Image.Resampling.BILINEAR)
        return transform

    return {
        "original": lambda image: image.copy(),
        "jpeg_quality_95": jpeg_variant(95),
        "jpeg_quality_85": jpeg_variant(85),
        "jpeg_quality_70": jpeg_variant(70),
        "jpeg_quality_50": jpeg_variant(50),
        "resize_down_up_75_percent": resize_variant(0.75),
        "resize_down_up_50_percent": resize_variant(0.50),
    }


def evaluate_variant(
    detector: Detector,
    samples: list[tuple[str, int]],
    transform: Callable[[Image.Image], Image.Image],
) -> dict:
    y_true: list[int] = []
    y_pred: list[int] = []
    ai_scores: list[float] = []

    for path, label in samples:
        try:
            with Image.open(path) as source:
                image = transform(source.convert("RGB"))
            result = detector.predict(image)
        except Exception as exc:
            raise RuntimeError(f"Could not evaluate image {path}: {exc}") from exc

        probabilities = result["probabilities"]
        y_true.append(label)
        y_pred.append(0 if result["label"] == CLASS_NAMES[0] else 1)
        ai_scores.append(float(probabilities[CLASS_NAMES[0]]))

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Rows are actual [AI-generated, real], columns are predicted
    # [AI-generated, real]. For real images, matrix[1, 0] is the false-positive
    # count (real incorrectly flagged as AI-generated).
    ai_count = int(matrix[0].sum())
    real_count = int(matrix[1].sum())

    try:
        auc = float(roc_auc_score([1 - value for value in y_true], ai_scores))
    except ValueError:
        auc = None

    return {
        "sample_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_ai_generated": auc,
        "real_false_positive_rate": (
            float(matrix[1, 0] / real_count) if real_count else None
        ),
        "ai_generated_false_negative_rate": (
            float(matrix[0, 1] / ai_count) if ai_count else None
        ),
        "confusion_matrix_actual_rows_predicted_columns": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VERIFAI robustness to JPEG compression and resizing."
    )
    parser.add_argument("--data", default="data/test", help="ImageFolder test directory.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/efficientnet_b0_best.pt",
        help="Path to a trained VERIFAI checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="reports/robustness_evaluation.json",
        help="Where to write the JSON report.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    if not data_dir.is_dir():
        parser.error(f"Test data directory does not exist: {data_dir}")
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint_path}")

    dataset = datasets.ImageFolder(str(data_dir))
    if dataset.class_to_idx != {name: index for index, name in enumerate(CLASS_NAMES)}:
        parser.error(
            "Dataset classes must be folders named 'ai_generated' and 'real'. "
            f"Found: {dataset.class_to_idx}"
        )
    if not dataset.samples:
        parser.error(f"No supported images found under: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = Detector(str(checkpoint_path))
    print(
        f"Evaluating {len(dataset.samples)} images on {device}; "
        f"class order: {', '.join(CLASS_NAMES)}"
    )

    results = {}
    for name, transform in build_variants().items():
        results[name] = evaluate_variant(detector, dataset.samples, transform)
        print(
            f"{name}: accuracy={results[name]['accuracy']:.4f}, "
            f"real FPR={results[name]['real_false_positive_rate']}"
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "test_data": str(data_dir),
        "device": str(detector.device),
        "class_order": list(CLASS_NAMES),
        "notes": [
            "This report applies only to the supplied test set and transformations.",
            "It does not establish generalization to all real-world photographs.",
            "Model probabilities are scores, not calibrated certainty or proof.",
        ],
        "variants": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved robustness report to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
