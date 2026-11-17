"""Run VERIFAI on private real-world samples without touching training/robustness jobs.

Expected folders:
real_world_test/{original,whatsapp,ai_generated}/

Outputs (all git-ignored):
reports/real_world_test/predictions.csv
reports/real_world_test/summary.json
reports/real_world_test/gradcam/*.png
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from backend.model import Detector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CATEGORIES = {
    "original": "real",
    "whatsapp": "real",
    "ai_generated": "ai_generated",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("real_world_test"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/efficientnet_b0_best.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/real_world_test"))
    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Skip Grad-CAM overlays to make the run faster.",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(
            f"Trained checkpoint not found: {args.checkpoint}\n"
            "Set --checkpoint to your existing trained .pt file. No model will be trained."
        )

    missing = [name for name in CATEGORIES if not (args.data / name).is_dir()]
    if missing:
        raise SystemExit(
            f"Missing category folder(s) under {args.data}: {', '.join(missing)}"
        )

    detector = Detector(str(args.checkpoint))
    args.output.mkdir(parents=True, exist_ok=True)
    gradcam_dir = args.output / "gradcam"
    if not args.no_gradcam:
        gradcam_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    category_stats: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "correct": 0, "predicted_real": 0, "predicted_ai_generated": 0}
    )
    errors: list[dict] = []

    for category, expected_label in CATEGORIES.items():
        image_paths = sorted(
            p for p in (args.data / category).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for index, image_path in enumerate(image_paths, start=1):
            image_id = f"{category}_{index:03d}"
            try:
                with Image.open(image_path) as source:
                    source.load()
                    image = source.convert("RGB")
                prediction = detector.predict(image)
                predicted = prediction["label"]
                stats = category_stats[category]
                stats["count"] += 1
                stats["correct"] += int(predicted == expected_label)
                stats[f"predicted_{predicted}"] += 1

                gradcam_status = "skipped"
                if not args.no_gradcam:
                    encoded = detector.gradcam(image)
                    if encoded:
                        (gradcam_dir / f"{image_id}_gradcam.png").write_bytes(
                            base64.b64decode(encoded)
                        )
                        gradcam_status = "saved"
                    else:
                        gradcam_status = "unavailable"

                rows.append({
                    "image_id": image_id,
                    "category": category,
                    "expected_label": expected_label,
                    "predicted_label": predicted,
                    "correct": predicted == expected_label,
                    "confidence": prediction["confidence"],
                    "prob_ai_generated": prediction["probabilities"]["ai_generated"],
                    "prob_real": prediction["probabilities"]["real"],
                    "model": prediction["model"],
                    "device": prediction["device"],
                    "gradcam": gradcam_status,
                    "status": "ok",
                })
                print(
                    f"{image_id}: {predicted} "
                    f"(confidence={prediction['confidence']:.4f})"
                )
            except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
                errors.append({"image_id": image_id, "category": category, "error": type(exc).__name__})
                rows.append({
                    "image_id": image_id,
                    "category": category,
                    "expected_label": expected_label,
                    "predicted_label": "",
                    "correct": "",
                    "confidence": "",
                    "prob_ai_generated": "",
                    "prob_real": "",
                    "model": detector.name,
                    "device": str(detector.device),
                    "gradcam": "not_run",
                    "status": f"error:{type(exc).__name__}",
                })
                print(f"{image_id}: ERROR ({type(exc).__name__})")

    csv_path = args.output / "predictions.csv"
    fields = [
        "image_id", "category", "expected_label", "predicted_label", "correct",
        "confidence", "prob_ai_generated", "prob_real", "model", "device",
        "gradcam", "status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_categories = {}
    for category, stats in category_stats.items():
        evaluated = stats["count"]
        summary_categories[category] = {
            **stats,
            "accuracy": (stats["correct"] / evaluated) if evaluated else None,
        }

    successful = [row for row in rows if row["status"] == "ok"]
    summary = {
        "model": detector.name,
        "device": str(detector.device),
        "checkpoint": args.checkpoint.name,
        "total_images_found": len(rows),
        "successful_predictions": len(successful),
        "failed_images": len(errors),
        "overall_accuracy": (
            sum(bool(row["correct"]) for row in successful) / len(successful)
            if successful else None
        ),
        "categories": summary_categories,
        "errors": errors,
        "interpretation": (
            "Predictions are screening scores, not proof of authenticity. "
            "Grad-CAM indicates model attention, not verified manipulated pixels. "
            "The original and WhatsApp folders are both labelled real for this comparison."
        ),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== VERIFAI REAL-WORLD TEST SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV: {csv_path}")
    print(f"Summary: {summary_path}")
    if not args.no_gradcam:
        print(f"Grad-CAM overlays: {gradcam_dir}")


if __name__ == "__main__":
    main()
