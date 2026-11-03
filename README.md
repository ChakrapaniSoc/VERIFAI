# VERIFAI — Explainable AI Image Forensics

VERIFAI is a research-backed screening tool for estimating whether an image is AI-generated. It combines fine-tuned image classifiers, Grad-CAM visual explanations, frequency-domain indicators and image metadata into a transparent report.

> **Important:** VERIFAI is a decision-support tool, not proof of authenticity. A classifier can be wrong, metadata can be stripped, and a heatmap shows model attention—not verified manipulation pixels. Always corroborate important claims independently.

## Project status

This repository contains the initial application and reproducible training pipeline. **No model has been trained and no accuracy claim is made yet.** You must obtain datasets, review their licenses, train/evaluate the models, and place selected weights in `artifacts/`.

## Planned experiments

- ResNet50, EfficientNet-B0 and ConvNeXt-Tiny using ImageNet-pretrained weights and fine-tuning.
- ImageFolder dataset layout with reproducible train/validation/test splits.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix and per-generator results.
- Grad-CAM explanations for supported CNN backbones.
- FFT spectrum summary, EXIF inspection and optional C2PA manifest presence check.
- Robustness evaluation for JPEG compression, resizing, crop and blur.
- FastAPI inference service and React/Vite client.

## Repository layout

```
backend/                 FastAPI API and forensic utilities
training/                Train, evaluate and robustness scripts
frontend/                React + TypeScript user interface
artifacts/               Put trained checkpoints here (not committed)
data/                    Dataset root (not committed)
reports/                 Evaluation outputs (not committed)
```

## 1. Environment

Python 3.10–3.12 recommended. A CUDA-enabled PyTorch build is recommended for NVIDIA GPUs.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch appropriate for your CUDA version from the official PyTorch install selector if the default pip build is CPU-only.

## 2. Dataset layout

Use datasets only after checking their terms, license and permitted research/commercial use. Do not redistribute datasets unless their license allows it.

```
data/
  train/
    real/
    ai_generated/
  val/
    real/
    ai_generated/
  test/
    real/
    ai_generated/
```

Use a **source-aware split**: keep near-duplicates, images from the same original source, and ideally generator families grouped appropriately. For a meaningful unseen-generator test, hold out one or more generators entirely from training and validation. Do not randomly split near-identical derivatives across splits.

## 3. Train models

```bash
python -m training.train --data data --model efficientnet_b0 --epochs 12 --batch-size 16
python -m training.train --data data --model resnet50 --epochs 12 --batch-size 16
python -m training.train --data data --model convnext_tiny --epochs 12 --batch-size 8
```

The script saves the best validation checkpoint under `artifacts/`. These are starting hyperparameters; tune them using validation data only. For a 6 GB GPU, reduce batch size or image size if CUDA runs out of memory.

## 4. Evaluate held-out test data

```bash
python -m training.evaluate --data data --checkpoint artifacts/efficientnet_b0_best.pt
```

Keep test data untouched until model selection is complete. Report confidence intervals and generator-wise performance where possible.

## 5. Run the API

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Set `VERIFAI_CHECKPOINT=artifacts/efficientnet_b0_best.pt` to load a trained checkpoint. Without one, the API returns HTTP 503 rather than pretending to make a prediction.

## 6. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL=http://localhost:8000` if the API is hosted elsewhere.

## Research datasets to investigate

- CIFAKE: useful introductory real-vs-synthetic benchmark; inspect its dataset card/license and limitations.
- GenImage: multi-generator benchmark useful for cross-generator evaluation.
- ForenSynths: synthetic-image forensics research dataset; verify access and terms.
- Consider newer localized-artifact datasets only after checking availability and annotation/license terms.

Dataset names are leads for research, not an assertion that they are bundled here or automatically licensed for every use.

## Security and privacy

- Uploaded images are processed in memory; this starter does not persist them.
- Do not upload sensitive or private images without authorization.
- Add authentication, rate limiting, file-size limits and abuse monitoring before public deployment.
- Do not expose raw stack traces in production.
- A missing EXIF/C2PA record is not evidence that an image is AI-generated.

## Next milestones

1. Review dataset licenses and prepare a source-aware split manifest.
2. Train all three candidate architectures.
3. Compare held-out and unseen-generator performance.
4. Add human-reviewed explanation quality analysis.
5. Deploy API and frontend with clear limitations and monitoring.
