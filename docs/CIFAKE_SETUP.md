# Phase 1 — CIFAKE baseline setup

## Why start with CIFAKE?

CIFAKE is a small, balanced starter benchmark with 120,000 images: 60,000 real images from CIFAR-10 and 60,000 synthetic images generated using Stable Diffusion 1.4. Its original split has 100,000 training images and 20,000 test images (10,000 per class). The source images are only 32×32 pixels, so this dataset is useful for validating the training pipeline but is **not sufficient by itself** to claim reliable detection of modern, high-resolution images or unseen generators.

Dataset sources:
- Kaggle download: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- Dataset description, citation and stated MIT license: https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images
- Original CIFAKE paper: Bird & Lotfi (2024), IEEE Access, DOI: 10.1109/ACCESS.2024.3356122

Review the dataset's current terms and attribution requirements before use. Do not commit the downloaded dataset to this GitHub repository.

## Download

1. Sign in to Kaggle and download the CIFAKE archive from the dataset page above.
2. Extract it locally. The source folder should contain:
   `train/REAL`, `train/FAKE`, `test/REAL`, and `test/FAKE`.
3. Keep the official test folders unchanged.

For example, on Windows, extract to:
`D:\VERIFAI_DATA\cifake_raw\`

## Prepare the dataset (Windows PowerShell)

From the VERIFAI repository root:

```powershell
python -m training.prepare_cifake --source "D:\VERIFAI_DATA\cifake_raw" --output data --val-fraction 0.1 --seed 42 --mode copy
```

This creates:
- `data/train/ai_generated`
- `data/train/real`
- `data/val/ai_generated`
- `data/val/real`
- `data/test/ai_generated`
- `data/test/real`

The validation subset is drawn only from the original training folders. The official test folders remain untouched.

If the source and output are on the same drive, `--mode link` can save disk space by creating hard links:
```powershell
python -m training.prepare_cifake --source "D:\VERIFAI_DATA\cifake_raw" --output data --mode link
```

## Train the first baseline

Activate your Python virtual environment and install dependencies first:

```powershell
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build that matches your machine from the official PyTorch installation selector if `torch.cuda.is_available()` returns false.

Start with EfficientNet-B0 on the 6 GB RTX 4050:

```powershell
python -m training.train --data data --model efficientnet_b0 --epochs 12 --batch-size 8 --workers 2
```

If memory is stable, try batch size 16. If CUDA runs out of memory, lower the batch size to 4. Then train the comparison models:

```powershell
python -m training.train --data data --model resnet50 --epochs 12 --batch-size 8 --workers 2
python -m training.train --data data --model convnext_tiny --epochs 12 --batch-size 4 --workers 2
```

Evaluate each selected checkpoint on the untouched official test split:

```powershell
python -m training.evaluate --data data --checkpoint artifacts/efficientnet_b0_best.pt
```

Robustness checks:

```powershell
python -m training.robustness --data data/test --checkpoint artifacts/efficientnet_b0_best.pt
```

## Research limitation to record

CIFAKE's real images come from CIFAR-10 and synthetic images from Stable Diffusion 1.4. A model may learn dataset-specific or resolution-related shortcuts. Upscaling 32×32 images to 224×224 does not create new detail. Therefore:
- Use CIFAKE as a pipeline baseline only.
- Do not describe CIFAKE test accuracy as real-world accuracy.
- Next, use a higher-resolution multi-generator benchmark such as GenImage for cross-generator experiments. GenImage is large and its dataset terms restrict use to non-commercial research/teaching/publication; read the upstream license before downloading or using it.
