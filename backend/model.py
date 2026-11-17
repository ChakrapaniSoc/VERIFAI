"""Model loading and Grad-CAM inference."""
from __future__ import annotations
import base64
import io
import logging
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

MODEL_NAMES = ("resnet50", "efficientnet_b0", "convnext_tiny")
CLASS_NAMES = ("ai_generated", "real")
logger = logging.getLogger(__name__)


def build_model(name: str, pretrained: bool = False) -> nn.Module:
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def target_layer(model: nn.Module, name: str):
    if name == "resnet50":
        return model.layer4[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    if name == "convnext_tiny":
        return model.features[-1][-1]
    raise ValueError(name)


class Detector:
    def __init__(self, checkpoint: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint must contain a dictionary.")
        if payload.get("model_name") != "efficientnet_b0":
            raise ValueError("Checkpoint must contain an EfficientNet-B0 model.")
        if not isinstance(payload.get("state_dict"), dict):
            raise ValueError("Checkpoint is missing a valid state_dict.")
        classes = payload.get("classes")
        class_to_idx = payload.get("class_to_idx")
        if classes is not None and classes != list(CLASS_NAMES):
            raise ValueError("Checkpoint classes do not match the VERIFAI class order.")
        if class_to_idx is not None and class_to_idx != {
            name: index for index, name in enumerate(CLASS_NAMES)
        }:
            raise ValueError("Checkpoint class mapping does not match the VERIFAI class order.")
        if payload.get("image_size", 224) != 224:
            raise ValueError("Checkpoint image_size must be 224.")
        self.name = payload["model_name"]
        self.model = build_model(self.name, pretrained=False)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        idx = int(max(range(2), key=lambda i: probs[i]))
        return {
            "label": CLASS_NAMES[idx],
            "probabilities": {"ai_generated": round(probs[0], 6), "real": round(probs[1], 6)},
            "confidence": round(probs[idx], 6),
            "model": self.name,
            "device": str(self.device),
            "explanation": "Probabilities are model scores, not calibrated certainty or proof.",
        }

    def gradcam(self, image: Image.Image) -> str | None:
        try:
            from training.gradcam import colorize_heatmap, compute_gradcam

            rgb = image.convert("RGB")
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            cam, _predicted_index, _confidence = compute_gradcam(
                self.model,
                tensor,
                target_layer(self.model, self.name),
                self.device,
            )
            heatmap = colorize_heatmap(cam, rgb.size)
            overlay = Image.blend(rgb, heatmap, alpha=0.4)
            out = io.BytesIO()
            overlay.save(out, format="PNG")
            return base64.b64encode(out.getvalue()).decode("ascii")
        except Exception:
            logger.exception("Grad-CAM generation failed.")
            return None
