"""Model loading and Grad-CAM inference."""
from __future__ import annotations
import os
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import io

MODEL_NAMES = ("resnet50", "efficientnet_b0", "convnext_tiny")
CLASS_NAMES = ("ai_generated", "real")


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
            "model": self.name,
            "device": str(self.device),
            "explanation": "Probabilities are model scores, not calibrated certainty or proof.",
        }

    def gradcam(self, image: Image.Image) -> str | None:
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            from pytorch_grad_cam.utils.image import show_cam_on_image
            import base64
            import numpy as np
            rgb = image.convert("RGB").resize((224, 224))
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            with GradCAM(model=self.model, target_layers=[target_layer(self.model, self.name)]) as cam:
                grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(0)])[0]
            overlay = show_cam_on_image(arr, grayscale, use_rgb=True)
            out = io.BytesIO()
            Image.fromarray(overlay).save(out, format="PNG")
            return base64.b64encode(out.getvalue()).decode("ascii")
        except Exception:
            return None
