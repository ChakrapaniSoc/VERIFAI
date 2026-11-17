"""Generate Grad-CAM visual explanations for a trained VERIFAI checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, UnidentifiedImageError
from torchvision import transforms

from backend.model import build_model


SUPPORTED_IMAGE_FORMATS = {"BMP", "JPEG", "PNG", "TIFF", "WEBP"}
CLASS_NAMES = {"ai_generated", "real"}
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def load_detector(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
        raise SystemExit(f"Could not load checkpoint '{checkpoint_path}': {error}") from error

    if not isinstance(checkpoint, dict):
        raise SystemExit("Checkpoint must contain a dictionary.")
    if checkpoint.get("model_name") != "efficientnet_b0":
        raise SystemExit(
            "Unsupported checkpoint model; expected 'efficientnet_b0', "
            f"found {checkpoint.get('model_name')!r}."
        )
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise SystemExit("Checkpoint is missing a valid 'state_dict'.")
    if checkpoint.get("image_size", 224) != 224:
        raise SystemExit("Checkpoint image_size must be 224 to match the evaluation preprocessing.")

    classes = checkpoint.get("classes")
    class_to_idx = checkpoint.get("class_to_idx")
    if (
        not isinstance(classes, list)
        or set(classes) != CLASS_NAMES
        or class_to_idx != {name: index for index, name in enumerate(classes)}
    ):
        raise SystemExit("Checkpoint class mapping must contain ai_generated and real in matching order.")

    try:
        model = build_model("efficientnet_b0", pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Checkpoint weights are incompatible with EfficientNet-B0: {error}") from error
    model.to(device).eval()
    return model, classes


def load_image(image_path: Path):
    if not image_path.is_file():
        raise SystemExit(f"Image file not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            image_format = image.format
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
                raise SystemExit(
                    f"Unsupported image format {image_format!r}. Supported formats: {supported}."
                )
            return image.convert("RGB")
    except SystemExit:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise SystemExit(f"Could not read image '{image_path}': {error}") from error


def compute_gradcam(model, image_tensor, target_layer, device):
    activations = {}
    gradients = {}

    def capture_activation(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda gradient: gradients.__setitem__("value", gradient))

    hook = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(image_tensor.to(device))
            predicted_index = int(logits.argmax(dim=1).item())
            logits[0, predicted_index].backward()
    finally:
        hook.remove()

    if "value" not in activations or "value" not in gradients:
        raise RuntimeError("Grad-CAM could not capture target-layer activations and gradients.")

    feature_maps = activations["value"]
    feature_gradients = gradients["value"]
    channel_weights = feature_gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((channel_weights * feature_maps).sum(dim=1, keepdim=True))
    cam = F.interpolate(
        cam,
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    maximum = cam.max()
    if not torch.isfinite(maximum) or maximum.item() <= 0:
        raise RuntimeError("Grad-CAM produced an empty or invalid heatmap.")
    cam = (cam / maximum).detach().cpu().numpy()

    probabilities = torch.softmax(logits.detach(), dim=1)[0].cpu()
    confidence = float(probabilities[predicted_index].item())
    return cam, predicted_index, confidence


def colorize_heatmap(cam: np.ndarray, size: tuple[int, int]) -> Image.Image:
    gray = Image.fromarray(np.uint8(np.clip(cam, 0.0, 1.0) * 255), mode="L")
    gray = gray.resize(size, Image.Resampling.BILINEAR)
    values = np.asarray(gray, dtype=np.float32) / 255.0
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.uint8(rgb * 255), mode="RGB")


def save_visualizations(original: Image.Image, heatmap: Image.Image, output: Path, stem: str):
    output.mkdir(parents=True, exist_ok=True)
    original_path = output / f"{stem}_original.png"
    heatmap_path = output / f"{stem}_heatmap.png"
    overlay_path = output / f"{stem}_overlay.png"
    panel_path = output / f"{stem}_gradcam.png"

    overlay = Image.blend(original, heatmap, alpha=0.4)
    original.save(original_path)
    heatmap.save(heatmap_path)
    overlay.save(overlay_path)

    width, height = original.size
    scale = max(1, min(8, 512 // max(width, height)))
    panel_size = (width * scale, height * scale)
    panel = Image.new("RGB", (panel_size[0] * 3, panel_size[1] + 28), "white")
    draw = ImageDraw.Draw(panel)
    for index, (image, label) in enumerate((
        (original, "Original"),
        (heatmap, "Grad-CAM"),
        (overlay, "Overlay"),
    )):
        x = index * panel_size[0]
        panel.paste(image.resize(panel_size, Image.Resampling.NEAREST), (x, 28))
        draw.text((x + 4, 6), label, fill="black")
    panel.save(panel_path)
    return original_path, heatmap_path, overlay_path, panel_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/efficientnet_b0_best.pt",
        help="EfficientNet-B0 best checkpoint",
    )
    parser.add_argument(
        "--output",
        default="reports/gradcam",
        help="Output directory for original, heatmap, overlay, and comparison images",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original = load_image(image_path)
    model, classes = load_detector(checkpoint_path, device)
    image_size = 224
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    input_tensor = transform(original).unsqueeze(0)
    target_layer = model.features[-1]

    try:
        cam, predicted_index, confidence = compute_gradcam(
            model, input_tensor, target_layer, device
        )
    except RuntimeError as error:
        raise SystemExit(f"Could not generate Grad-CAM: {error}") from error

    heatmap = colorize_heatmap(cam, original.size)
    try:
        original_output, heatmap_output, overlay_output, panel_output = save_visualizations(
            original, heatmap, output_path, image_path.stem
        )
    except OSError as error:
        raise SystemExit(f"Could not save Grad-CAM outputs to '{output_path}': {error}") from error

    print(f"Predicted class: {classes[predicted_index]}")
    print(f"Prediction confidence: {confidence:.4f} ({confidence * 100:.2f}%)")
    print(f"Grad-CAM output: {panel_output}")
    print(f"Original image: {original_output}")
    print(f"Heatmap: {heatmap_output}")
    print(f"Overlay: {overlay_output}")


if __name__ == "__main__":
    main()
