"""Lightweight forensic indicators. These are descriptive signals, not proof."""
from __future__ import annotations
import io
import math
from PIL import Image, ExifTags
import numpy as np


def inspect_metadata(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif = image.getexif()
        tags = {}
        for key, value in exif.items():
            name = ExifTags.TAGS.get(key, str(key))
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            tags[name] = str(value)[:300]
        return {
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "exif_present": bool(tags),
            "exif": tags,
            "interpretation": "Metadata may be absent after export, editing, or social-media processing; absence is not evidence of AI generation.",
        }
    except Exception as exc:
        return {"error": f"Metadata inspection failed: {type(exc).__name__}"}


def frequency_summary(image: Image.Image) -> dict:
    """Return coarse FFT spectrum descriptors; not a standalone AI detector."""
    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))
    h, w = magnitude.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    high = float(magnitude[radius > 65].mean())
    low = float(magnitude[radius <= 65].mean())
    return {
        "mean_log_magnitude": round(float(magnitude.mean()), 4),
        "high_frequency_mean": round(high, 4),
        "low_frequency_mean": round(low, 4),
        "high_to_low_ratio": round(high / (low + 1e-8), 4),
        "note": "Exploratory signal only. Do not interpret these values as proof of synthetic origin.",
    }


def inspect_c2pa_presence(image_bytes: bytes) -> dict:
    # Full C2PA verification requires a trusted verifier and signature/manifest validation.
    # This safe starter reports presence as unknown rather than claiming authenticity.
    return {
        "status": "not_verified",
        "message": "C2PA provenance verification is not configured. Presence or absence must not be inferred from this result.",
    }
