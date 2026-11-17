from __future__ import annotations
import io
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.forensics import frequency_summary, inspect_c2pa_presence, inspect_metadata
from backend.model import Detector

app = FastAPI(title="VERIFAI API", version="0.1.0", description="Explainable AI-generated image screening and forensic indicators.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
MAX_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
    "image/bmp": "BMP",
}
_detector: Detector | None = None
_detector_lock = threading.Lock()
logger = logging.getLogger(__name__)


def get_detector():
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                checkpoint = os.getenv("VERIFAI_CHECKPOINT", "artifacts/efficientnet_b0_best.pt")
                if not Path(checkpoint).is_file():
                    raise HTTPException(
                        status_code=503,
                        detail="No trained checkpoint found. Set VERIFAI_CHECKPOINT to a valid checkpoint.",
                    )
                try:
                    _detector = Detector(checkpoint)
                except Exception as exc:
                    logger.exception("Could not load the VERIFAI model checkpoint.")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Could not load model: {type(exc).__name__}",
                    ) from exc
    return _detector


def decode_uploaded_image(data: bytes, content_type: str) -> Image.Image:
    expected_format = SUPPORTED_IMAGE_TYPES.get(content_type)
    if expected_format is None:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WEBP or BMP image.")

    try:
        with Image.open(io.BytesIO(data)) as candidate:
            if candidate.format != expected_format:
                raise HTTPException(
                    status_code=415,
                    detail="The uploaded image format does not match its content type.",
                )
            if candidate.width * candidate.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions exceed the supported limit.",
                )
            candidate.verify()

        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        ) from exc


@app.get("/health")
def health():
    checkpoint = os.getenv("VERIFAI_CHECKPOINT", "artifacts/efficientnet_b0_best.pt")
    return {"status": "ok", "model_loaded": _detector is not None, "checkpoint_exists": Path(checkpoint).is_file()}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 12 MB limit.")
        image = decode_uploaded_image(data, file.content_type or "")
        detector = get_detector()
        prediction = detector.predict(image)
        gradcam_png = detector.gradcam(image)
        gradcam_note = (
            "Grad-CAM visualizes model attention, not verified manipulated pixels."
            if gradcam_png is not None
            else "Grad-CAM could not be generated for this request. Prediction results remain available; see backend logs for details."
        )
        return {
            "filename": Path(file.filename or "uploaded-image").name,
            "prediction": prediction,
            "explainability": {"gradcam_png_base64": gradcam_png, "note": gradcam_note},
            "forensics": {
                "metadata": inspect_metadata(data),
                "frequency": frequency_summary(image),
                "provenance": inspect_c2pa_presence(data),
            },
            "limitations": [
                "This is a screening estimate, not proof of authenticity.",
                "Performance on unseen generators may differ from validation results.",
                "Missing metadata does not mean an image is AI-generated.",
                "Verify important claims using independent sources.",
            ],
        }
    finally:
        await file.close()
