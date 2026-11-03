from __future__ import annotations
import io
import os
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from backend.forensics import inspect_metadata, frequency_summary, inspect_c2pa_presence
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
_detector = None


def get_detector():
    global _detector
    checkpoint = os.getenv("VERIFAI_CHECKPOINT", "artifacts/efficientnet_b0_best.pt")
    if not Path(checkpoint).is_file():
        raise HTTPException(status_code=503, detail="No trained checkpoint found. Train a model and set VERIFAI_CHECKPOINT.")
    if _detector is None:
        try:
            _detector = Detector(checkpoint)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not load model: {type(exc).__name__}")
    return _detector


@app.get("/health")
def health():
    checkpoint = os.getenv("VERIFAI_CHECKPOINT", "artifacts/efficientnet_b0_best.pt")
    return {"status": "ok", "model_loaded": _detector is not None, "checkpoint_exists": Path(checkpoint).is_file()}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WEBP or BMP image.")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MB limit.")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid supported image.")
    detector = get_detector()
    prediction = detector.predict(image)
    return {
        "filename": Path(file.filename or "uploaded-image").name,
        "prediction": prediction,
        "explainability": {"gradcam_png_base64": detector.gradcam(image), "note": "Grad-CAM visualizes model attention, not verified manipulated pixels."},
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
