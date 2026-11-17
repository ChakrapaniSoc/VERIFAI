import base64
import asyncio
import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import Headers, UploadFile

from backend import main


class StubDetector:
    def predict(self, _image):
        return {
            "label": "real",
            "probabilities": {"ai_generated": 0.2, "real": 0.8},
            "confidence": 0.8,
            "model": "efficientnet_b0",
            "explanation": "Test prediction.",
            "device": "cpu",
        }

    def gradcam(self, _image):
        return base64.b64encode(b"test overlay").decode("ascii")


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (4, 3), color=(20, 30, 40)).save(output, format="PNG")
    return output.getvalue()


class BackendApiTests(unittest.TestCase):
    def setUp(self):
        self.previous_detector = main._detector
        main._detector = StubDetector()

    def tearDown(self):
        main._detector = self.previous_detector

    def analyze(self, filename, data, content_type):
        upload = UploadFile(
            file=io.BytesIO(data),
            filename=filename,
            headers=Headers({"content-type": content_type}),
        )
        return asyncio.run(main.analyze(upload))

    def test_health_reports_model_state(self):
        response = main.health()

        self.assertTrue(response["model_loaded"])

    def test_detector_is_loaded_once_and_reused(self):
        main._detector = None
        with patch.object(main, "Detector", return_value=StubDetector()) as detector:
            first = main.get_detector()
            second = main.get_detector()

        self.assertIs(first, second)
        detector.assert_called_once()

    def test_analyze_preserves_frontend_response_contract_and_confidence(self):
        result = self.analyze("sample.png", png_bytes(), "image/png")

        self.assertEqual(
            set(result),
            {"filename", "prediction", "explainability", "forensics", "limitations"},
        )
        self.assertEqual(
            set(result["prediction"]),
            {"label", "probabilities", "confidence", "model", "explanation", "device"},
        )
        self.assertEqual(set(result["prediction"]["probabilities"]), {"ai_generated", "real"})
        self.assertEqual(
            set(result["explainability"]),
            {"gradcam_png_base64", "note"},
        )
        self.assertEqual(
            set(result["forensics"]),
            {"metadata", "frequency", "provenance"},
        )
        self.assertEqual(result["prediction"]["confidence"], 0.8)
        self.assertEqual(result["filename"], "sample.png")

    def test_analyze_rejects_unreadable_image(self):
        with self.assertRaises(HTTPException) as error:
            self.analyze("invalid.png", b"not an image", "image/png")

        self.assertEqual(error.exception.status_code, 400)

    def test_analyze_rejects_content_type_format_mismatch(self):
        with self.assertRaises(HTTPException) as error:
            self.analyze("sample.png", png_bytes(), "image/jpeg")

        self.assertEqual(error.exception.status_code, 415)

    def test_analyze_preserves_upload_size_limit(self):
        with self.assertRaises(HTTPException) as error:
            self.analyze("large.png", b"x" * (main.MAX_BYTES + 1), "image/png")

        self.assertEqual(error.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
