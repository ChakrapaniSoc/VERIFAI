import contextlib
import io
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from backend.model import Detector
from training import train


MODEL_NAME = "efficientnet_b0"
CLASSES = ["ai_generated", "real"]
CLASS_TO_IDX = {"ai_generated": 0, "real": 1}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(2, 2)
        self.classifier = nn.Sequential(nn.Linear(2, 2))

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


class DummyScaler:
    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        pass


def create_checkpoint(path, epoch, freeze_epochs):
    model = TinyModel()
    learning_rate = 0.01
    optimizer, phase, optimizer_lr = train.configure_resume_optimizer(
        model, MODEL_NAME, freeze_epochs, epoch, learning_rate
    )
    optimizer.zero_grad(set_to_none=True)
    sum(param.square().sum() for param in model.parameters() if param.requires_grad).backward()
    optimizer.step()

    checkpoint = {
        "model_name": MODEL_NAME,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": {},
        "epoch": epoch,
        "best_validation_f1": 0.75,
        "history": [],
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "training_config": {
            "data": str(Path(path).parent.resolve()),
            "model": MODEL_NAME,
            "epochs": 4,
            "batch_size": 2,
            "lr": learning_rate,
            "workers": 0,
            "freeze_epochs": freeze_epochs,
        },
        "image_size": 224,
        "validation_f1": 0.75,
        **train.capture_random_states(),
    }
    torch.save(checkpoint, path)
    return torch.load(path, map_location="cpu", weights_only=False), model, phase, optimizer_lr


class TrainingResumeTests(unittest.TestCase):
    def restore_dummy_checkpoint(
        self, checkpoint, freeze_epochs, expected_phase, expected_lr, source_model
    ):
        model = TinyModel()
        optimizer, phase, optimizer_lr = train.configure_resume_optimizer(
            model, MODEL_NAME, freeze_epochs, checkpoint["epoch"], checkpoint["training_config"]["lr"]
        )
        self.assertEqual(phase, expected_phase)
        self.assertEqual(optimizer_lr, expected_lr)

        train.restore_checkpoint_state(
            checkpoint, model, optimizer, DummyScaler(), optimizer_lr, phase
        )

        expected_params = [param for param in model.parameters() if param.requires_grad]
        actual_params = [
            param for group in optimizer.param_groups for param in group["params"]
        ]
        self.assertEqual(len(actual_params), len(expected_params))
        self.assertTrue(all(actual is expected for actual, expected in zip(actual_params, expected_params)))
        self.assertEqual(optimizer.param_groups[0]["lr"], expected_lr)
        self.assertTrue(optimizer.state)
        for key, value in source_model.state_dict().items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))
        return model, optimizer

    def test_resume_before_freeze_transition_restores_warmup_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, source_model, _, _ = create_checkpoint(Path(directory) / "resume.pt", 1, 2)
            model, optimizer = self.restore_dummy_checkpoint(
                checkpoint, 2, "warm-up", 0.01, source_model
            )
            self.assertFalse(model.features.weight.requires_grad)
            self.assertTrue(model.classifier[0].weight.requires_grad)
            self.assertEqual(len(optimizer.param_groups[0]["params"]), 2)

            optimizer = train.configure_finetuning_optimizer(model, 0.01)
            self.assertTrue(all(param.requires_grad for param in model.parameters()))
            self.assertEqual(len(optimizer.param_groups[0]["params"]), 4)
            self.assertEqual(optimizer.param_groups[0]["lr"], 0.002)

    def test_resume_after_freeze_transition_restores_finetuning_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, source_model, _, _ = create_checkpoint(Path(directory) / "resume.pt", 2, 1)
            model, optimizer = self.restore_dummy_checkpoint(
                checkpoint, 1, "fine-tuning", 0.002, source_model
            )
            self.assertTrue(all(param.requires_grad for param in model.parameters()))
            self.assertEqual(len(optimizer.param_groups[0]["params"]), 4)

    def test_optimizer_restore_rejects_wrong_learning_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _, _, _ = create_checkpoint(Path(directory) / "resume.pt", 2, 1)
            model = TinyModel()
            _, phase, optimizer_lr = train.configure_resume_optimizer(
                model, MODEL_NAME, 1, 2, 0.02
            )
            optimizer = torch.optim.AdamW(
                (param for param in model.parameters() if param.requires_grad),
                lr=optimizer_lr,
            )
            with self.assertRaisesRegex(SystemExit, "learning rate"):
                train.restore_checkpoint_state(
                    checkpoint, model, optimizer, DummyScaler(), optimizer_lr, phase
                )

    def test_python_numpy_and_torch_rng_states_restore(self):
        random.seed(41)
        np.random.seed(41)
        torch.manual_seed(41)
        rng_state = train.capture_random_states()
        expected = (random.random(), np.random.random(), torch.rand(1))

        random.random()
        np.random.random()
        torch.rand(1)
        train.restore_random_states(rng_state)

        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        self.assertTrue(torch.equal(torch.rand(1), expected[2]))

    def test_invalid_checkpoint_epoch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _, _, _ = create_checkpoint(Path(directory) / "resume.pt", 1, 1)
            checkpoint["epoch"] = 0
            with self.assertRaisesRegex(SystemExit, "positive integer"):
                train.validate_resume_checkpoint(
                    checkpoint, MODEL_NAME, CLASSES, CLASS_TO_IDX, directory, 4
                )

    def test_incomplete_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "missing fields"):
            train.validate_resume_checkpoint({}, MODEL_NAME, CLASSES, CLASS_TO_IDX, ".", 4)

    def test_completed_requested_epochs_are_reported_without_remaining_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            (root / "train").mkdir(parents=True)
            (root / "val").mkdir()
            checkpoint_path = Path(directory) / "resume.pt"
            checkpoint, _, _, _ = create_checkpoint(checkpoint_path, 4, 1)
            checkpoint["training_config"]["data"] = str(root)
            torch.save(checkpoint, checkpoint_path)

            class DummyDataset:
                classes = CLASSES
                class_to_idx = CLASS_TO_IDX

                def __len__(self):
                    return 1

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                with patch.object(
                    sys,
                    "argv",
                    ["training.train", "--data", str(root), "--epochs", "4", "--resume", str(checkpoint_path)],
                ), patch.object(
                    train.datasets, "ImageFolder", return_value=DummyDataset()
                ), patch.object(train, "DataLoader") as data_loader, patch.object(
                    train, "build_model"
                ) as build_model:
                    train.main()
            self.assertIn("there are no remaining epochs to train", output.getvalue())
            self.assertIn("Best validation F1: 0.7500", output.getvalue())
            data_loader.assert_not_called()
            build_model.assert_not_called()

    def test_best_checkpoint_keeps_inference_compatible_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyModel()
            checkpoint_path = Path(directory) / "best.pt"
            torch.save({
                "model_name": MODEL_NAME,
                "state_dict": model.state_dict(),
                "classes": CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "image_size": 224,
                "validation_f1": 0.75,
            }, checkpoint_path)
            with patch("backend.model.build_model", return_value=TinyModel()):
                detector = Detector(str(checkpoint_path))
            self.assertEqual(detector.name, MODEL_NAME)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(detector.model.state_dict()[key], value))


if __name__ == "__main__":
    unittest.main()
