"""Fine-tune ImageNet-pretrained torchvision classifiers on an ImageFolder dataset."""
from __future__ import annotations
import argparse
import math
import pickle
import random
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from backend.model import build_model, MODEL_NAMES


def capture_random_states():
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_states(checkpoint):
    rng_fields = (
        "python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_state",
    )
    missing_fields = [field for field in rng_fields if field not in checkpoint]
    if missing_fields:
        print("Warning: checkpoint is missing random state(s): " + ", ".join(missing_fields))

    try:
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda_rng_state = checkpoint.get("cuda_rng_state")
        if cuda_rng_state is not None:
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state])
            else:
                print("Warning: CUDA random state was saved, but CUDA is unavailable; skipping it.")
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Checkpoint contains invalid random state: {error}") from error


def validate_training_config(training_config, model_name):
    if not isinstance(training_config, dict):
        raise SystemExit("Checkpoint training_config must be a dictionary.")
    required_fields = {"data", "model", "epochs", "batch_size", "lr", "workers", "freeze_epochs"}
    missing_fields = required_fields.difference(training_config)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise SystemExit(f"Checkpoint training_config is incomplete; missing fields: {missing}.")
    if not isinstance(training_config["data"], str) or not training_config["data"].strip():
        raise SystemExit("Checkpoint training_config 'data' must be a non-empty path string.")
    if training_config["model"] != model_name:
        raise SystemExit("Checkpoint training_config model does not match the checkpoint model.")
    for field in ("epochs", "batch_size"):
        value = training_config[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise SystemExit(f"Checkpoint training_config '{field}' must be a positive integer.")
    for field in ("workers", "freeze_epochs"):
        value = training_config[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SystemExit(f"Checkpoint training_config '{field}' must be a non-negative integer.")
    learning_rate = training_config["lr"]
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise SystemExit("Checkpoint training_config 'lr' must be a finite positive number.")


def validate_optimizer_resume(optimizer, model, expected_lr, phase):
    expected_params = [param for param in model.parameters() if param.requires_grad]
    optimizer_params = [
        param for group in optimizer.param_groups for param in group["params"]
    ]
    if len(optimizer_params) != len(expected_params) or any(
        actual is not expected for actual, expected in zip(optimizer_params, expected_params)
    ):
        raise SystemExit(
            f"Checkpoint optimizer parameters do not match the expected {phase} phase."
        )
    if len(optimizer.param_groups) != 1 or not math.isclose(
        optimizer.param_groups[0]["lr"], expected_lr, rel_tol=1e-9
    ):
        raise SystemExit(
            f"Checkpoint optimizer learning rate does not match the expected {phase} phase "
            f"rate ({expected_lr:g})."
        )


def validate_resume_checkpoint(checkpoint, model_name, classes, class_to_idx, data_path, epochs):
    if not isinstance(checkpoint, dict):
        raise SystemExit("Resume checkpoint must contain a training checkpoint dictionary.")
    required_fields = {
        "model_name", "state_dict", "optimizer_state_dict", "scaler_state_dict", "epoch",
        "best_validation_f1", "history", "classes", "class_to_idx", "training_config",
        "image_size", "validation_f1",
    }
    missing_fields = required_fields.difference(checkpoint)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise SystemExit(f"Checkpoint cannot be resumed; missing fields: {missing}.")
    epoch = checkpoint["epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
        raise SystemExit("Checkpoint 'epoch' must be a positive integer.")
    if not isinstance(checkpoint["state_dict"], dict):
        raise SystemExit("Checkpoint 'state_dict' must be a dictionary.")
    if not isinstance(checkpoint["optimizer_state_dict"], dict):
        raise SystemExit("Checkpoint 'optimizer_state_dict' must be a dictionary.")
    if not isinstance(checkpoint["scaler_state_dict"], dict):
        raise SystemExit("Checkpoint 'scaler_state_dict' must be a dictionary.")
    if not isinstance(checkpoint["history"], list):
        raise SystemExit("Checkpoint 'history' must be a list.")
    for field in ("best_validation_f1", "validation_f1"):
        value = checkpoint[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
            or value > 1
            or not math.isfinite(value)
        ):
            raise SystemExit(f"Checkpoint '{field}' must be a finite number between 0 and 1.")
    if checkpoint["model_name"] != model_name:
        raise SystemExit(
            f"Checkpoint model is {checkpoint['model_name']!r}, but --model is {model_name!r}."
        )
    if checkpoint["classes"] != classes or checkpoint["class_to_idx"] != class_to_idx:
        raise SystemExit("Checkpoint class mapping does not match the current dataset.")
    training_config = checkpoint["training_config"]
    validate_training_config(training_config, checkpoint["model_name"])
    if Path(training_config["data"]).expanduser().resolve() != Path(data_path).expanduser().resolve():
        print(
            f"Warning: checkpoint dataset path is {training_config['data']!r}, "
            f"but the current --data path is {str(data_path)!r}."
        )
    if epoch >= epochs:
        print(
            f"Checkpoint is at epoch {epoch} and --epochs is {epochs}; "
            "there are no remaining epochs to train."
        )
    return training_config


def configure_resume_optimizer(model, model_name, freeze_epochs, completed_epochs, learning_rate):
    for param in model.parameters():
        param.requires_grad = True
    phase = "fine-tuning"
    if freeze_epochs > 0 and completed_epochs <= freeze_epochs:
        if model_name == "resnet50":
            head = model.fc
        elif model_name == "efficientnet_b0":
            head = model.classifier
        else:
            head = model.classifier[2]
        for param in model.parameters():
            param.requires_grad = False
        for param in head.parameters():
            param.requires_grad = True
        phase = "warm-up"
    optimizer_lr = learning_rate if phase == "warm-up" or freeze_epochs == 0 else learning_rate * 0.2
    optimizer = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=optimizer_lr,
    )
    return optimizer, phase, optimizer_lr


def configure_finetuning_optimizer(model, learning_rate):
    for param in model.parameters():
        param.requires_grad = True
    return torch.optim.AdamW(model.parameters(), lr=learning_rate * 0.2)


def restore_checkpoint_state(checkpoint, model, optimizer, scaler, expected_lr, phase):
    try:
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Could not restore resume checkpoint state: {error}") from error
    validate_optimizer_resume(optimizer, model, expected_lr, phase)
    restore_random_states(checkpoint)


def has_remaining_epochs(completed_epochs, requested_epochs):
    return completed_epochs < requested_epochs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--model", choices=MODEL_NAMES, default="efficientnet_b0")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--freeze-epochs", type=int, default=1)
    p.add_argument("--resume", help="Path to a training checkpoint to resume from")
    args = p.parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be a positive integer.")
    root = Path(args.data)
    if not all((root / split).is_dir() for split in ("train", "val")):
        raise SystemExit("Expected data/train and data/val folders.")
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406], [.229,.224,.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406], [.229,.224,.225]),
    ])
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
    expected = {"ai_generated", "real"}
    if set(train_ds.classes) != expected or train_ds.classes != val_ds.classes:
        raise SystemExit(f"Class folders must be {sorted(expected)} in both splits; found {train_ds.classes}, {val_ds.classes}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Dataset split is empty.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    start_epoch = 0
    if args.resume:
        if not Path(args.resume).is_file():
            raise SystemExit(f"Resume checkpoint not found: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
            raise SystemExit(f"Could not load resume checkpoint '{args.resume}': {error}") from error
        training_config = validate_resume_checkpoint(
            checkpoint, args.model, train_ds.classes, train_ds.class_to_idx, root, args.epochs
        )
        args.batch_size = training_config["batch_size"]
        args.lr = training_config["lr"]
        args.workers = training_config["workers"]
        args.freeze_epochs = training_config["freeze_epochs"]
        start_epoch = checkpoint["epoch"]
        print(f"Resuming {args.model} from {args.resume} after epoch {start_epoch}.")
        if not has_remaining_epochs(start_epoch, args.epochs):
            print(f"Best validation F1: {checkpoint['best_validation_f1']:.4f}")
            return
    else:
        print(f"Starting fresh training for {args.model}.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type=="cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type=="cuda")
    model = build_model(args.model, pretrained=checkpoint is None).to(device)
    # Warm-up classifier-only phase, then fine-tune the full network.
    criterion = nn.CrossEntropyLoss()
    optimizer, phase, optimizer_lr = configure_resume_optimizer(
        model, args.model, args.freeze_epochs, start_epoch, args.lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    best_f1 = -1.0 if checkpoint is None else checkpoint["best_validation_f1"]
    history = [] if checkpoint is None else checkpoint["history"]
    if checkpoint is not None:
        restore_checkpoint_state(checkpoint, model, optimizer, scaler, optimizer_lr, phase)

    Path("artifacts").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    training_config = {
        "data": str(root),
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "workers": args.workers,
        "freeze_epochs": args.freeze_epochs,
    }
    for epoch in range(start_epoch, args.epochs):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            optimizer = configure_finetuning_optimizer(model, args.lr)
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"train {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * y.size(0)
        model.eval()
        ys, ps = [], []
        with torch.inference_mode():
            for x, y in val_loader:
                logits = model(x.to(device))
                ps.extend(logits.argmax(1).cpu().tolist())
                ys.extend(y.tolist())
        acc = accuracy_score(ys, ps)
        f1 = f1_score(ys, ps, average="binary", pos_label=train_ds.class_to_idx["ai_generated"])
        row = {"epoch": epoch+1, "train_loss": running/len(train_ds), "val_accuracy": acc, "val_f1": f1}
        history.append(row)
        print(json.dumps(row))
        history_path = Path("reports") / f"{args.model}_training.json"
        history_path.write_text(json.dumps(history, indent=2))
        improved = f1 > best_f1
        if improved:
            best_f1 = f1
        saved_checkpoint = {
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch + 1,
            "best_validation_f1": best_f1,
            "history": history,
            "classes": train_ds.classes,
            "class_to_idx": train_ds.class_to_idx,
            "training_config": training_config,
            **capture_random_states(),
            "image_size": 224,
            "validation_f1": best_f1,
        }
        last_path = Path("artifacts") / f"{args.model}_last.pt"
        temporary_path = last_path.with_suffix(".pt.tmp")
        torch.save(saved_checkpoint, temporary_path)
        temporary_path.replace(last_path)
        if improved:
            torch.save(saved_checkpoint, f"artifacts/{args.model}_best.pt")
    print(f"Best validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
