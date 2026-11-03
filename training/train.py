"""Fine-tune ImageNet-pretrained torchvision classifiers on an ImageFolder dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from backend.model import build_model, MODEL_NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--model", choices=MODEL_NAMES, default="efficientnet_b0")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--freeze-epochs", type=int, default=1)
    args = p.parse_args()
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type=="cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type=="cuda")
    model = build_model(args.model, pretrained=True).to(device)
    # Warm-up classifier-only phase, then fine-tune the full network.
    for param in model.parameters():
        param.requires_grad = True
    if args.freeze_epochs > 0:
        if args.model == "resnet50":
            head = model.fc
        elif args.model == "efficientnet_b0":
            head = model.classifier
        else:
            head = model.classifier[2]
        for param in model.parameters():
            param.requires_grad = False
        for param in head.parameters():
            param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    best_f1 = -1.0
    Path("artifacts").mkdir(exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.2)
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
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx,
                "image_size": 224,
                "validation_f1": best_f1,
            }, f"artifacts/{args.model}_best.pt")
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/{args.model}_training.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
