"""Prepare the official CIFAKE folder layout for VERIFAI.

Expected input (downloaded and extracted separately):
  cifake_raw/train/REAL, cifake_raw/train/FAKE
  cifake_raw/test/REAL,  cifake_raw/test/FAKE

The official test split is copied unchanged. A deterministic validation split
is sampled only from the official training split. Images are copied, not
modified. The resulting folders match torchvision ImageFolder labels used by
training/train.py.
"""
from __future__ import annotations
import argparse
import random
import shutil
from pathlib import Path

EXTENSIONS={".jpg",".jpeg",".png",".bmp",".webp"}

def images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)

def transfer(items, destination: Path, mode: str):
    destination.mkdir(parents=True,exist_ok=True)
    for src in items:
        target=destination/src.name
        # Avoid collisions if source contains nested folders with repeated names.
        if target.exists():
            target=destination/(src.parent.name+"_"+src.name)
        if mode=="link":
            try: target.hardlink_to(src)
            except OSError: shutil.copy2(src,target)
        else:
            shutil.copy2(src,target)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,help="Extracted CIFAKE root containing train/ and test/.")
    parser.add_argument("--output",default="data",help="VERIFAI output dataset directory.")
    parser.add_argument("--val-fraction",type=float,default=0.1)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--mode",choices=("copy","link"),default="copy",help="Hardlinks save disk space but require same filesystem.")
    args=parser.parse_args()
    if not 0.05<=args.val_fraction<=0.3:
        parser.error("--val-fraction must be between 0.05 and 0.30")
    src=Path(args.source).resolve()
    out=Path(args.output).resolve()
    mapping={"FAKE":"ai_generated","REAL":"real"}
    rng=random.Random(args.seed)
    summary={}
    for original,normalized in mapping.items():
        train_dir=src/"train"/original
        test_dir=src/"test"/original
        if not train_dir.is_dir() or not test_dir.is_dir():
            raise SystemExit(f"Missing expected folder: {train_dir} or {test_dir}")
        train_files=images(train_dir); test_files=images(test_dir)
        if len(train_files)<10 or not test_files:
            raise SystemExit(f"Insufficient images for {original}: train={len(train_files)}, test={len(test_files)}")
        rng.shuffle(train_files)
        val_count=max(1,round(len(train_files)*args.val_fraction))
        val_files=train_files[:val_count]
        fit_files=train_files[val_count:]
        transfer(fit_files,out/"train"/normalized,args.mode)
        transfer(val_files,out/"val"/normalized,args.mode)
        transfer(test_files,out/"test"/normalized,args.mode)
        summary[normalized]={"train":len(fit_files),"val":len(val_files),"test":len(test_files)}
    print("Prepared CIFAKE dataset:")
    for label,counts in summary.items():
        print(f"  {label}: "+", ".join(f"{k}={v:,}" for k,v in counts.items()))
    print(f"Output: {out}")
    print("Reminder: CIFAKE uses 32x32 CIFAR-10-derived real images and Stable Diffusion 1.4 synthetic images; it is a baseline, not proof of generalization to modern generators.")

if __name__=="__main__": main()
