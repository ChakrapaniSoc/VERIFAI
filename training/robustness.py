"""Evaluate simple image transformations against a trained checkpoint.

Example:
python -m training.robustness --data data/test --checkpoint artifacts/efficientnet_b0_best.pt
"""
import argparse, json, io
from pathlib import Path
import torch
from PIL import Image, ImageFilter
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from backend.model import build_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/test")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--batch-size",type=int,default=16)
    a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
    model=build_model(ckpt["model_name"]); model.load_state_dict(ckpt["state_dict"]); model.to(device).eval()
    base=datasets.ImageFolder(a.data)
    norm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    variants={
      "original":lambda im: im,
      "jpeg_quality_55":lambda im: Image.open(io.BytesIO(_jpeg(im,55))).convert("RGB"),
      "resize_down_up":lambda im: im.resize((max(32,im.width//2),max(32,im.height//2))).resize(im.size),
      "gaussian_blur":lambda im: im.filter(ImageFilter.GaussianBlur(radius=0.7)),
      "center_crop":lambda im: im.crop((im.width*.08,im.height*.08,im.width*.92,im.height*.92)).resize(im.size),
    }
    results={}
    for name,fn in variants.items():
      ys=[];ps=[]
      for path,label in base.samples:
        with Image.open(path) as src: im=fn(src.convert("RGB"))
        x=norm(im).unsqueeze(0).to(device)
        with torch.inference_mode(): pred=int(model(x).argmax(1).item())
        ys.append(label); ps.append(pred)
      results[name]={"accuracy":accuracy_score(ys,ps),"macro_f1":f1_score(ys,ps,average="macro")}
    Path("reports").mkdir(exist_ok=True)
    out=Path("reports")/f"{ckpt['model_name']}_robustness.json"
    out.write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))

def _jpeg(image,quality):
    out=io.BytesIO(); image.save(out,format="JPEG",quality=quality); return out.getvalue()

if __name__=="__main__": main()
