"""Evaluate one saved checkpoint on the untouched data/test split."""
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from backend.model import build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--batch-size",type=int,default=16)
    a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    ds=datasets.ImageFolder(Path(a.data)/"test",transform=tf)
    model=build_model(ckpt["model_name"])
    model.load_state_dict(ckpt["state_dict"]); model.to(device).eval()
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False)
    ys=[]; preds=[]; scores=[]
    with torch.inference_mode():
        for x,y in loader:
            logits=model(x.to(device)); prob=logits.softmax(1)[:,ds.class_to_idx["ai_generated"]]
            scores.extend(prob.cpu().tolist()); preds.extend(logits.argmax(1).cpu().tolist()); ys.extend(y.tolist())
    positive=ds.class_to_idx["ai_generated"]
    report=classification_report(ys,preds,target_names=ds.classes,output_dict=True,zero_division=0)
    result={"model":ckpt["model_name"],"classes":ds.classes,"classification_report":report,"confusion_matrix":confusion_matrix(ys,preds).tolist(),"roc_auc_ai_generated":roc_auc_score([int(y==positive) for y in ys],scores)}
    Path("reports").mkdir(exist_ok=True)
    out=Path("reports")/f"{ckpt['model_name']}_test_metrics.json"
    out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
