# train_resnet.py
import os, json, argparse
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

torch.backends.cudnn.benchmark = True

# ----------------- CSV dataset -----------------
class CSVDataset(Dataset):
    def __init__(self, split_dir, size=224, augment=False):
        self.root = split_dir
        self.df = pd.read_csv(os.path.join(split_dir, "labels.csv"))
        self.size = size
        self.augment = augment
        self.t_train = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
        ])
        self.t_eval = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        rel = self.df.iloc[i]["image"]
        lab = int(self.df.iloc[i]["label"])
        img = Image.open(os.path.join(self.root, rel)).convert("L")
        arr = np.array(img)
        rgb = Image.fromarray(np.stack([arr, arr, arr], axis=2))  # 1→3ch
        x = (self.t_train if self.augment else self.t_eval)(rgb)
        return x, lab

# ----------------- build model -----------------
def build_resnet(arch="resnet18", num_classes=3, imagenet=False):
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if imagenet else None
        m = models.resnet18(weights=weights)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if imagenet else None
        m = models.resnet50(weights=weights)
    else:
        raise ValueError("arch must be resnet18 or resnet50")
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ----------------- evaluation -----------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        ys.extend(yb.numpy().tolist())
        preds.extend(pred.tolist())
    ys = np.array(ys); preds = np.array(preds)
    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds, average="macro")
    cm = confusion_matrix(ys, preds).tolist()
    rep = classification_report(ys, preds, output_dict=True)
    return {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm, "report": rep}

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)       # path to cbis_cls3
    ap.add_argument("--arch", default="resnet18", choices=["resnet18","resnet50"])
    ap.add_argument("--weights", default="imagenet", choices=["none","imagenet"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=0)  # Windows → keep small
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="resnet_baseline")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, torch.cuda.get_device_name(0) if device.type=="cuda" else "")

    # datasets/loaders
    train_ds = CSVDataset(os.path.join(args.data, "train"), size=args.size, augment=True)
    val_ds   = CSVDataset(os.path.join(args.data, "val"),   size=args.size, augment=False)
    test_ds  = CSVDataset(os.path.join(args.data, "test"),  size=args.size, augment=False)

    pin = (device.type == "cuda")
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.workers, pin_memory=pin)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin)

    # model/opt
    model = build_resnet(args.arch, 3, imagenet=(args.weights=="imagenet")).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type=="cuda"))

    out_dir = os.path.join("runs_baseline", f"{args.arch}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    # train
    best_val = -1.0
    for ep in range(1, args.epochs+1):
        model.train()
        run_loss, n = 0.0, 0
        pbar = tqdm(train_ld, desc=f"[epoch {ep}/{args.epochs}]", ncols=80)
        for xb, yb in pbar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss = crit(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                logits = model(xb); loss = crit(logits, yb)
                loss.backward(); opt.step()
            run_loss += loss.item()*xb.size(0); n += xb.size(0)
            pbar.set_postfix(loss=f"{run_loss/max(n,1):.4f}")
        # val
        val_metrics = evaluate(model, val_ld, device)
        print(f"  val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['macro_f1']:.4f}")
        # save best
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            with open(os.path.join(out_dir, "metrics_val.json"), "w") as f:
                json.dump(val_metrics, f, indent=2)

    # final test with best
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    test_metrics = evaluate(model, test_ld, device)
    with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # tiny README of the run
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[DONE] Baseline saved to: {out_dir}")
    print("Test accuracy:", f"{test_metrics['accuracy']:.4f}", "Macro-F1:", f"{test_metrics['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
