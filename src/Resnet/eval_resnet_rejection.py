# eval_resnet_rejection.py
import os, json, argparse
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score

# ---------- dataset ----------
class CSVDataset(Dataset):
    def __init__(self, split_dir, size=224):
        self.root = split_dir
        self.df = pd.read_csv(os.path.join(split_dir, "labels.csv"))
        self.t = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        rel = self.df.iloc[i]["image"]; y = int(self.df.iloc[i]["label"])
        img = Image.open(os.path.join(self.root, rel)).convert("L")
        arr = np.array(img); rgb = Image.fromarray(np.stack([arr,arr,arr],2))
        return self.t(rgb), y

# ---------- build model (same as training) ----------
def build_resnet(arch="resnet18", num_classes=3):
    if arch == "resnet18":
        m = models.resnet18(weights=None)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
    else:
        raise ValueError("arch must be resnet18 or resnet50")
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

@torch.no_grad()
def forward_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for xb, yb in tqdm(loader, desc="Eval", ncols=80):
        xb = xb.to(device, non_blocking=True)
        p = torch.softmax(model(xb), dim=1).cpu().numpy()
        probs.append(p)
        labels.append(yb.numpy())
    return np.concatenate(probs,0), np.concatenate(labels,0)

def plot_curve(probs, labels, method, out_png):
    if method == "msp":
        score = probs.max(1)            # high=confident
        order = np.argsort(-score)
    elif method == "energy":
        logits = np.log(probs + 1e-12)
        energy = -np.log(np.exp(logits).sum(1) + 1e-12)  # low=confident
        order = np.argsort(energy)
    else:
        raise ValueError
    cov, accs, f1s = [], [], []
    for k in range(1, len(order)+1):
        idx = order[:k]
        pred = probs[idx].argmax(1)
        cov.append(k/len(order))
        accs.append(accuracy_score(labels[idx], pred))
        f1s.append(f1_score(labels[idx], pred, average="macro"))
    plt.figure()
    plt.plot(cov, accs, label="Accuracy")
    plt.plot(cov, f1s, label="Macro-F1")
    plt.xlabel("Coverage"); plt.ylabel("Score")
    plt.title(f"Coverage–Performance ({method.upper()})")
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18","resnet50"])
    ap.add_argument("--ckpt", required=True)  # path to runs_baseline/.../best.pt
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--tag", default="eval_rejection")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, torch.cuda.get_device_name(0) if device.type=="cuda" else "")

    val_ds  = CSVDataset(os.path.join(args.data, "val"),  size=args.size)
    test_ds = CSVDataset(os.path.join(args.data, "test"), size=args.size)
    pin = (device.type == "cuda")
    val_ld  = DataLoader(val_ds,  batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin)
    test_ld = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin)

    model = build_resnet(args.arch, 3).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    out_dir = os.path.join("runs_rejection", f"{args.arch}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    # VAL
    probs, labels = forward_probs(model, val_ld, device)
    np.savez(os.path.join(out_dir, "val_probs_labels.npz"), probs=probs, labels=labels)
    plot_curve(probs, labels, "msp",    os.path.join(out_dir, "curve_val_msp.png"))
    plot_curve(probs, labels, "energy", os.path.join(out_dir, "curve_val_energy.png"))

    # TEST
    probs, labels = forward_probs(model, test_ld, device)
    np.savez(os.path.join(out_dir, "test_probs_labels.npz"), probs=probs, labels=labels)
    plot_curve(probs, labels, "msp",    os.path.join(out_dir, "curve_test_msp.png"))
    plot_curve(probs, labels, "energy", os.path.join(out_dir, "curve_test_energy.png"))

    print(f"[DONE] Rejection eval saved to: {out_dir}")

if __name__ == "__main__":
    main()
