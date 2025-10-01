# -*- coding: utf-8 -*-
import os, math, json, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score

torch.backends.cudnn.benchmark = True

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_float_and_clip(arr: np.ndarray, lo_p=0.5, hi_p=99.5) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    small = arr[::4, ::4]
    lo, hi = np.percentile(small, lo_p), np.percentile(small, hi_p)
    if hi <= lo:
        m, s = arr.mean(), arr.std() + 1e-6
        arr = np.clip((arr - m) / (3 * s) + 0.5, 0, 1)
    else:
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return arr

class CSVDatasetBin(Dataset):
    def __init__(self, split_dir: Path, size=448, augment=False):
        self.size = size
        csv_path = Path(split_dir) / "labels.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        items = [ln.split(",")[:2] for ln in lines[1:]]
        self.paths  = [Path(split_dir) / p for p, _ in items]
        self.labels = np.array([int(y) for _, y in items], dtype=np.int64)

        self.size = size  # <-- ADD THIS LINE
        csv_path = Path(split_dir) / "labels.csv"
        
        if augment:
            self.tfm = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=(0.65, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5, fill=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.25]),
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize(self.size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.25]),
            ])

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        # 1) open grayscale
        img = Image.open(p).convert("L")
        # 2) resize first (small tensor -> cheap normalization)
        if isinstance(self.tfm.transforms[0], transforms.RandomResizedCrop) or isinstance(self.tfm.transforms[0], transforms.Resize):
            # make a small, deterministic resize for normalization step
            img_small = img.resize((self.size, self.size), Image.BICUBIC)
        else:
            img_small = img

        # 3) percentile normalize on the small image
        arr = np.array(img_small, dtype=np.uint8)
        small = arr[::4, ::4]
        lo, hi = np.percentile(small, 0.5), np.percentile(small, 99.5)
        if hi <= lo:
            m, s = arr.mean(), arr.std() + 1e-6
            arr = np.clip((arr - m) / (3*s) + 0.5, 0, 1)
        else:
            arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        img_norm = Image.fromarray((arr * 255).astype(np.uint8), "L")

        # 4) now run the augmentation/resize pipeline (already expects PIL L in [0,255])
        x = self.tfm(img_norm)
        y = int(self.labels[idx])
        return x, y


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha[targets] * (1-pt)**self.gamma * ce).mean()

def class_balanced_alpha(counts, beta=0.9999):
    n = torch.tensor(counts, dtype=torch.float32)
    eff = (1-beta)/(1-torch.pow(beta, n).clamp(min=1e-6))
    return eff/eff.sum()

def build_resnet50_binary(imagenet=True):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if imagenet else None)
    w = m.conv1.weight.data.clone()
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    with torch.no_grad(): m.conv1.weight[:] = w.mean(1, keepdim=True)
    m.fc = nn.Linear(2048, 2)
    return m

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, ps = [], []
    for xb, yb in loader:
        prob = model(xb.to(device)).softmax(1)
        ys.append(yb.numpy()); ps.append(prob.cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return accuracy_score(y, p.argmax(1)), f1_score(y, p.argmax(1), average="macro")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--imagenet", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--use_sampler", action="store_true")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--tag", default="bm_clean")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.data)
    out = Path("runs_baseline")/f"resnet50_{args.tag}_s{args.size}"
    out.mkdir(parents=True, exist_ok=True)

    train_ds = CSVDatasetBin(root/"train", args.size, True)
    val_ds   = CSVDatasetBin(root/"val", args.size, False)
    test_ds  = CSVDatasetBin(root/"test", args.size, False)

    c0, c1 = (train_ds.labels==0).sum(), (train_ds.labels==1).sum()
    alpha = class_balanced_alpha([c0, c1])

    if args.use_sampler:
        freq = np.array([c0,c1], float)
        weights = 1.0/freq
        samples_w = weights[train_ds.labels]
        sampler = WeightedRandomSampler(samples_w, len(samples_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_resnet50_binary(args.imagenet).to(device)
    loss_fn = FocalLoss(alpha, gamma=args.gamma).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_f1, best_path = -1, out/"best.pt"
    for epoch in range(args.epochs):
        model.train(); running=0.0
        pbar = tqdm(train_loader, ncols=100, desc=f"[epoch {epoch+1}/{args.epochs}]")
        opt.zero_grad(set_to_none=True)
        for i,(xb,yb) in enumerate(pbar,1):
            xb,yb = xb.to(device), yb.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = loss_fn(model(xb), yb)/max(1,args.accum)
            scaler.scale(loss).backward()
            if i%args.accum==0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            running+=loss.item()*xb.size(0)
        val_acc,val_f1 = evaluate(model,val_loader,device)
        print(f"  val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
        if val_f1>best_f1:
            best_f1=val_f1; torch.save(model.state_dict(),best_path)
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_acc,te_f1=evaluate(model,test_loader,device)
    print(f"[DONE] Test acc={te_acc:.4f} Macro-F1={te_f1:.4f}")

if __name__=="__main__": main()
