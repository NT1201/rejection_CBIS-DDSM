# train_resnet_strong.py
# Robust CBIS-DDSM baseline trainer (2-class or 3-class).
# - Auto-detect num_classes from labels.csv unless --num_classes is provided
# - Memory-safe grayscale->RGB conversion (no large numpy stacks)
# - Class-weighted CE / Focal / CE+LabelSmoothing
# - WeightedRandomSampler option
# - Warmup + Cosine LR schedule, AMP, early-stop on macro-F1
# - Deterministic seeding; Windows-safe (workers=0 by default)

import os, json, argparse, math, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report

torch.backends.cudnn.benchmark = True


# ---------------- utils ----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # tensor of per-class weights or None
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')
    def forward(self, logits, target):
        ce = self.ce(logits, target)  # [B]
        with torch.no_grad():
            pt = torch.softmax(logits, dim=1).gather(1, target.view(-1,1)).squeeze(1).clamp_(1e-7, 1-1e-7)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss

def build_resnet(arch="resnet50", num_classes=3, imagenet=True):
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

def _read_labels(csv_path):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise RuntimeError(f"'label' column not found in {csv_path}")
    # ensure int labels
    df["label"] = df["label"].astype(str).str.strip()
    mapping = {"NORMAL":0, "BENIGN":1, "MALIGNANT":2, "ABNORMAL":1}
    def parse(v):
        if v.isdigit(): return int(v)
        return mapping.get(v.upper(), None)
    df["label"] = df["label"].map(parse)
    if df["label"].isna().any():
        bad = df[df["label"].isna()]
        raise RuntimeError(f"Unrecognized labels in {csv_path} rows:\n{bad.head()}")
    return df

def detect_num_classes(train_csv):
    df = _read_labels(train_csv)
    uniq = sorted(df["label"].unique().tolist())
    # ensure labels form 0..K-1
    remap = {old:i for i, old in enumerate(uniq)}
    df["label"] = df["label"].map(remap)
    return len(uniq), df  # K, remapped df

def compute_class_weights_from_df(df, num_classes):
    counts = df["label"].value_counts().reindex(range(num_classes), fill_value=0).values
    inv = 1.0 / np.maximum(counts, 1)
    w = inv / inv.sum() * num_classes
    return counts.tolist(), torch.tensor(w, dtype=torch.float32)

def make_sampler_from_df(df, num_classes):
    counts = df["label"].value_counts().reindex(range(num_classes), fill_value=0).values
    class_w = 1.0 / np.maximum(counts, 1)
    sample_w = df["label"].map(lambda y: class_w[int(y)]).values
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_w),
        num_samples=len(sample_w),
        replacement=True
    )

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb).argmax(1).cpu().numpy()
        ys.extend(yb.numpy().tolist()); ps.extend(pred.tolist())
    acc = accuracy_score(ys, ps)
    f1  = f1_score(ys, ps, average="macro", zero_division=0)
    rep = classification_report(ys, ps, output_dict=True, zero_division=0)
    return {"accuracy": float(acc), "macro_f1": float(f1), "report": rep}


# -------------- dataset --------------
class CSVDataset(Dataset):
    def __init__(self, split_dir, size=224, augment=False, imagenet=True):
        self.root = Path(split_dir)
        self.df = _read_labels(self.root / "labels.csv")
        self.df = self.df.reset_index(drop=True)

        mean = [0.485, 0.456, 0.406] if imagenet else [0.5, 0.5, 0.5]
        std  = [0.229, 0.224, 0.225] if imagenet else [0.5, 0.5, 0.5]
        norm = transforms.Normalize(mean=mean, std=std)

        # Convert to 3ch AFTER resize/crop (no big numpy stacks)
        aug = [
            transforms.Resize(int(size*1.15)),
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            norm,
        ]
        evalt = [
            transforms.Resize((size, size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            norm,
        ]
        self.t = transforms.Compose(aug if augment else evalt)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img_path = self.root / r["image"]
        img = Image.open(img_path).convert("L")
        x = self.t(img)
        y = int(r["label"])
        return x, y


# -------------- training --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)            # root with train/val/test
    ap.add_argument("--arch", choices=["resnet18","resnet50"], default="resnet50")
    ap.add_argument("--num_classes", type=int, default=-1, help="-1 = auto-detect from labels.csv")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=0)   # Windows safer
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--imagenet", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--use_sampler", action="store_true")
    ap.add_argument("--loss", choices=["ce","focal","ce_ls"], default="ce")
    ap.add_argument("--ls", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, torch.cuda.get_device_name(0) if device.type=="cuda" else "")

    train_dir = Path(args.data) / "train"
    val_dir   = Path(args.data) / "val"
    test_dir  = Path(args.data) / "test"

    # Auto-detect num_classes from train labels if needed
    if args.num_classes == -1:
        auto_K, train_df_for_stats = detect_num_classes(train_dir / "labels.csv")
        num_classes = auto_K
        train_df_stats = train_df_for_stats
    else:
        num_classes = args.num_classes
        # remake df aligned to 0..K-1 (in case labels are already 0/1)
        train_df_stats = _read_labels(train_dir / "labels.csv")
        if sorted(train_df_stats["label"].unique()) != list(range(num_classes)):
            # remap compactly 0..K-1
            uniq = sorted(train_df_stats["label"].unique().tolist())
            remap = {old:i for i, old in enumerate(uniq)}
            train_df_stats["label"] = train_df_stats["label"].map(remap)

    # datasets & loaders
    train_ds = CSVDataset(train_dir, size=args.size, augment=True,  imagenet=args.imagenet)
    val_ds   = CSVDataset(val_dir,   size=args.size, augment=False, imagenet=args.imagenet)
    test_ds  = CSVDataset(test_dir,  size=args.size, augment=False, imagenet=args.imagenet)

    pin = (device.type == "cuda")
    if args.use_sampler:
        sampler = make_sampler_from_df(train_df_stats, num_classes)
        train_ld = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=args.workers, pin_memory=pin)
    else:
        train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=pin)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=pin)

    # model
    model = build_resnet(args.arch, num_classes, imagenet=args.imagenet).to(device)

    # loss (class weights from TRAIN)
    counts, class_w = compute_class_weights_from_df(train_df_stats, num_classes)
    print("Train class counts:", counts, " -> class weights:", class_w.tolist())
    class_w = class_w.to(device)
    if args.loss == "focal":
        criterion = FocalLoss(alpha=class_w, gamma=2.0)
    elif args.loss == "ce_ls":
        criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=args.ls)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_w)

    # optimizer + warmup-cosine
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(len(train_ld), 1) * args.epochs
    warmup = max(int(0.05 * total_steps), 100)

    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / float(max(total_steps - warmup, 1))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type=="cuda"))

    out_dir = Path("runs_baseline") / f"{args.arch}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    bad = 0
    step = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        pbar = tqdm(train_ld, desc=f"[epoch {ep}/{args.epochs}]", ncols=100)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(args.amp and device.type=="cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            run_loss += loss.item() * xb.size(0); n += xb.size(0)
            pbar.set_postfix(loss=f"{run_loss/max(n,1):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # validation
        val_metrics = evaluate(model, val_ld, device)
        print(f"  val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]; bad = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
            with open(out_dir / "metrics_val.json", "w") as f:
                json.dump(val_metrics, f, indent=2)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EARLY STOP] no val F1 improvement for {args.patience} epochs.")
                break

    # test with best
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_ld, device)
    with open(out_dir / "metrics_test.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(out_dir / "run_config.json", "w") as f:
        cfg = vars(args).copy(); cfg["resolved_num_classes"] = num_classes
        json.dump(cfg, f, indent=2)
    print(f"[DONE] Saved to: {out_dir}")
    print("Test accuracy:", f"{test_metrics['accuracy']:.4f}", "Macro-F1:", f"{test_metrics['macro_f1']:.4f}")
    

if __name__ == "__main__":
    main()
