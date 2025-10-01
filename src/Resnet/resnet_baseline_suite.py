# resnet_baseline_suite.py
import os, json, argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# ----- CSV dataset (your structure) -----
class CSVDataset(Dataset):
    def __init__(self, root_split_dir, size=224, augment=False):
        self.root = root_split_dir
        self.df = pd.read_csv(os.path.join(root_split_dir, "labels.csv"))
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel = self.df.iloc[idx]["image"]           # e.g. "images/train_000123.jpg"
        label = int(self.df.iloc[idx]["label"])
        img = Image.open(os.path.join(self.root, rel)).convert("L")
        arr = np.array(img)
        rgb = Image.fromarray(np.stack([arr, arr, arr], axis=2))  # 1→3 channels
        x = (self.t_train if self.augment else self.t_eval)(rgb)
        return x, label

# ----- model -----
def build_resnet18(num_classes=3, imagenet=False):
    if imagenet:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ----- eval & plots -----
@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    probs_list, labels_list = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(probs)
        labels_list.append(yb.numpy())
    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    preds = probs.argmax(1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return probs, labels, acc, macro_f1

def plot_coverage_curve(probs, labels, method, out_png):
    if method == "msp":
        score = probs.max(1)               # higher = more confident
        order = np.argsort(-score)
        keep_mask = [np.arange(len(score))[:k] for k in range(1, len(score)+1)]
    elif method == "energy":
        # energy from probs (proxy): lower = more confident
        logits = np.log(probs + 1e-12)
        score = -np.log(np.exp(logits).sum(1) + 1e-12)
        order = np.argsort(score)          # lower first
        keep_mask = [order[:k] for k in range(1, len(score)+1)]
    else:
        raise ValueError("method must be 'msp' or 'energy'")

    cov, accs, f1s = [], [], []
    y = labels
    for idxs in keep_mask:
        p = probs[idxs]
        yk = y[idxs]
        pred = p.argmax(1)
        cov.append(len(idxs)/len(y))
        accs.append(accuracy_score(yk, pred))
        f1s.append(f1_score(yk, pred, average="macro"))
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
    ap.add_argument("--data", required=True)  # path to cbis_cls3
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--weights", type=str, default="none", choices=["none","imagenet"])
    ap.add_argument("--tag", type=str, default="resnet18_suite")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CSVDataset(os.path.join(args.data, "train"), size=args.size, augment=True)
    val_ds   = CSVDataset(os.path.join(args.data, "val"),   size=args.size, augment=False)
    test_ds  = CSVDataset(os.path.join(args.data, "test"),  size=args.size, augment=False)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_resnet18(3, imagenet=(args.weights=="imagenet")).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ----- train -----
    for ep in range(1, args.epochs+1):
        model.train()
        run_loss, n = 0.0, 0
        for xb, yb in tqdm(train_ld, desc=f"[epoch {ep}/{args.epochs}]"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item()*xb.size(0); n += xb.size(0)
        print(f"  train_loss={run_loss/max(n,1):.4f}")

    # ----- evaluate + curves -----
    out_dir = os.path.join("runs_resnet_baseline", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    for split, ld in [("val", val_ld), ("test", test_ld)]:
        probs, labels, acc, f1 = eval_model(model, ld, device)
        with open(os.path.join(out_dir, f"metrics_{split}.json"), "w") as f:
            json.dump({"accuracy": acc, "macro_f1": f1}, f, indent=2)
        np.savez(os.path.join(out_dir, f"probs_labels_{split}.npz"), probs=probs, labels=labels)

        plot_coverage_curve(probs, labels, "msp",    os.path.join(out_dir, f"curve_{split}_msp.png"))
        plot_coverage_curve(probs, labels, "energy", os.path.join(out_dir, f"curve_{split}_energy.png"))

    print(f"[DONE] baseline+MSP+Energy → {out_dir}")

if __name__ == "__main__":
    main()
