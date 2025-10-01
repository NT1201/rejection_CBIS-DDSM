# eval_resnet_standard.py
import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------- Model -----------------
def build_resnet(arch="resnet50", num_classes=2, imagenet=False):
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


# ----------------- Data ------------------
class CSVDataset(Dataset):
    def __init__(self, split_dir, size=224, imagenet=True):
        self.root = Path(split_dir)
        self.df = pd.read_csv(self.root / "labels.csv")
        self.df["label"] = self.df["label"].astype(str).str.strip()

        # normalize any text labels to ints
        mapping = {"NORMAL": "0", "BENIGN": "1", "MALIGNANT": "2", "ABNORMAL": "1"}
        self.df["label"] = self.df["label"].map(lambda x: mapping.get(x.upper(), x))
        self.df["label"] = self.df["label"].astype(int)

        mean = [0.485, 0.456, 0.406] if imagenet else [0.5, 0.5, 0.5]
        std  = [0.229, 0.224, 0.225] if imagenet else [0.5, 0.5, 0.5]
        self.t = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r["image"]).convert("L")
        x = self.t(img)
        y = int(r["label"])
        return x, y


def detect_num_classes_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    labs = df["label"].astype(str).str.strip().tolist()
    mapping = {"NORMAL": "0", "BENIGN": "1", "MALIGNANT": "2", "ABNORMAL": "1"}
    labs = [mapping.get(x.upper(), x) for x in labs]
    uniq = sorted(set(int(x) for x in labs))
    # compact labels to 0..K-1 if needed (not strictly necessary for eval)
    return len(uniq)


# ------------- Plot helpers --------------
def save_confusion(cm, class_names, out_png):
    fig = plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curves(y_true, y_prob, class_names, out_png, out_json):
    # y_prob shape: [N, C] softmax probabilities
    n_classes = y_prob.shape[1]
    result = {}

    # One-vs-rest ROC AUC per class (if >1 class)
    fig = plt.figure()
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve((y_true == c).astype(int), y_prob[:, c])
        auc = roc_auc_score((y_true == c).astype(int), y_prob[:, c])
        result[class_names[c]] = float(auc)
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc:.3f})")
    # micro/macro
    y_true_onehot = np.eye(n_classes)[y_true]
    micro_auc = roc_auc_score(y_true_onehot.ravel(), y_prob.ravel())
    macro_auc = np.mean(list(result.values())) if n_classes > 1 else list(result.values())[0]
    result["micro"] = float(micro_auc)
    result["macro"] = float(macro_auc)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves")
    plt.legend()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)


def save_pr_curves(y_true, y_prob, class_names, out_png, out_json):
    n_classes = y_prob.shape[1]
    result = {}

    fig = plt.figure()
    for c in range(n_classes):
        prec, rec, _ = precision_recall_curve((y_true == c).astype(int), y_prob[:, c])
        ap = average_precision_score((y_true == c).astype(int), y_prob[:, c])
        result[class_names[c]] = float(ap)
        plt.plot(rec, prec, label=f"{class_names[c]} (AP={ap:.3f})")

    # micro/macro AP
    y_true_onehot = np.eye(n_classes)[y_true]
    micro_ap = average_precision_score(y_true_onehot.ravel(), y_prob.ravel())
    macro_ap = np.mean(list(result.values())) if n_classes > 1 else list(result.values())[0]
    result["micro"] = float(micro_ap)
    result["macro"] = float(macro_ap)

    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curves")
    plt.legend()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)


# ----------------- Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root with train/val/test")
    ap.add_argument("--arch", choices=["resnet18","resnet50"], default="resnet50")
    ap.add_argument("--ckpt", required=True, help="path to best.pt")
    ap.add_argument("--num_classes", type=int, default=-1, help="-1 => auto-detect from test labels.csv")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, torch.cuda.get_device_name(0) if device.type=="cuda" else "")

    test_dir = Path(args.data) / "test"
    os.makedirs(args.out, exist_ok=True)

    # num classes
    if args.num_classes == -1:
        num_classes = detect_num_classes_from_csv(test_dir / "labels.csv")
    else:
        num_classes = args.num_classes

    # class names (best-effort)
    if num_classes == 2:
        class_names = ["BENIGN","MALIGNANT"]
    elif num_classes == 3:
        class_names = ["NORMAL","BENIGN","MALIGNANT"]
    else:
        class_names = [f"C{i}" for i in range(num_classes)]

    # data
    test_ds = CSVDataset(test_dir, size=args.size, imagenet=True)
    test_ld = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=(device.type=="cuda"))

    # model
    model = build_resnet(args.arch, num_classes=num_classes, imagenet=False).to(device)
    state = torch.load(args.ckpt, map_location=device)  # add weights_only=True if your torch supports it
    model.load_state_dict(state)
    model.eval()

    # inference
    all_logits, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_ld:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_targets.append(yb.clone())
    logits = torch.cat(all_logits, 0).numpy()
    y_true = torch.cat(all_targets, 0).numpy()

    # softmax probs
    y_prob = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_pred = y_prob.argmax(1)

    # metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # save plots + metrics
    save_confusion(cm, class_names, os.path.join(args.out, "confusion_matrix.png"))
    save_roc_curves(y_true, y_prob, class_names,
                    os.path.join(args.out, "roc_per_class.png"),
                    os.path.join(args.out, "auc_roc.json"))
    save_pr_curves(y_true, y_prob, class_names,
                   os.path.join(args.out, "pr_per_class.png"),
                   os.path.join(args.out, "ap_pr.json"))

    with open(os.path.join(args.out, "metrics_test.json"), "w") as f:
        json.dump({"accuracy": float(acc), "macro_f1": float(macro_f1)}, f, indent=2)
    with open(os.path.join(args.out, "report_test.json"), "w") as f:
        json.dump(rep, f, indent=2)

    # console summary
    print(f"[EVAL DONE] Saved to: {args.out}")
    print(f"Test accuracy: {acc:.4f}  Macro-F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
