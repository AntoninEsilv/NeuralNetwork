"""
Transfer learning DenseNet121 (pré-entraîné ImageNet) sur Fashion-MNIST.

Données : dataset_split/train.csv et test.csv (générés par Separation.py, 70/30 stratifié,
seed 42), avec 20 % du train en validation comme le CNN de network.py
-> 5600 train / 1400 val / 3000 test, même jeu de test que le CNN.

Étapes :
  1. Préparation des données : CSV -> images 28x28 -> redimensionnées, 3 canaux,
     normalisées comme ImageNet (fait à la volée sur le GPU, batch par batch).
  2. Chargement de DenseNet121 pré-entraîné + remplacement de la tête (1000 -> 10 classes).
  3. Phase 1 (feature extraction) : backbone gelé, seule la tête est entraînée.
  4. Phase 2 (fine-tuning) : tout le réseau est dégelé, learning rate faible.
  5. Évaluation sur le jeu de test : accuracy, rapport par classe, matrice de confusion.

Usage :
  python densenet_transfer.py              # entraînement complet
  python densenet_transfer.py --quick      # test rapide sur un petit sous-ensemble
"""

import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import DenseNet121_Weights, densenet121

# Avertissement sans conséquence : GradScaler saute parfois le 1er pas (calibrage fp16)
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()`")

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "dataset_split"  # généré par Separation.py
OUT_MODELS = ROOT / "models"
OUT_FIGURES = ROOT / "figures"

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# 1. Données
# ---------------------------------------------------------------------------
def load_data(seed: int, quick: bool):
    """Charge le découpage commun du groupe et renvoie des tenseurs uint8 (N, 1, 28, 28).

    - dataset_split/train.csv (7000) et test.csv (3000) sont produits par Separation.py
      (70/30 stratifié, seed 42) : mêmes images de test que network.py.
    - Validation = les 20 % DERNIÈRES lignes de train.csv, exactement comme
      `validation_split=0.20` de Keras dans network.py -> 5600 train / 1400 val.
    On garde les images en 28x28 uint8 (≈ 8 Mo) : le redimensionnement est fait
    à la volée, sinon 10 000 images 3x224x224 en float32 occuperaient ~6 Go.
    """
    def read(path):
        df = pd.read_csv(path)
        y = df["label"].to_numpy()
        X = df.drop(columns="label").to_numpy(dtype=np.uint8).reshape(-1, 1, 28, 28)
        return X, y

    X_tr, y_tr = read(SPLIT_DIR / "train.csv")
    X_test, y_test = read(SPLIT_DIR / "test.csv")
    n_val = int(len(X_tr) * 0.20)
    X_train, y_train = X_tr[:-n_val], y_tr[:-n_val]
    X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]

    if quick:  # petit sous-ensemble pour vérifier que tout tourne
        X_train, y_train = X_train[:700], y_train[:700]
        X_val, y_val = X_val[:100], y_val[:100]
        X_test, y_test = X_test[:200], y_test[:200]

    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    return {
        "train": (to_t(X_train), to_t(y_train).long()),
        "val": (to_t(X_val), to_t(y_val).long()),
        "test": (to_t(X_test), to_t(y_test).long()),
    }


def preprocess(x_uint8: torch.Tensor, img_size: int, augment: bool) -> torch.Tensor:
    """(B,1,28,28) uint8 -> (B,3,img_size,img_size) float normalisé ImageNet."""
    x = x_uint8.float() / 255.0
    if augment:
        # retournement horizontal aléatoire (un vêtement retourné reste le même vêtement)
        flip = torch.rand(x.size(0), device=x.device) < 0.5
        x[flip] = x[flip].flip(-1)
        # petite translation aléatoire (±2 pixels) via padding + crop
        x = F.pad(x, (2, 2, 2, 2))
        i, j = np.random.randint(0, 5, size=2)
        x = x[:, :, i:i + 28, j:j + 28]
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)  # gris -> 3 canaux identiques
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def iterate(X, y, batch_size, shuffle):
    idx = torch.randperm(len(X), device=X.device) if shuffle else torch.arange(len(X), device=X.device)
    for k in range(0, len(X), batch_size):
        b = idx[k:k + batch_size]
        yield X[b], y[b]


# ---------------------------------------------------------------------------
# 2. Modèle
# ---------------------------------------------------------------------------
def build_model(num_classes: int = 10) -> nn.Module:
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)  # 1024 -> 10
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for p in model.features.parameters():
        p.requires_grad = trainable


# ---------------------------------------------------------------------------
# 3-4. Entraînement
# ---------------------------------------------------------------------------
def run_epoch(model, data, args, device, optimizer=None, scaler=None, scheduler=None,
              freeze_bn=False):
    train = optimizer is not None
    model.train(train)
    if train and freeze_bn:
        model.features.eval()  # backbone gelé : on ne met pas à jour les stats BatchNorm
    X, y = data
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in iterate(X, y, args.batch_size, shuffle=train):
            xb = preprocess(xb, args.img_size, augment=train)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=0.1 if train else 0.0)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def train_phase(name, model, splits, args, device, epochs, lr, freeze_backbone, history, best):
    set_backbone_trainable(model, not freeze_backbone)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(splits["train"][0]) + args.batch_size - 1) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    n_trainable = sum(p.numel() for p in params)
    print(f"\n=== {name} : {epochs} époque(s), lr={lr}, paramètres entraînables={n_trainable:,} ===")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, splits["train"], args, device, optimizer, scaler, scheduler,
                                    freeze_bn=freeze_backbone)
        va_loss, va_acc = run_epoch(model, splits["val"], args, device)
        history.append({"phase": name, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        flag = ""
        if va_acc > best["acc"]:
            best["acc"] = va_acc
            best["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = "  <- meilleur"
        print(f"[{name}] époque {ep}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.0f}s{flag}")


# ---------------------------------------------------------------------------
# 5. Évaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, data, args, device):
    model.eval()
    X, y = data
    preds = []
    for xb, _ in iterate(X, y, args.batch_size, shuffle=False):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            preds.append(model(preprocess(xb, args.img_size, augment=False)).argmax(1).cpu())
    return torch.cat(preds).numpy(), y.cpu().numpy()


def save_figures(history, cm, suffix):
    OUT_FIGURES.mkdir(exist_ok=True)
    h = pd.DataFrame(history)
    ep = np.arange(1, len(h) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "acc"]):
        ax.plot(ep, h[f"train_{metric}"], "o-", label="train")
        ax.plot(ep, h[f"val_{metric}"], "o-", label="validation")
        n_head = (h["phase"] == "Phase 1").sum()
        ax.axvline(n_head + 0.5, color="gray", ls="--", lw=1)
        ax.set_title(f"{metric} (pointillés = début du fine-tuning)")
        ax.set_xlabel("époque")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / f"densenet_courbes{suffix}.png", dpi=120)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("prédit")
    ax.set_ylabel("réel")
    ax.set_title("Matrice de confusion (test)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / f"densenet_confusion{suffix}.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-ft", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="petit sous-ensemble, 1 époque par phase")
    args = ap.parse_args()
    if args.quick:
        args.epochs_head, args.epochs_ft = 1, 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
                                  "  /!\\ pas de GPU : ce sera très lent"))
    torch.backends.cudnn.benchmark = True

    splits = load_data(args.seed, args.quick)
    splits = {k: (X.to(device), y.to(device)) for k, (X, y) in splits.items()}
    print("Tailles : " + ", ".join(f"{k}={len(v[0])}" for k, v in splits.items())
          + f" | images redimensionnées en {args.img_size}x{args.img_size}x3")

    # NB : pas de channels_last ici, il rend DenseNet ~16x plus lent sur RTX 2060 (Turing)
    model = build_model().to(device)
    history, best = [], {"acc": -1.0, "state": None}

    train_phase("Phase 1", model, splits, args, device, args.epochs_head, args.lr_head,
                freeze_backbone=True, history=history, best=best)
    train_phase("Phase 2", model, splits, args, device, args.epochs_ft, args.lr_ft,
                freeze_backbone=False, history=history, best=best)

    model.load_state_dict(best["state"])
    y_pred, y_true = predict(model, splits["test"], args, device)
    acc = (y_pred == y_true).mean()
    print(f"\n=== Test (meilleur modèle sur la validation, val acc={best['acc']:.4f}) ===")
    print(f"Accuracy test : {acc:.4f}\n")
    print(classification_report(y_true, y_pred, labels=range(10), target_names=CLASSES, digits=4,
                                zero_division=0))

    suffix = "_quick" if args.quick else ""
    save_figures(history, confusion_matrix(y_true, y_pred, labels=range(10)), suffix)
    OUT_MODELS.mkdir(exist_ok=True)
    torch.save({"state_dict": best["state"], "img_size": args.img_size, "classes": CLASSES,
                "test_acc": float(acc)}, OUT_MODELS / f"densenet121_fmnist{suffix}.pt")
    print(f"Figures -> {OUT_FIGURES}\nModèle -> {OUT_MODELS / f'densenet121_fmnist{suffix}.pt'}")


if __name__ == "__main__":
    main()
