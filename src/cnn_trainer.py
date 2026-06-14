"""
cnn_trainer.py — Treino de ResNet50 com Transfer Learning para 50 categorias DeepFashion
=========================================================================================
Arquitetura: ResNet50 (pre-treinado ImageNet) → FC layer substituída para 50 classes.
O treino pode ser em 2 modos:
  • feature-extraction (congela backbone, treina só a FC)
  • fine-tuning (treina tudo com lr baixo)
"""
import json
import os
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from cnn_dataset import DeepFashionCategoryDataset, get_train_transforms, get_val_transforms, NUM_CLASSES

warnings.filterwarnings("ignore")

# ── Model Builders ──
def build_resnet50(num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = False):
    """Constroi ResNet50 com FC layer adaptada para num_classes."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Substitui a última FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet(num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = False):
    """Constroi EfficientNet-B0 (alternativa mais leve)."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ── Treino por época ──
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ── Validação ──
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct += pred[0].eq(labels).sum().item()
            top5_correct += pred.eq(labels.view(1, -1).expand_as(pred)).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_top5 = top5_correct / total
    return epoch_loss, epoch_acc, epoch_top5


# ── Treino completo ──
def train_model(
    data_root: str,
    model_name: str = "resnet50",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = None,
    output_dir: str = "models",
    patience: int = 5,
    freeze_epochs: int = 3,
    num_workers: int = 4,
):
    """
    Treina um modelo CNN com transfer learning.
    Fase 1: feature-extraction (freeze backbone, epochs=freeze_epochs)
    Fase 2: fine-tuning (descongela tudo, lr reduzido)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Trainer] Device: {device}")
    print(f"[Trainer] Model: {model_name}, Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")

    # Datasets
    train_ds = DeepFashionCategoryDataset(data_root, split="train", transform=get_train_transforms())
    val_ds = DeepFashionCategoryDataset(data_root, split="val", transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Modelo
    if model_name == "resnet50":
        model = build_resnet50(pretrained=True, freeze_backbone=True)
    elif model_name == "efficientnet":
        model = build_efficientnet(pretrained=True, freeze_backbone=True)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_acc = 0.0
    history = []
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # Transição: descongela backbone após freeze_epochs
        if epoch == freeze_epochs + 1:
            print(f"[Trainer] Epoch {epoch}: a descongelar backbone para fine-tuning…")
            for param in model.parameters():
                param.requires_grad = True
            # Re-cria optimizer para incluir todos os params
            optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_top5 = validate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top5={val_top5:.4f} | "
              f"{elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_top5": val_top5,
        })

        # Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            ckpt_path = output_dir / f"{model_name}_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_top5": val_top5,
                "model_name": model_name,
            }, ckpt_path)
            print(f"[Trainer] Novo melhor modelo gravado: {ckpt_path} (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Trainer] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Grava histórico
    hist_path = output_dir / f"{model_name}_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[Trainer] Histórico gravado: {hist_path}")

    return model, history, best_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", type=str, default="deepfashion")
    parser.add_argument("--output", type=str, default="models")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--freeze", type=int, default=3)
    args = parser.parse_args()

    train_model(
        data_root=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        output_dir=args.output,
        patience=args.patience,
        freeze_epochs=args.freeze,
    )
