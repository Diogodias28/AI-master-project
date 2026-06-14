#!/usr/bin/env python3
"""
scripts/eval_cnn.py — Avaliação e benchmark da CNN no test set
===============================================================
Calcula métricas: Top-1, Top-5, Per-class accuracy.
Gera gráficos e tabela comparativa com CLIP (se houver resultados).

Exemplo:
    python scripts/eval_cnn.py --checkpoint models/resnet50_best.pth --model resnet50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnn_dataset import CAT_NAMES, DeepFashionCategoryDataset, get_val_transforms
from cnn_trainer import build_resnet50, build_efficientnet

warnings.filterwarnings("ignore")


def evaluate(checkpoint_path: str, model_name: str = "resnet50", data_root: str = "deepfashion",
             batch_size: int = 64, num_workers: int = 4, output_dir: str = "models"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    test_ds = DeepFashionCategoryDataset(data_root, split="test", transform=get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # Modelo
    if model_name == "resnet50":
        model = build_resnet50(pretrained=False)
    else:
        model = build_efficientnet(pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    print(f"[Eval] A avaliar {model_name} em {len(test_ds)} amostras de teste…")

    all_labels = []
    all_preds = []
    all_top5 = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            outputs = model(images)
            _, pred_top1 = outputs.topk(1, 1, True, True)
            _, pred_top5 = outputs.topk(5, 1, True, True)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred_top1.squeeze().cpu().numpy())
            all_top5.extend(pred_top5.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_top5 = np.array(all_top5)

    # Métricas
    top1_acc = accuracy_score(all_labels, all_preds)
    top5_acc = np.mean([all_labels[i] in all_top5[i] for i in range(len(all_labels))])

    print(f"\n[Eval] Top-1 Accuracy: {top1_acc:.4f}")
    print(f"[Eval] Top-5 Accuracy: {top5_acc:.4f}")

    # Per-class accuracy
    class_correct = np.zeros(len(CAT_NAMES))
    class_total = np.zeros(len(CAT_NAMES))
    for label, pred in zip(all_labels, all_preds):
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1

    class_acc = class_correct / np.maximum(class_total, 1)
    per_class = {CAT_NAMES[i]: float(class_acc[i]) for i in range(len(CAT_NAMES))}

    # Grava resultados
    results = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "test_samples": int(len(all_labels)),
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "per_class_accuracy": per_class,
    }

    results_path = output_dir / f"{model_name}_test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Eval] Resultados gravados: {results_path}")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=CAT_NAMES, zero_division=0)
    report_path = output_dir / f"{model_name}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Eval] Report gravado: {report_path}")
    print("\n" + report)

    # Gráfico: per-class accuracy bar chart
    plt.figure(figsize=(14, 6))
    sorted_idx = np.argsort(class_acc)[::-1]
    colors = ["#4ade80" if class_acc[i] > 0.5 else "#fbbf24" if class_acc[i] > 0.3 else "#f87171" for i in sorted_idx]
    plt.bar(range(len(CAT_NAMES)), class_acc[sorted_idx], color=colors)
    plt.xticks(range(len(CAT_NAMES)), [CAT_NAMES[i] for i in sorted_idx], rotation=90, fontsize=8)
    plt.ylabel("Accuracy")
    plt.title(f"Per-Class Accuracy — {model_name} (Top-1: {top1_acc:.3f})")
    plt.tight_layout()
    chart_path = output_dir / f"{model_name}_per_class_accuracy.png"
    plt.savefig(chart_path, dpi=150)
    print(f"[Eval] Gráfico gravado: {chart_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho para o .pth do melhor modelo")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet"])
    parser.add_argument("--data", type=str, default="deepfashion")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--output", type=str, default="models")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.model, args.data, args.batch, output_dir=args.output)
