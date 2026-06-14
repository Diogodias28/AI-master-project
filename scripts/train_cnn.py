#!/usr/bin/env python3
"""
scripts/train_cnn.py — Entrypoint para treinar a CNN
====================================================
Exemplo:
    python scripts/train_cnn.py --model resnet50 --epochs 20 --batch 64 --lr 1e-3
"""
import sys
from pathlib import Path

# Adiciona src/ ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnn_trainer import train_model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Treina CNN (ResNet50/EfficientNet) no DeepFashion")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet"],
                        help="Arquitetura CNN a usar")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data", type=str, default="deepfashion", help="Caminho para o dataset")
    parser.add_argument("--output", type=str, default="models", help="Diretório para checkpoints")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--freeze", type=int, default=3, help="Épocas de feature-extraction (backbone congelado)")
    parser.add_argument("--workers", type=int, default=4, help="Número de workers do DataLoader")
    args = parser.parse_args()

    print("=" * 60)
    print("  Treino CNN — DeepFashion Category Classification")
    print("=" * 60)

    model, history, best_acc = train_model(
        data_root=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        output_dir=args.output,
        patience=args.patience,
        freeze_epochs=args.freeze,
        num_workers=args.workers,
    )

    print("\n" + "=" * 60)
    print("  Treino Concluído")
    print("=" * 60)
    print(f"  Melhor val_acc: {best_acc:.4f}")
    print(f"  Checkpoints:   {args.output}/")
    print(f"  Histórico:     {args.output}/{args.model}_history.json")


if __name__ == "__main__":
    main()
