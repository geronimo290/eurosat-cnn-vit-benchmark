"""
Script de entrada principal (CLI) para entrenar y evaluar modelos en EuroSAT.

Uso:
    python train.py --model cnn --epochs 5 --batch-size 64
    python train.py --model hybrid --epochs 5 --batch-size 64
"""

import argparse
import json
import sys
from pathlib import Path
import torch

from src.data.dataset import get_dataloaders
from src.models.cnn import build_cnn_model
from src.models.hybrid import HybridCNNViT
from src.training.trainer import fit, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento y evaluación en EuroSAT")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "hybrid"],
        help="Arquitectura a entrenar: 'cnn' (EfficientNet-B0) o 'hybrid' (CNN-ViT)"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamaño del lote (batch size)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Tasa de aprendizaje (learning rate)")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Ruta a los datos de EuroSAT")
    parser.add_argument("--num-workers", type=int, default=0, help="Subprocesos de carga de datos")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de reproducibilidad")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model.upper()

    print(f"\n{'='*50}")
    print(f" Iniciando Pipeline de Entrenamiento: Modelo {model_name}")
    print(f" Épocas: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"{'='*50}\n")

    # 1. Cargar DataLoaders
    print("Cargando dataset EuroSAT (Train 70%, Val 15%, Test 15%)...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    num_classes = len(class_names)
    print(f"Clases detectadas ({num_classes}): {class_names}")

    # 2. Instanciar arquitectura
    if args.model == "cnn":
        print("Construyendo Baseline CNN (EfficientNet-B0 Transfer Learning)...")
        model = build_cnn_model(num_classes=num_classes, freeze_backbone=True)
    elif args.model == "hybrid":
        print("Construyendo Modelo Híbrido (EfficientNet-B0 + Vision Transformer)...")
        model = HybridCNNViT(num_classes=num_classes, embed_dim=128, num_heads=4, num_transformer_layers=2)

    # 3. Rutas de persistencia
    checkpoint_path = f"weights/best_{args.model}.pth"
    metrics_path = f"reports/metrics_{args.model}.json"
    test_report_path = f"reports/test_report_{args.model}.json"

    # 4. Entrenar y evaluar en Validación
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=checkpoint_path,
        metrics_export_path=metrics_path
    )

    # 5. Evaluación Final en Conjunto de Test (Holdout intocable)
    print(f"\n{'='*50}")
    print(" Ejecutando Evaluación Final sobre el Conjunto de TEST")
    print(f"{'='*50}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, device)

    loss_val = test_metrics['loss']
    acc_val = test_metrics['accuracy'] * 100
    f1_m = test_metrics['f1_macro']
    f1_w = test_metrics['f1_weighted']

    print(f"\nResultados Finales de Test ({model_name}):")
    print(f"  - Loss: {loss_val:.4f}")
    print(f"  - Accuracy: {acc_val:.2f}%")
    print(f"  - F1-Score (Macro): {f1_m:.4f}")
    print(f"  - F1-Score (Weighted): {f1_w:.4f}")

    # Guardar reporte final de test
    with open(test_report_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Reporte de test guardado en: {test_report_path}\n")


if __name__ == "__main__":
    main()
