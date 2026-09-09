"""
Motor de entrenamiento, evaluación y serialización de métricas para EuroSAT.
Soporta hardware agnóstico (CUDA/CPU), métricas de Scikit-Learn y guardado del mejor checkpoint.
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Entrena el modelo durante una época completa."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(dataloader, desc="Entrenando", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evalúa el modelo sin calcular gradientes y retorna métricas clave."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluando", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return {
        "loss": float(total_loss),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted)
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    checkpoint_path: str = "weights/best_model.pth",
    metrics_export_path: str = "reports/metrics.json"
) -> Dict[str, list]:
    """
    Ejecuta el ciclo de entrenamiento completo, guarda el mejor modelo según val_loss
    y exporta el historial de métricas en formato JSON.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo de cómputo: {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_val_loss = float("inf")
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1_macro": []
    }

    # Asegurar que existan directorios de destino
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_export_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        print(
            f"Época [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1-Macro: {val_metrics['f1_macro']:.4f}"
        )

        # Guardar checkpoint si mejora la pérdida de validación
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Checkpoint guardado con mejor Val Loss: {best_val_loss:.4f}")

    # Exportar métricas finales a JSON
    with open(metrics_export_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"Métricas exportadas exitosamente en: {metrics_export_path}")

    return history


if __name__ == "__main__":
    print("Módulo de entrenamiento cargado correctamente.")