"""
Script de comparación automática de métricas para EuroSAT.
Lee los reportes JSON reales y genera un gráfico comparativo.
"""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

reports_dir = Path("reports")
cnn_report_path = reports_dir / "test_report_cnn.json"
hybrid_report_path = reports_dir / "test_report_hybrid.json"

data = []
if cnn_report_path.exists():
    with open(cnn_report_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        data.append({
            "Modelo": "EfficientNet-B0 (CNN)",
            "Accuracy": metrics["accuracy"] * 100,
            "F1-Macro": metrics["f1_macro"] * 100,
            "F1-Weighted": metrics["f1_weighted"] * 100,
            "Loss": metrics["loss"]
        })

if hybrid_report_path.exists():
    with open(hybrid_report_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        data.append({
            "Modelo": "Hybrid CNN-ViT",
            "Accuracy": metrics["accuracy"] * 100,
            "F1-Macro": metrics["f1_macro"] * 100,
            "F1-Weighted": metrics["f1_weighted"] * 100,
            "Loss": metrics["loss"]
        })

if not data:
    print("No se encontraron reportes en reports/. Asegúrate de correr train.py primero.")
else:
    df = pd.DataFrame(data)
    print("\n" + "="*55)
    print(" RESUMEN COMPARATIVO DE TEST (HOLDOUT 15% - 4.050 IMÁGENES)")
    print("="*55)
    print(df.to_string(index=False))

    df_plot = df.melt(id_vars=["Modelo"], value_vars=["Accuracy", "F1-Macro", "F1-Weighted"],
                      var_name="Métrica", value_name="Porcentaje")

    plt.figure(figsize=(9, 5))
    models = df["Modelo"].unique()
    num_models = len(models)
    width = 0.35

    for i, model_name in enumerate(models):
        subset = df_plot[df_plot["Modelo"] == model_name]
        positions = [x + (i - (num_models - 1) / 2) * width for x in range(len(subset))]
        plt.bar(positions, subset["Porcentaje"], width=width, label=model_name)

    plt.xticks(range(3), ["Accuracy (%)", "F1-Macro (%)", "F1-Weighted (%)"])
    plt.ylim(60, 100)
    plt.ylabel("Puntuación (%)")
    plt.title("Comparación de Rendimiento en Test Set (EuroSAT Sentinel-2)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    out_plot = reports_dir / "benchmark_comparison.png"
    plt.savefig(out_plot, dpi=300)
    print(f"\nGráfico guardado exitosamente en: {out_plot}\n")
