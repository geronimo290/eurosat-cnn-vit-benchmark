# 🛰️ EuroSAT Land Use Classification: CNN vs. Hybrid Vision Transformer Benchmark

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![CI](https://github.com/geronimo290/eurosat-cnn-vit-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/geronimo290/eurosat-cnn-vit-benchmark/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning Engineering study comparing **Convolutional Neural Networks (CNN)** and a **Hybrid CNN-Vision Transformer (ViT)** for satellite land use classification on the **Sentinel-2 EuroSAT** benchmark.

---

## 📌 Business & Technical Motivation

Land use and land cover (LULC) classification from satellite imagery is critical for environmental monitoring, precision agriculture, urban planning, and climate risk modeling.
- **CNNs (EfficientNet-B0):** Excel at extracting localized spatial textures (roads, crop boundaries, residential building clusters) with high parameter efficiency and low inference latency.
- **Vision Transformers (ViTs):** Leverage Multi-Head Self-Attention to model long-range global semantic dependencies across large territorial expanses (e.g., transport corridors intersecting industrial zones and waterways).
- **Study Goal:** Design, benchmark, and deploy a **Hybrid CNN-ViT architecture** against a pure CNN baseline to evaluate trade-offs across **Accuracy, F1-Score, Parameter Count, and Inference Latency**.

---

## 🏗️ Repository Architecture

```text
eurosat-cnn-vit-benchmark/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated CI: Linting (Ruff) + Unit Tests (pytest)
├── app/
│   └── main.py                    # Interactive Gradio UI (ready for Hugging Face Spaces)
├── data/
│   └── raw/                       # Local EuroSAT dataset storage (gitignored)
├── reports/                       # Serialized metrics and test reports (*.json)
├── src/
│   ├── data/
│   │   └── dataset.py             # DataLoaders, TransformedSubset (prevents leakage), 70/15/15 split
│   ├── models/
│   │   ├── cnn.py                 # EfficientNet-B0 baseline with Transfer Learning
│   │   ├── vit_components.py      # PatchEmbed, Multi-Head Self-Attention, TransformerBlock
│   │   └── hybrid.py              # Hybrid CNN-ViT with [CLS] token & Positional Embeddings
│   ├── training/
│   │   └── trainer.py             # Hardware-agnostic trainer (CUDA/CPU), checkpointing, metrics logger
│   └── inference/
│       └── predict.py             # Defensive image sanitization, RGB conversion, predictor
├── tests/
│   ├── test_models.py             # Unit tests: Tensor shapes, stability (No NaNs/Infs)
├── weights/                       # Trained model checkpoints (*.pth, gitignored)
├── train.py                       # CLI entrypoint for training and test evaluation
├── .gitignore                     # Strict rules excluding large weights, caches, and datasets
├── README.md                      # Production-grade engineering documentation
└── requirements.txt               # Pinned cross-platform dependencies
```

---

## 📊 Dataset: Sentinel-2 EuroSAT

- **Volume:** 27,000 labeled optical RGB satellite images.
- **Classes (10 balanced):** `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`.
- **Methodology & Partitioning:**
  - **Train (70% - ~18,900 images):** Includes spatial Data Augmentation (Horizontal/Vertical Flips, Rotations).
  - **Validation (15% - ~4,050 images):** Evaluated strictly on clean, non-augmented images for early stopping and tuning.
  - **Test (15% - ~4,050 images):** Isolated holdout partition used strictly for unbiased final evaluation.
  - **Data Leakage Prevention:** Custom `TransformedSubset` ensures validation and test sets never inherit training augmentations.

---

## 🧠 Architectures

### 1. Baseline CNN (`src/models/cnn.py`)
- **Backbone:** `EfficientNet-B0` pre-trained on ImageNet using modern PyTorch weight enums (`EfficientNet_B0_Weights.DEFAULT`).
- **Feature Adaptor:** Linear classification head mapping 1,280 features to 10 land cover classes.

### 2. Hybrid CNN-ViT (`src/models/hybrid.py`)
- **Feature Extractor:** `EfficientNet-B0` convolutional stem generating localized 7x7 spatial feature maps.
- **1x1 Projection:** Projects 1,280 channels down to `embed_dim` (128) -> 49 spatial visual tokens.
- **Classification Token (`[CLS]`):** Learnable token prepended to sequence (50 tokens total) to aggregate global scene semantics.
- **Positional Encoding:** Learnable positional vectors added to maintain spatial topology.
- **Transformer Encoder:** Apiled Multi-Head Self-Attention blocks with residual connections, LayerNorm, and GELU MLP.
- **Classification Head:** Linear classifier operating exclusively on the output representation of the `[CLS]` token.

---

## 🚀 Quickstart

### 1. Setup Environment
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Automated Tests
```bash
python -m pytest tests/ -v
```

### 3. Training via CLI (`train.py`)
```bash
# Train Baseline CNN:
python train.py --model cnn --epochs 5 --batch-size 64

# Train Hybrid CNN-ViT:
python train.py --model hybrid --epochs 5 --batch-size 64
```
*Outputs are saved automatically to `weights/best_<model>.pth` and `reports/test_report_<model>.json`.*

### 4. Run Interactive Demo (Gradio)
```bash
python -m app.main
```
Navigate to `http://127.0.0.1:7860` in your browser.

---

## 📈 Empirical Benchmark Results (Test Set - 4,050 Images)

Evaluated strictly on the unseen **15% holdout test partition**:

| Architecture | Backbone / Components | Accuracy | F1-Macro | F1-Weighted | Test Loss |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline CNN** | EfficientNet-B0 (Frozen Stem + Linear Head) | **83.60%** | **0.8314** | **0.8348** | 0.7438 |
| **Hybrid CNN-ViT** | EfficientNet-B0 + [CLS] + PosEmbedding + TransformerBlocks | **92.79%** | **0.9265** | **0.9275** | **0.2059** |

> 📊 Automatically generated via `python scripts/compare_models.py`, which parses serialized `reports/test_report_*.json` files.

---

## 🌐 Deployment to Hugging Face Spaces

This repository includes a production-ready entrypoint (`app.py`) for one-click deployment on Hugging Face Spaces:

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) selecting the **Gradio** SDK.
2. Clone your Space repository or link it directly to your GitHub repository.
3. The Space will automatically detect `app.py` and `requirements.txt`, launching the interactive satellite classifier live on the web with a public URL!
