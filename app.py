"""
Aplicación interactiva en Gradio para demostración en Hugging Face Spaces.
Permite comparar en vivo predicciones entre CNN y Vision Transformer.
"""

import sys
from pathlib import Path

# 1. Soporte ZeroGPU de Hugging Face (debe ir antes de cualquier llamada a CUDA)
try:
    import spaces
    gpu_decorator = spaces.GPU
except (ImportError, AttributeError):
    def gpu_decorator(fn):
        return fn

# 2. Asegurar que la raíz del proyecto esté en sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import gradio as gr
import torch

from src.models.cnn import build_cnn_model
from src.models.hybrid import HybridCNNViT
from src.inference.predict import predict_image, EUROSAT_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializamos modelos
cnn_model = build_cnn_model(num_classes=10, freeze_backbone=True).to(device)
hybrid_model = HybridCNNViT(num_classes=10, embed_dim=128, num_heads=4, num_transformer_layers=2).to(device)

# Cargar pesos si existen
cnn_weights = Path("weights/best_cnn.pth")
if cnn_weights.exists():
    cnn_model.load_state_dict(torch.load(cnn_weights, map_location=device))

hybrid_weights = Path("weights/best_hybrid.pth")
if hybrid_weights.exists():
    hybrid_model.load_state_dict(torch.load(hybrid_weights, map_location=device))


@gpu_decorator
def classify_satellite_image(image, model_choice):
    if image is None:
        return {}

    selected_model = cnn_model if model_choice == "EfficientNet-B0 (CNN)" else hybrid_model
    predictions = predict_image(image, selected_model, device, EUROSAT_CLASSES)
    return predictions


demo = gr.Interface(
    fn=classify_satellite_image,
    inputs=[
        gr.Image(type="pil", label="Subir Imagen Satelital"),
        gr.Radio(
            choices=["EfficientNet-B0 (CNN)", "Hybrid CNN-ViT (Transformer)"],
            value="Hybrid CNN-ViT (Transformer)",
            label="Arquitectura del Modelo"
        )
    ],
    outputs=gr.Label(num_top_classes=5, label="Predicción de Cobertura de Suelo"),
    title="🛰️ Sentinel-2 Land Use Classifier: CNN vs. Vision Transformer",
    description="Subí una imagen satelital para predecir el uso de suelo (bosque, río, cultivo, autopista, etc.) y comparar arquitecturas.",
    examples=[]
)

if __name__ == "__main__":
    demo.launch()
