"""
Módulo de inferencia defensiva para clasificación de imágenes satelitales.
"""

from typing import Dict, Union
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Clases oficiales de EuroSAT
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """Retorna el pipeline de preprocesamiento para inferencia en producción."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image(
    image: Union[str, Image.Image],
    model: nn.Module,
    device: torch.device,
    class_names: list = EUROSAT_CLASSES
) -> Dict[str, float]:
    """
    Recibe una imagen (ruta de archivo o PIL Image), valida formato, ejecuta inferencia
    y retorna un diccionario de probabilidades {Clase: Probabilidad}.
    """
    # 1. Validación defensiva de formato
    if isinstance(image, str):
        image = Image.open(image)

    # Conversión obligatoria a RGB (elimina canal Alfa de PNGs o escala de grises)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 2. Preprocesamiento
    transform = get_inference_transforms()
    tensor_input = transform(image).unsqueeze(0).to(device)  # Agrega dimensión de Batch (1, 3, 224, 224)

    # 3. Inferencia sin gradientes
    model.eval()
    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # 4. Formateo de salida para interfaces web (Gradio / FastAPI)
    results = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }

    # Ordenar de mayor a menor probabilidad
    return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))


if __name__ == "__main__":
    print("Módulo de inferencia listo.")