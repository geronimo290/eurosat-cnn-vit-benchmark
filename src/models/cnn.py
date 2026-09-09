"""
Módulo del modelo convolucional de referencia (Baseline CNN).
Implementa Transfer Learning con EfficientNet-B0 adaptado para EuroSAT (10 clases).
"""
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_cnn_model(
    num_classes: int = 10,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Construye y adapta un modelo EfficientNet-B0 preentrenado para clasificación de uso de suelo.
    Args:
        num_classes: Número de clases de salida (10 para EuroSAT).
        freeze_backbone: Si es True, congela los pesos de las capas convolucionales
                         para entrenar únicamente el cabezal de clasificación.
    Returns:
        Modelo PyTorch adaptado (nn.Module).
    """
    #Carga de modelos
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    #Congelamientop del extractor de caracteristicas
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    #Cambio del cabezal clasificador En EfficientNet, model.classifier es un Sequential([Dropout, Linear])
    in_features = model.classifier[1].in_features

    #Remplazo de la ultima capa
    model.classifier = nn.Linear(in_features=in_features, out_features=num_classes)

    return model

if __name__ == "__main__":
    # Smoke test rápido para validar dimensiones
    model = build_cnn_model(num_classes=10, freeze_backbone=True)
    
    # Creamos un tensor dummy simulando un batch de 2 imágenes (Batch=2, Canales=3, H=224, W=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Salida del modelo para batch de prueba: {output.shape}")
    assert output.shape == (2, 10), f"Error de dimensiones: se esperaba (2, 10), se obtuvo {output.shape}"
    print("¡CNN Baseline lista y dimensiones validadas correctamente!")