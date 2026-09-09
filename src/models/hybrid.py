"""
Módulo del modelo Híbrido CNN-ViT.
Combina un extractor de características convolucional (EfficientNet-B0)
con un encoder de Vision Transformer para razonamiento contextual global.
"""

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.models.vit_components import TransformerBlock


class HybridCNNViT(nn.Module):
    """
    Arquitectura Híbrida CNN-Vision Transformer.
    """
    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        freeze_cnn: bool = True
    ):
        super().__init__()
        # 1. Backbone Convolucional: Usamos solo las capas de características de EfficientNet
        weights = EfficientNet_B0_Weights.DEFAULT
        efficientnet = efficientnet_b0(weights=weights)
        self.cnn_backbone = efficientnet.features

        if freeze_cnn:
            for param in self.cnn_backbone.parameters():
                param.requires_grad = False

        # EfficientNet-B0 entrega 1280 canales en su mapa final (7x7 para entradas de 224x224)
        cnn_out_channels = 1280

        # 2. Proyección 1x1 para adaptar los 1280 canales a la dimensión de embedding del Transformer
        self.proj = nn.Conv2d(cnn_out_channels, embed_dim, kernel_size=1)

        # 3. Token de clasificación [CLS] y Embeddings posicionales aprendibles
        # 7x7 de resolución espacial = 49 parches + 1 token [CLS] = 50 tokens en total
        self.num_tokens = 49 + 1
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        # 4. Encoder del Transformer: Apilamos N bloques de autoatención
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_transformer_layers)
        ])

        # 5. Cabezal de Clasificación final (MLP Head)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Paso 1: Extracción convolucional -> (Batch, 1280, 7, 7)
        features = self.cnn_backbone(x)

        # Paso 2: Proyección de canales a embed_dim -> (Batch, embed_dim, 7, 7)
        projected = self.proj(features)

        # Paso 3: Aplanar a tokens de secuencia -> (Batch, 49, embed_dim)
        tokens = projected.flatten(2).transpose(1, 2)

        # Paso 4: Concatenar el token [CLS] expandido para todo el batch -> (Batch, 50, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # Paso 5: Sumar embeddings posicionales
        tokens = tokens + self.pos_embedding

        # Paso 6: Procesar a través del Transformer
        encoded_tokens = self.transformer(tokens)

        # Paso 7: Extraer el token [CLS] (índice 0) para clasificar
        cls_output = encoded_tokens[:, 0]

        # Paso 8: Predicción de logits de salida -> (Batch, num_classes)
        logits = self.mlp_head(cls_output)
        return logits


if __name__ == "__main__":
    print("Inicializando modelo Híbrido CNN-ViT...")
    model = HybridCNNViT(num_classes=10, embed_dim=128, num_heads=4, num_transformer_layers=2)
    
    dummy_batch = torch.randn(2, 3, 224, 224)
    out = model(dummy_batch)

    print(f"Salida del modelo Híbrido: {out.shape}")
    assert out.shape == (2, 10), f"Error en shape: se esperaba (2, 10), se obtuvo {out.shape}"
    print("¡Arquitectura Híbrida CNN-ViT completada y testeada con éxito!")