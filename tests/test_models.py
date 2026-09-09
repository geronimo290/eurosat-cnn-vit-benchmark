"""
Tests unitarios para verificar la integridad y dimensiones de las arquitecturas.
"""

import pytest
import torch

from src.models.cnn import build_cnn_model
from src.models.vit_components import PatchEmbed, TransformerBlock
from src.models.hybrid import HybridCNNViT


def test_cnn_output_shape():
    """Valida que la CNN baseline entregue exactamente (Batch, Num_Classes)."""
    batch_size = 4
    num_classes = 10
    model = build_cnn_model(num_classes=num_classes, freeze_backbone=True)
    
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (batch_size, num_classes), f"Shape incorrecto: {output.shape}"
    assert not torch.isnan(output).any(), "La salida contiene valores NaN"


def test_patch_embed_dimensions():
    """Valida que PatchEmbed proyecte correctamente la grilla de parches."""
    batch_size = 2
    img_size = 224
    patch_size = 16
    embed_dim = 128
    expected_patches = (img_size // patch_size) ** 2  # 196

    patch_layer = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    tokens = patch_layer(dummy_input)

    assert tokens.shape == (batch_size, expected_patches, embed_dim)


def test_hybrid_cnn_vit_stability():
    """Valida que el modelo híbrido procese imágenes y mantenga estabilidad numérica."""
    batch_size = 2
    num_classes = 10
    model = HybridCNNViT(num_classes=num_classes, embed_dim=128, num_heads=4, num_transformer_layers=1)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    logits = model(dummy_input)

    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any(), "El modelo Híbrido produjo NaNs en la inferencia"
    assert not torch.isinf(logits).any(), "El modelo Híbrido produjo Infs en la inferencia"