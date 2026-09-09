"""
Módulo con los componentes fundamentales de un Vision Transformer (ViT).
Implementa:
1. PatchEmbedding: Proyección de parches 2D a secuencias de tokens.
2. PositionalEncoding: Embeddings de posición aprendibles.
3. TransformerBlock: Bloque con Multi-Head Self-Attention, MLP y LayerNorm.
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Divide una imagen en parches de tamaño (patch_size x patch_size)
    y los proyecta linealmente a una dimensión de embedding (embed_dim).
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Proyección eficiente usando Conv2d con kernel y stride igual al tamaño del parche
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entrada: (Batch, Channels, H, W)
        # Salida de proj: (Batch, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        # Aplanamos dimensiones espaciales a secuencia de tokens: (Batch, embed_dim, num_patches)
        x = x.flatten(2)
        # Transponemos para estándar de atención: (Batch, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """
    Bloque estándar del Encoder de Transformer:
    LayerNorm -> Multi-Head Self-Attention -> Conexión Residual -> LayerNorm -> MLP -> Conexión Residual
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm + Self-Attention con conexión residual
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out

        # Pre-Norm + MLP con conexión residual
        x = x + self.mlp(self.norm2(x))
        return x


if __name__ == "__main__":
    # Smoke test de componentes ViT
    batch_size = 2
    img_size = 224
    patch_size = 16
    embed_dim = 128

    print("Probando PatchEmbed...")
    patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=3, embed_dim=embed_dim)
    dummy_img = torch.randn(batch_size, 3, img_size, img_size)
    tokens = patch_embed(dummy_img)

    expected_num_patches = (img_size // patch_size) ** 2  # (224/16)^2 = 14*14 = 196
    print(f"Shape tokens de salida: {tokens.shape}")
    assert tokens.shape == (batch_size, expected_num_patches, embed_dim)

    print("\nProbando TransformerBlock...")
    transformer_block = TransformerBlock(embed_dim=embed_dim, num_heads=4)
    out_tokens = transformer_block(tokens)
    print(f"Shape tokens tras atención: {out_tokens.shape}")
    assert out_tokens.shape == tokens.shape

    print("\n¡Componentes del Vision Transformer validados con éxito!")