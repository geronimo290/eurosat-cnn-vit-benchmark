"""
Módulo de ingestión y preparación de datos para EuroSAT.

Este módulo se encarga de:
1. Definir los pipelines de transformaciones (Data Augmentation para train, solo resize y normalización para val/test).
2. Resolver de forma defensiva la ubicación de las imágenes (soporta data/raw/archive/EuroSAT, etc.).
3. Dividir el dataset en Train (70%), Validation (15%) y Test (15%) de forma reproducible con semilla fija.
4. Aplicar transformaciones diferenciadas a los subsets para evitar Data Leakage durante la evaluación.
5. Construir DataLoaders optimizados para streaming hacia GPU/CPU.
"""

import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class TransformedSubset(Dataset):
    """
    Subconjunto de datos con transformación personalizada.
    
    ¿Por qué existe esta clase?
    En PyTorch, 'random_split' divide un dataset en subconjuntos, pero todos heredan
    la misma transformación del dataset base. Si el dataset base tiene Data Augmentation,
    el conjunto de validación y test también recibirían imágenes rotadas/alteradas,
    lo que distorsiona la evaluación real del modelo (Data Leakage conceptual).
    
    Esta clase permite envolver un subset y aplicarle su transformación específica.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Genera los pipelines de transformación para entrenamiento y evaluación.

    - Train: Redimensionamiento a 224x224 (requerido por EfficientNet y ViT),
      aumento de datos espacial y normalización con estadísticas de ImageNet.
    - Eval (Val/Test): Sin aumento aleatorio. Solo redimensionamiento y normalización.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transforms, eval_transforms


def get_dataloaders(
    data_dir: str = "data/raw",
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Construye y retorna los DataLoaders de Train, Val y Test para EuroSAT.

    Args:
        data_dir: Directorio raíz donde reside el dataset.
        batch_size: Tamaño de lote para los DataLoaders.
        num_workers: Número de subprocesos para carga de datos (0 recomendado en Windows).
        seed: Semilla para garantizar un split reproducible.
        image_size: Resolución cuadrada de entrada para las redes (default 224).

    Returns:
        Tupla con (train_loader, val_loader, test_loader, class_names).
    """
    path = Path(data_dir)

    # Búsqueda defensiva del directorio que contiene las carpetas de clases
    candidate_paths = [
        path / "archive" / "EuroSAT",
        path / "EuroSAT",
        path / "2750",
        path
    ]

    dataset_path = None
    for candidate in candidate_paths:
        if candidate.exists() and (candidate / "AnnualCrop").exists():
            dataset_path = candidate
            break

    if dataset_path is None:
        raise FileNotFoundError(
            f"No se encontraron las clases de EuroSAT en {data_dir}. "
            "Asegúrate de que la carpeta contenga subdirectorios como 'AnnualCrop', 'Forest', etc."
        )

    # Transformaciones separadas
    train_tf, eval_tf = get_transforms(image_size=image_size)

    # Cargamos el dataset base sin transformaciones para extraer índices y clases
    base_dataset = datasets.ImageFolder(root=str(dataset_path))
    class_names = base_dataset.classes
    total_samples = len(base_dataset)

    # Cálculo de particiones: 70% Train, 15% Validation, 15% Test
    train_size = int(0.70 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size

    # Split reproducible con generador determinista
    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(
        base_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Aplicamos transformaciones independientes mediante TransformedSubset
    train_dataset = TransformedSubset(train_sub, transform=train_tf)
    val_dataset = TransformedSubset(val_sub, transform=eval_tf)
    test_dataset = TransformedSubset(test_sub, transform=eval_tf)

    pin_memory = torch.cuda.is_available()

    # Creación de DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, class_names