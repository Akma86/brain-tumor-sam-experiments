"""
PyTorch Dataset and DataLoaders for Brain Tumor MRI Scans
"""

from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src import config

def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns image transformation pipelines for training (with augmentations)
    and testing/evaluation (deterministic preprocessing).
    """
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])

    return train_transform, test_transform

def get_dataloaders(
    train_dir: Path = config.TRAIN_DIR,
    test_dir: Path = config.TEST_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns train and test DataLoaders using ImageFolder.
    """
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
