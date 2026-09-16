"""
Configuration settings for Brain Tumor MRI Classification
"""

from pathlib import Path
import torch

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Dataset Specification
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# Model Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Normalization parameters (Standard ImageNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
