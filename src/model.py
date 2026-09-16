"""
Model Architecture Definitions for Brain Tumor MRI Classification
Supports pretrained transfer learning backbones (ResNet, EfficientNet).
"""

import torch
import torch.nn as nn
from torchvision import models
from src import config

def build_model(
    model_name: str = "resnet50",
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True
) -> nn.Module:
    """
    Builds a vision model with customized classification head.

    Args:
        model_name: Backbone architecture ('resnet50', 'resnet18', 'efficientnet_b0').
        num_classes: Number of target categories (default: 4).
        pretrained: Whether to load ImageNet pre-trained weights.
    """
    weights = "DEFAULT" if pretrained else None

    if model_name == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    elif model_name == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. Supported: resnet50, resnet18, efficientnet_b0")

    return model
