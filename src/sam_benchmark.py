"""
Brain Tumor MRI SAM Benchmarking Module
Comparative framework for Segment Anything Model (SAM) variants:
1. Meta SAM (Vanilla ViT-B / ViT-H)
2. MedSAM (Medical Segment Anything Model - Wang et al.)
3. SAM 2 / MobileSAM (Next-gen / lightweight real-time architectures)
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import cv2
from PIL import Image

def calculate_dice_score(pred_mask: np.ndarray, target_mask: np.ndarray, smooth: float = 1e-6) -> float:
    """Computes Sørensen–Dice coefficient between binary predicted and target masks."""
    pred = (pred_mask > 0).astype(np.float32)
    target = (target_mask > 0).astype(np.float32)
    intersection = (pred * target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def calculate_iou_score(pred_mask: np.ndarray, target_mask: np.ndarray, smooth: float = 1e-6) -> float:
    """Computes Intersection over Union (Jaccard Index)."""
    pred = (pred_mask > 0).astype(np.float32)
    target = (target_mask > 0).astype(np.float32)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + smooth) / (union + smooth))

def generate_box_prompt_from_mask(mask: np.ndarray, perturbation_pixels: int = 5) -> np.ndarray:
    """
    Simulates a bounding box prompt with optional jitter/perturbation
    to test prompt robustness of SAM variants.
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return np.array([0, 0, mask.shape[1], mask.shape[0]])

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Apply perturbation
    h, w = mask.shape
    x_min = max(0, x_min - np.random.randint(0, perturbation_pixels + 1))
    y_min = max(0, y_min - np.random.randint(0, perturbation_pixels + 1))
    x_max = min(w - 1, x_max + np.random.randint(0, perturbation_pixels + 1))
    y_max = min(h - 1, y_max + np.random.randint(0, perturbation_pixels + 1))

    return np.array([x_min, y_min, x_max, y_max])

def generate_point_prompts(mask: np.ndarray, num_points: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Generates positive foreground point prompts centered inside the target lesion."""
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0:
        # Fallback to image center
        h, w = mask.shape
        return np.array([[w // 2, h // 2]]), np.array([1])

    sampled_indices = np.random.choice(len(x_indices), size=min(num_points, len(x_indices)), replace=False)
    points = np.stack([x_indices[sampled_indices], y_indices[sampled_indices]], axis=1)
    labels = np.ones(len(points), dtype=np.int32)  # 1 = foreground point
    return points, labels

class SAMBenchmarkRunner:
    """
    Benchmarking harness for multi-variant SAM experiments.
    Supports Vanilla SAM, MedSAM, and SAM 2 / MobileSAM interfaces.
    """
    def __init__(self, variant_name: str, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        self.variant_name = variant_name.lower()
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.predictor = None

    def initialize_model(self):
        """Initializes predictor based on the chosen SAM variant."""
        print(f"[*] Initializing SAM variant: '{self.variant_name}' on device: {self.device}")
        # Placeholder / hook for model loading
        # In practice:
        # if self.variant_name == 'sam':
        #     from segment_anything import sam_model_registry, SamPredictor
        #     sam = sam_model_registry["vit_b"](checkpoint=self.checkpoint_path).to(self.device)
        #     self.predictor = SamPredictor(sam)
        # elif self.variant_name == 'medsam':
        #     ...
        return True

    def run_benchmark_on_sample(
        self,
        image_path: Path,
        ground_truth_mask: Optional[np.ndarray] = None,
        prompt_type: str = "box"
    ) -> Dict[str, float]:
        """
        Runs inference on a single MRI scan and returns latency and overlap metrics.
        """
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Simulation metrics for pipeline verification
        metrics = {
            "variant": self.variant_name,
            "prompt_type": prompt_type,
            "image_path": str(image_path),
            "dice": 0.0,
            "iou": 0.0,
            "inference_time_ms": 0.0
        }
        return metrics

if __name__ == "__main__":
    print("=" * 60)
    print("      BRAIN TUMOR SAM BENCHMARKING HARNESS INITIALIZED       ")
    print("=" * 60)
    print("Variants supported:")
    print("  1. Meta SAM (ViT-B / ViT-L / ViT-H)")
    print("  2. MedSAM (Medical Segment Anything - Domain Specialized)")
    print("  3. SAM 2 / MobileSAM (Next-Gen & Real-Time Promptable Segmentation)")
    print("=" * 60)
