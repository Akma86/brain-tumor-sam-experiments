"""
Model Evaluation and Confusion Matrix Generation
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
from tqdm import tqdm

from src import config
from src.dataset import get_dataloaders
from src.model import build_model

@torch.no_grad()
def evaluate_model(checkpoint_path: str, output_dir: str = "reports"):
    device = torch.device(config.DEVICE)
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_name = checkpoint.get("model_name", "resnet50")
    classes = checkpoint.get("classes", config.CLASSES)

    print(f"[*] Loading model '{model_name}' from {checkpoint_path}")
    model = build_model(model_name=model_name, num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    _, test_loader = get_dataloaders(batch_size=config.BATCH_SIZE)

    all_preds = []
    all_targets = []

    print("[*] Running inference on test dataset...")
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Classification report
    print("\n" + "=" * 60)
    print("                CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(all_targets, all_preds, target_names=classes, digits=4)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_path = out_dir / f"confusion_matrix_{model_name}.png"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth Label")
    plt.title(f"Confusion Matrix - {model_name.upper()}")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"[OK] Confusion matrix saved to: {cm_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Classifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to save evaluation artifacts")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.output_dir)

if __name__ == "__main__":
    main()
