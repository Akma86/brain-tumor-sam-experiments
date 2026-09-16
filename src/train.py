"""
Training Pipeline for Brain Tumor MRI Classification
"""

import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src import config
from src.dataset import get_dataloaders
from src.model import build_model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Train Brain Tumor MRI Classifier")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--save-dir", type=str, default=str(config.CHECKPOINT_DIR))
    args = parser.parse_args()

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_file = save_path / f"best_{args.model}.pth"

    device = torch.device(config.DEVICE)
    print(f"[*] Training on device: {device}")
    print(f"[*] Backbone model: {args.model}")

    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size)
    model = build_model(model_name=args.model, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": config.CLASSES
            }, best_model_file)
            print(f"  --> Saved new best checkpoint to {best_model_file} (Val Acc: {val_acc * 100:.2f}%)")

    total_time = (time.time() - start_time) / 60
    print(f"\n[OK] Training completed in {total_time:.2f} minutes.")
    print(f"[OK] Best Validation Accuracy: {best_val_acc * 100:.2f}%")

if __name__ == "__main__":
    main()
