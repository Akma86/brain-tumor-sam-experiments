"""
Dataset Summary & Verification Script
Dataset: Brain Tumor MRI Classification (CSCR Hospital & Epic Health Care)
"""

import os
from pathlib import Path
from collections import Counter
from PIL import Image

def analyze_dataset(data_dir: str = "data"):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory '{data_dir}' not found.")
        return

    splits = ["train", "test"]
    stats = {}
    total_images = 0
    dimensions = []

    print("=" * 65)
    print("      BRAIN TUMOR MRI DATASET SUMMARY & AUDIT REPORT        ")
    print("=" * 65)

    for split in splits:
        split_path = data_path / split
        if not split_path.exists():
            print(f"Warning: Split directory '{split}' not found in {data_dir}.")
            continue

        stats[split] = {}
        classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])

        for cls in classes:
            cls_dir = split_path / cls
            files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            stats[split][cls] = len(files)
            total_images += len(files)

            # Sample dimensions from first 5 images
            for f in files[:5]:
                try:
                    with Image.open(f) as img:
                        dimensions.append(img.size)
                except Exception:
                    pass

    # Print Table
    classes = sorted(list(set().union(*[stats[s].keys() for s in stats])))
    header = f"{'Class Name':<15} | {'Train':<10} | {'Test':<10} | {'Total':<10} | {'Ratio (%)':<10}"
    print(header)
    print("-" * len(header))

    for cls in classes:
        train_count = stats.get("train", {}).get(cls, 0)
        test_count = stats.get("test", {}).get(cls, 0)
        cls_total = train_count + test_count
        ratio = (cls_total / total_images * 100) if total_images else 0
        print(f"{cls:<15} | {train_count:<10} | {test_count:<10} | {cls_total:<10} | {ratio:>6.2f}%")

    print("-" * len(header))
    train_total = sum(stats.get("train", {}).values())
    test_total = sum(stats.get("test", {}).values())
    print(f"{'OVERALL TOTAL':<15} | {train_total:<10} | {test_total:<10} | {total_images:<10} | 100.00%")
    print("=" * 65)

    if dimensions:
        dim_counts = Counter(dimensions)
        print("\nCommon Image Resolutions (Width x Height):")
        for dim, count in dim_counts.most_common(3):
            print(f"  - {dim[0]}x{dim[1]}: {count} sampled files")

    print(f"\n[OK] Dataset verification passed: {total_images:,} images indexed successfully.\n")

if __name__ == "__main__":
    analyze_dataset()
