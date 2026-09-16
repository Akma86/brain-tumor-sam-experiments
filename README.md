# 🧠 Brain Tumor SAM Experiments

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: 12K+ Scans](https://img.shields.io/badge/Dataset-12%2C064%20MRIs-orange.svg)]()
[![Models: SAM | MedSAM | SAM 2](https://img.shields.io/badge/Experiments-3%20SAM%20Variants-purple.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A hands-on portfolio project exploring and experimenting with different variants of Meta's **Segment Anything Model (SAM)** on clinical brain MRI scans collected from **CSCR Hospital** and **Epic Health Care** (Chittagong, Bangladesh).

---

## 📌 Overview

The goal of this project is to test how foundation segmentation models behave when applied to medical imaging, specifically brain tumor MRI scans:

- Can generalist vision models like **Vanilla SAM** identify brain tumors out-of-the-box?
- How much does domain-specific training in **MedSAM** improve boundary delineation?
- How do next-gen architectures like **SAM 2 / MobileSAM** perform in terms of speed and prompt responsiveness?

This repo serves as a playground and documentation of my experiments trying out different prompting styles (points, bounding boxes) across multiple SAM models.

---

## 🤖 SAM Variants Explored

| Model | Origin / Architecture | Why Test It? | Role in Experiments |
| :--- | :--- | :--- | :--- |
| **Vanilla SAM** | Meta AI *(ViT-B)* | Strong generalist foundation model trained on 11M natural images. | Baseline zero-shot prompt test |
| **MedSAM** | Ma et al. *(ViT-B Medical Neck)* | Specialized model fine-tuned on 1.5M+ medical image-mask pairs. | Comparing general vs. medical domain adaptation |
| **SAM 2 / MobileSAM** | Meta AI / Zhang et al. | Faster, more lightweight promptable segmentation architectures. | Testing inference speed and real-time feel |

---

## 📊 Dataset

The experiments use **12,064 clinical axial brain MRI scans** organized into standard splits:

| Class | Notes | Train Scans | Test Scans | Total Scans |
| :--- | :--- | :---: | :---: | :---: |
| **Glioma** | Brain tumors arising from glial cells (infiltrative borders) | 3,018 | 755 | **3,773** |
| **Meningioma** | Tumors originating in the meninges (typically distinct borders) | 2,183 | 546 | **2,729** |
| **Pituitary** | Tumors affecting the pituitary gland at the skull base | 2,504 | 626 | **3,130** |
| **No Tumor** | Healthy normal control scans | 1,945 | 487 | **2,432** |
| **Total** | — | **9,650** | **2,414** | **12,064** |

*Dataset Origin: Clinical scans collected from CSCR Hospital and Epic Health Care (Chittagong, Bangladesh).*

---

## 📂 Repository Structure

```text
brain-tumor-sam-experiments/
├── .gitignore                          # PyTorch checkpoints, caches, and system files
├── LICENSE                             # MIT License
├── README.md                           # Project documentation
├── requirements.txt                    # Environment dependencies
│
├── data/                               # 12,064 clinical MRI scans
│   ├── train/                          # 9,650 training scans
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── test/                           # 2,414 test scans
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # Interactive data exploration & scan inspection
│
├── src/                                # Experiment scripts & utilities
│   ├── __init__.py
│   ├── config.py                       # Paths, device settings, and parameters
│   ├── dataset.py                      # DataLoaders & image transforms
│   ├── sam_benchmark.py                # Prompt generation and evaluation helpers
│   ├── model.py                        # Baseline classification head
│   ├── train.py                        # Baseline training script
│   └── evaluate.py                     # Evaluation metrics & confusion matrix
│
└── scripts/
    └── dataset_summary.py              # Dataset verification script
```

---

## 🎯 Experimenting with Prompts

The experiments focus on 2 main prompting strategies:

1. **Bounding Box Prompts ($B_{box}$)**:
   - Providing a rough box around the tumor region (with slight random jitter to simulate human clicks).
2. **Point Prompts ($P_{point}$)**:
   - Providing single or multiple positive clicks inside the tumor lesion.

### Key Metrics Tracked
- **Dice Similarity Coefficient (DSC)**: Measures segmentation mask overlap.
- **Intersection over Union (IoU)**: Evaluates region precision.
- **Inference Time (ms)**: Measures latency per scan.

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Akma86/brain-tumor-sam-experiments.git
cd brain-tumor-sam-experiments

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Optional: Install SAM Packages
```bash
# Meta Vanilla SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Meta SAM 2
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 3. Verify Dataset
```bash
python scripts/dataset_summary.py
```

### 4. Run Exploratory Notebook
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## ⚖️ Disclaimer

> [!NOTE]
> This repository is a **personal learning & portfolio project**. It is meant for educational and experimental purposes only and is not intended for clinical use or medical diagnosis.

---

## 👨‍💻 Author & Acknowledgments

- **Author**: [Akmal Yaasir Fauzaan](https://github.com/Akma86)
- **Clinical Data Source**: CSCR Hospital and Epic Health Care (Chittagong, Bangladesh).
- **Model Frameworks**: Meta AI (SAM, SAM 2) and the MedSAM team.

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
