# 🧠 Brain Tumor MRI Classification & Diagnostic Benchmark

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: 12K+ MRIs](https://img.shields.io/badge/Dataset-12%2C064%20Scans-orange.svg)]()
[![Classes: 4 Diagnoses](https://img.shields.io/badge/Classes-4%20Categories-purple.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A deep learning computer vision pipeline designed for multi-class classification of brain tumors from clinical Magnetic Resonance Imaging (MRI) scans. Built on clinical T1-weighted contrast-enhanced scans collected from **CSCR Hospital** and **Epic Health Care** (Chittagong, Bangladesh).

---

## 📌 Table of Contents

- [Overview & Clinical Background](#-overview--clinical-background)
- [Dataset Architecture](#-dataset-architecture)
- [Repository Structure](#-repository-structure)
- [Model Architecture & Pipeline](#-model-architecture--pipeline)
- [Quick Start & Installation](#-quick-start--installation)
- [Workflow & Usage](#-workflow--usage)
  - [1. Audit Dataset](#1-audit-dataset)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Train Model](#3-train-model)
  - [4. Evaluate & Generate Reports](#4-evaluate--generate-reports)
- [Clinical & Ethical Disclaimer](#-clinical--ethical-disclaimer)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🔬 Overview & Clinical Background

Brain tumors represent some of the most critical and challenging conditions in neuro-oncology. Early and precise classification of intracranial lesions directly impacts surgical planning, radiotherapy, and patient survival rates:

- **Glioma**: Aggressive tumors originating in glial tissue (astrocytomas, oligodendrogliomas, glioblastomas) with variable invasiveness.
- **Meningioma**: Typically benign, slow-growing tumors arising from the protective membranes (meninges) surrounding the brain and spinal cord.
- **Pituitary Tumor**: Adenomas affecting the pituitary gland at the base of the skull, frequently leading to hormonal dysfunction and optic chiasm compression.
- **No Tumor (Normal)**: Healthy control scans devoid of pathological neoplasm.

This repository provides an end-to-end, reproducible deep learning framework featuring state-of-the-art transfer learning architectures (ResNet, EfficientNet) to classify brain MRI scans into these 4 clinical categories.

---

## 📊 Dataset Architecture

The dataset comprises **12,064 preprocessed T1-weighted contrast-enhanced axial MRI scans**, split into an 80% training set and a 20% testing benchmark:

| Diagnostic Class | Clinical Significance | Train Images | Test Images | Total Images | Distribution |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Glioma** | Glial tissue neoplasm | 3,018 | 755 | **3,773** | 31.27% |
| **Meningioma** | Dural/meningeal tumor | 2,183 | 546 | **2,729** | 22.62% |
| **No Tumor** | Healthy / normal control | 1,945 | 487 | **2,432** | 20.16% |
| **Pituitary** | Sellar / pituitary adenoma | 2,504 | 626 | **3,130** | 25.94% |
| **Overall Total** | — | **9,650** | **2,414** | **12,064** | **100.00%** |

*Data Source: Clinical cohorts from CSCR Hospital and Epic Health Care, Chittagong, Bangladesh.*

---

## 📂 Repository Structure

```text
Brain-Tumor-MRI-Classification/
├── .gitignore                      # Python, PyTorch checkpoints, OS ignores
├── LICENSE                         # MIT License
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Environment dependencies
│
├── data/                           # Clinical dataset
│   ├── train/                      # Training subset (9,650 images)
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── test/                       # Test benchmark subset (2,414 images)
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── notebooks/                      # Exploratory Data Analysis & experiments
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/                            # Modular PyTorch pipeline
│   ├── __init__.py
│   ├── config.py                   # Centralized hyperparameter & path configs
│   ├── dataset.py                  # PyTorch DataLoader & data augmentation pipelines
│   ├── model.py                    # Transfer learning models (ResNet, EfficientNet)
│   ├── train.py                    # Training & validation loop with checkpointing
│   └── evaluate.py                 # Multi-class evaluation & confusion matrix generator
│
└── scripts/                        # Automation & audit utilities
    └── dataset_summary.py          # Dataset verification & class distribution audit
```

---

## 🧠 Model Architecture & Pipeline

The pipeline implements transfer learning leveraging deep convolutional neural networks initialized with ImageNet pre-trained weights:

1. **Preprocessing & Augmentation**:
   - Resized to standard input resolution ($224 \times 224$).
   - Random horizontal flips ($p = 0.5$) and stochastic angular rotation ($\pm 15^\circ$) for anatomical invariance.
   - Contrast/Brightness jitter for scan-acquisition robustness.
   - Channel normalization based on ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

2. **Backbones Supported**:
   - **ResNet-50**: Deep residual network with bottleneck blocks, well-suited for fine structural tumor boundary detection.
   - **ResNet-18**: Lightweight residual baseline for fast inference.
   - **EfficientNet-B0**: Compound-scaled neural architecture optimizing efficiency and FLOPs.

3. **Classification Head**:
   - Global Average Pooling (GAP) $\rightarrow$ Dropout ($p=0.3$) $\rightarrow$ Linear(256) $\rightarrow$ ReLU $\rightarrow$ Dropout ($p=0.2$) $\rightarrow$ Linear(4, Softmax).

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support recommended (CPU is also supported)

### 1. Clone the Repository
```bash
git clone https://github.com/Akma86/Brain-Tumor-MRI-Classification.git
cd Brain-Tumor-MRI-Classification
```

### 2. Create and Activate Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Workflow & Usage

### 1. Audit Dataset
Verify integrity across all 12,064 scans:
```bash
python scripts/dataset_summary.py
```

### 2. Exploratory Data Analysis (EDA)
Launch the interactive Jupyter notebook to inspect samples, class distributions, and pixel histograms:
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### 3. Train Model
Train a classifier (e.g. `resnet50` backbone) with automated learning rate scheduling and best checkpoint saving:
```bash
python src/train.py --model resnet50 --epochs 15 --batch-size 32 --lr 1e-4
```
Supported `--model` options: `resnet50`, `resnet18`, `efficientnet_b0`.

### 4. Evaluate & Generate Reports
Evaluate the best saved checkpoint on the unseen test set ($n = 2,414$ images) to compute precision, recall, F1-score, and render the confusion matrix:
```bash
python src/evaluate.py --checkpoint checkpoints/best_resnet50.pth --output-dir reports
```

Generated outputs:
- Per-class Precision, Recall, and F1-score metrics table.
- High-resolution Confusion Matrix heatmap saved to `reports/confusion_matrix_resnet50.png`.

---

## ⚖️ Clinical & Ethical Disclaimer

> [!CAUTION]
> **Medical Imaging Disclaimer**:
> This codebase and associated models are developed strictly for **academic, benchmark, and research purposes**. They have not been certified or approved as medical devices by any regulatory body (e.g., US FDA, CE Mark). This software **must not** be used as a primary diagnostic tool in clinical decision-making or patient healthcare management without direct board-certified radiologist oversight.

---

## 👨‍💻 Author & Acknowledgments

- **Author**: [Akmal Yaasir Fauzaan](https://github.com/Akma86)
- **Clinical Data Source**: CSCR Hospital and Epic Health Care (Chittagong, Bangladesh).
- **Mendeley Data Reference**: *A Large Brain Tumor MRI Dataset Collected from CSCR Hospital and Epic Health Care, Chittagong, Bangladesh* / *BDNeuro-MRI Dataset*.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
