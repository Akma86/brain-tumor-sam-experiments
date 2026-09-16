# 🧠 Brain Tumor SAM Benchmark: Comparative Evaluation of Segment Anything Models on Clinical MRIs

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: 12K+ Scans](https://img.shields.io/badge/Dataset-12%2C064%20MRIs-orange.svg)]()
[![Architectures: SAM | MedSAM | SAM 2](https://img.shields.io/badge/Benchmark-3%20SAM%20Variants-purple.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A deep learning benchmark evaluating and comparing **multiple variants of the Segment Anything Model (SAM)** on clinical T1-weighted contrast-enhanced brain MRI scans collected from **CSCR Hospital** and **Epic Health Care** (Chittagong, Bangladesh).

---

## 📌 Table of Contents

- [Executive Summary & Motivation](#-executive-summary--motivation)
- [Target SAM Architectures Compared](#-target-sam-architectures-compared)
- [Dataset Overview & Clinical Scope](#-dataset-overview--clinical-scope)
- [Repository Structure](#-repository-structure)
- [Prompting & Evaluation Methodology](#-prompting--evaluation-methodology)
- [Quick Start & Installation](#-quick-start--installation)
- [Benchmarking Workflow](#-benchmarking-workflow)
- [Comparative Metrics](#-comparative-metrics)
- [Clinical & Ethical Disclaimer](#-clinical--ethical-disclaimer)
- [References & Acknowledgments](#-references--acknowledgments)

---

## 🔬 Executive Summary & Motivation

Foundation vision models like Meta's **Segment Anything Model (SAM)** have demonstrated remarkable zero-shot promptable segmentation on natural images. However, their clinical utility on **medical imaging modalities (such as brain MRI)** presents unique challenges:
- High variability in intracranial soft-tissue contrast.
- Subtle, infiltrative boundaries of **Gliomas** vs. well-circumscribed **Meningiomas** and skull-base **Pituitary** tumors.
- Domain shift between standard natural RGB photography (SA-1B) and radiological grayscale distributions.

This repository provides a standardized experimental testbed to benchmark and compare **2–3 prominent SAM variants**:
1. **Vanilla SAM (Meta AI)** — The generalist baseline foundation model.
2. **MedSAM (Wang et al.)** — Domain-adapted foundation model fine-tuned on 1.57M+ medical 2D/3D image-mask pairs.
3. **SAM 2 / MobileSAM** — Next-generation unified streaming architectures and lightweight edge distillations.

---

## ⚖️ Target SAM Architectures Compared

| Model Variant | Backbone / Paradigm | Pretraining Domain | Primary Strength in Medical Imaging | Intended Role in Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla SAM** *(Kirillov et al.)* | ViT-B / ViT-L / ViT-H | Natural images (SA-1B dataset, 11M images, 1B masks) | Generalist zero-shot baseline, high semantic capacity | Baseline zero-shot performance |
| **MedSAM** *(Ma & Wang et al.)* | ViT-B + Medical Neck | 1.57M+ medical image-mask pairs across 10+ modalities | Specialized in low-contrast tissue margins & radiological features | Domain-specific benchmark standard |
| **SAM 2 / MobileSAM** *(Ravi et al. / Zhang et al.)* | Memory Attention / TinyViT | SA-V video dataset / Knowledge Distillation | Faster inference latency, real-time interactive radiologist workflow | Efficiency vs. accuracy trade-off analysis |

---

## 📊 Dataset Overview & Clinical Scope

The benchmark utilizes **12,064 clinical axial brain MRI scans** organized into standardized splits:

| Class | Clinical Description | Train Scans | Test Scans | Total Scans | Proportion |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Glioma** | Primary intra-axial glial tumor with irregular infiltrative borders | 3,018 | 755 | **3,773** | 31.27% |
| **Meningioma** | Extra-axial dural-tail tumor with defined boundaries | 2,183 | 546 | **2,729** | 22.62% |
| **Pituitary** | Sellar and parasellar intracranial adenoma | 2,504 | 626 | **3,130** | 25.94% |
| **No Tumor** | Normal brain anatomy / negative control cohort | 1,945 | 487 | **2,432** | 20.16% |
| **Overall Total** | — | **9,650** | **2,414** | **12,064** | **100.00%** |

*Clinical Source: CSCR Hospital and Epic Health Care, Chittagong, Bangladesh.*

---

## 📂 Repository Structure

```text
Brain-Tumor-SAM-Benchmark/
├── .gitignore                          # PyTorch checkpoints, caches, and system files
├── LICENSE                             # MIT License
├── README.md                           # Comprehensive documentation
├── requirements.txt                    # Dependencies including PyTorch, SAM, OpenCV, MONAI
│
├── data/                               # 12,064 clinical MRI scans
│   ├── train/                          # 9,650 training scans across 4 classes
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── test/                           # 2,414 test benchmark scans
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # Interactive data exploration & scan inspection
│
├── src/                                # Modular Benchmarking & Modeling Pipeline
│   ├── __init__.py
│   ├── config.py                       # Global paths, device settings, and hyperparameters
│   ├── dataset.py                      # DataLoaders & medical image augmentations
│   ├── sam_benchmark.py                # Multi-SAM evaluation harness & prompt generators
│   ├── model.py                        # Transfer learning classification baseline
│   ├── train.py                        # Model training pipeline
│   └── evaluate.py                     # Evaluation metrics & confusion matrix generator
│
└── scripts/
    └── dataset_summary.py              # Automated dataset verification audit
```

---

## 🎯 Prompting & Evaluation Methodology

Each SAM variant is evaluated across standard clinical interaction paradigms:

1. **Bounding Box Prompts ($B_{box}$)**:
   - Simulates radiologist region-of-interest (ROI) selection.
   - Perturbed with stochastic boundary jitter ($\pm 5\text{px}$) to assess model stability under imperfect clinical prompts.
2. **Point Prompts ($P_{point}$)**:
   - Single and multiple positive foreground points located near tumor centroids.
   - Negative background points to suppress false positives in adjacent normal sulci/ventricles.
3. **Automatic Mask Generation (AMG)**:
   - Grid-based zero-shot evaluation without clinician prompt guidance.

### Quantitative Metrics
- **Sørensen–Dice Coefficient (DSC)**: Overlap fidelity between predicted lesion and ground truth.
- **Intersection over Union (mIoU / Jaccard Index)**: Spatial precision metric.
- **Inference Latency ($T_{inf}$)**: Runtime per scan (ms) on GPU/CPU to measure clinical practicality.

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Akma86/Brain-Tumor-SAM-Benchmark.git
cd Brain-Tumor-SAM-Benchmark
```

### 2. Environment Setup
```bash
# Setup virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Optional: Install SAM Frameworks
```bash
# Meta Vanilla SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Meta SAM 2 (Next-Gen)
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

---

## 💻 Benchmarking Workflow

### 1. Audit Dataset Scans
```bash
python scripts/dataset_summary.py
```

### 2. Run SAM Benchmark Suite
Run the multi-model comparison harness:
```bash
python src/sam_benchmark.py
```

### 3. Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## ⚖️ Clinical & Ethical Disclaimer

> [!CAUTION]
> **Medical Research Disclaimer**:
> This repository and its benchmark results are strictly intended for **academic, benchmark, and comparative research purposes**. Neither the models nor this software are approved medical devices (US FDA / CE Mark). They **must not** be used as a primary diagnostic tool in patient care without direct supervision by board-certified radiologists.

---

## 📚 References & Acknowledgments

1. **SAM**: Kirillov, A., et al. *"Segment Anything"*, ICCV 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
2. **MedSAM**: Ma, J., Wang, B., et al. *"Segment Anything in Medical Images"*, Nature Communications 2024. [arXiv:2304.12306](https://arxiv.org/abs/2304.12306)
3. **SAM 2**: Ravi, N., et al. *"SAM 2: Segment Anything in Images and Videos"*, Meta AI 2024. [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)
4. **Clinical Dataset**: Clinical MRI cohorts sourced from CSCR Hospital and Epic Health Care (Chittagong, Bangladesh).

---

## 📄 License

Distributed under the **MIT License** — see [LICENSE](LICENSE) for details.
