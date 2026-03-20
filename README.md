# 🔬 Clinical Trial Cell Imaging — CNN Transfer Learning

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/mderya/pharma-cell-classification-cnn)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

## 🎯 Project Overview

End-to-end **CNN-based cellular classification pipeline** for **Phase II clinical trials**, comparing 11 deep learning architectures using Transfer Learning. Manual pathology analysis was taking months and causing critical delays in drug development timelines — this solution reduced analysis time from months to minutes.

**Key Results:**
- ✅ **99% classification accuracy** (EfficientNet best performer)
- ✅ **90% cost savings** vs. manual analysis
- ✅ **11 CNN architectures** benchmarked (ResNet, EfficientNet, VGG, DenseNet, etc.)
- ✅ Analysis time reduced from **months to minutes**

## 💊 Pharma Context

In pharmaceutical **Phase II clinical trials**, accurate cellular classification is critical for:
- Evaluating drug efficacy at the cellular level
- Detecting adverse cellular responses early
- Accelerating go/no-go decisions in drug development
- Reducing cost of failed trials

This project demonstrates how **Transfer Learning** can be applied to pharma imaging data with minimal labeled samples — a key challenge in clinical settings.

## 🤖 Models Benchmarked

| Model | Accuracy | Parameters |
|-------|----------|------------|
| **EfficientNetB0** | **99.2%** | 5.3M |
| ResNet50 | 98.7% | 25.6M |
| VGG16 | 97.9% | 138M |
| DenseNet121 | 98.4% | 8M |
| InceptionV3 | 97.5% | 23.9M |
| MobileNetV2 | 96.8% | 3.4M |
| + 5 more architectures | ... | ... |

## 🏗️ Project Structure
```
├── notebooks/
│   └── clinical_trial_cell_imaging_cnn.ipynb   # Full analysis notebook
├── requirements.txt                              # Dependencies
└── README.md
```

## 🔧 Tech Stack

- **Deep Learning:** PyTorch, TensorFlow/Keras
- **Models:** ResNet, EfficientNet, VGG, DenseNet, InceptionV3, MobileNet
- **Techniques:** Transfer Learning, Fine-tuning, Data Augmentation
- **Visualization:** Matplotlib, Seaborn, Grad-CAM

## 🔧 Installation
```bash
git clone https://github.com/mattderya/clinical-trial-cell-imaging-cnn
cd clinical-trial-cell-imaging-cnn
pip install -r requirements.txt
```

## 👤 Author

**Matt Derya** | Data Scientist | 15+ years pharmaceutical domain expertise

- 🌐 [Website](https://mattderya.com)
- 🔗 [LinkedIn](https://linkedin.com/in/mttdryai)
- 📓 [Kaggle Notebook](https://www.kaggle.com/code/mderya/pharma-cell-classification-cnn)
