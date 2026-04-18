# Mars-Bench XGBoost: Benchmarking mb-domars16k

This repository contains the code and methodology for benchmarking the `mb-domars16k` dataset from the **Mars-Bench** paper using a ViT-L/16 feature extraction and XGBoost classification pipeline.

##  Key Results
We established a new benchmark for the `mb-domars16k` dataset using frozen ViT-L/16 embeddings:

| Metric | IQM Score (7 seeds) | baseline (Gemini 2.0 Flash) |
| :--- | :--- | :--- |
| **Weighted F1-score** | **0.8352** | 0.32 |
| **Accuracy** | **0.8371** | - |

## 🛠️ Pipeline Overview
The pipeline consists of four main steps:
1. **Data Download**: Downloads the dataset from HuggingFace (`S-Lab/mb-domars16k`).
2. **Feature Extraction**: Uses a frozen ViT-L/16 (`google/vit-large-patch16-224`) to extract 1024-dimensional [CLS] token embeddings.
3. **Training & Imbalance Handling**: Trains XGBoost with multiple strategies to handle class imbalance (Sample Weighting, SMOTE, and Hybrid).
4. **Evaluation**: Rigorous 7-seed evaluation reporting InterQuartile Mean (IQM) and bootstrap confidence intervals.

##  Installation

### Prerequisites
- Python 3.12
- NVIDIA GPU (RTX 4070 or better recommended)
- CUDA 12.1+

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd mars-bench-xgboost

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows

# Install requirements
pip install -r requirements.txt
```

##  Usage

Run the end-to-end pipeline:
```bash
python run_pipeline.py
```

Or run individual steps:
```bash
python data/download_data.py
python src/extract_features.py
python src/train_xgboost.py
python src/evaluate.py
```

##  Repository Structure
```
mars_bench_xgboost/
├── data/
│   └── download_data.py      # Dataset loading and preprocessing
├── src/
│   ├── extract_features.py   # ViT feature extraction
│   ├── train_xgboost.py      # XGBoost training with imbalance handling
│   ├── evaluate.py           # 7-seed statistical evaluation
│   └── utils.py              # Helper functions
├── configs/
│   └── xgboost_config.yaml   # Model hyperparameters
├── outputs/                  # Saved artifacts (gitignore)
└── requirements.txt
```

##  Methodology Reference
This evaluation follows the rigorous statistical standards described in:
> *Mars-Bench: Benchmarking Game Models for Mars Explorations*
> Link: https://arxiv.org/abs/2510.24010

##  License
MIT
