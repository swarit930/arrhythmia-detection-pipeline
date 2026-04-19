# Arrhythmia Detection Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![MIT-BIH](https://img.shields.io/badge/dataset-MIT--BIH-green.svg)](https://physionet.org/content/mitdb/)

An end-to-end machine-learning pipeline for detecting cardiac arrhythmias from
single-lead ECG signals using the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/).

The pipeline covers every stage from raw signal acquisition to model explainability:

| Phase | Description |
|---|---|
| 1 Data | Download & load 48 MIT-BIH records via `wfdb` |
| 2 Pre-processing | Bandpass filter → baseline removal → normalisation → R-peak detection → beat segmentation |
| 3 Features | HRV metrics, morphological descriptors, delta-RR context features |
| 4 Baseline | Scikit-learn classifier (Random Forest, Logistic Regression, SVM, …) |
| 5 CNN | 1-D Convolutional Neural Network for single-beat classification |
| 6 LSTM | Bidirectional LSTM with attention for sequence-level classification |
| 7 Evaluation | Patient-aware splits, confusion matrices, PR curves, AUC-ROC |
| 8 Explainability | Integrated Gradients (captum) and LSTM attention visualisation |

---

## Project structure

```
arrhythmia-detection-pipeline/
├── README.md
├── requirements.txt
├── config.py                  # all hyper-parameters and path constants
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # ECGDataLoader – wfdb wrapper
│   ├── preprocessor.py        # ECGPreprocessor – signal processing chain
│   ├── feature_extractor.py   # FeatureExtractor – hand-crafted features
│   ├── baseline_model.py      # BaselineModel – sklearn pipeline
│   ├── models_pytorch.py      # ECG_CNN, ECG_LSTM
│   ├── loss_functions.py      # FocalLoss, WeightedCrossEntropyLoss
│   ├── trainer.py             # Trainer – training loop + early stopping
│   ├── evaluator.py           # PatientSplitter, ModelEvaluator
│   ├── evaluation_report.py   # EvaluationReport – JSON + figure outputs
│   ├── explainability.py      # GradientExplainer, LSTMAttentionVisualizer
│   ├── inference.py           # InferenceEngine – real-time prediction
│   └── utils.py               # logging, seeding, plotting, AverageMeter
├── scripts/
│   ├── train_baseline.py      # train sklearn baseline
│   ├── train_cnn.py           # train CNN model
│   ├── train_lstm.py          # train LSTM model
│   ├── evaluate_all.py        # compare all trained models
│   └── demo.py                # end-to-end demonstration
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   └── 03_model_training.ipynb
└── tests/
    ├── __init__.py
    ├── test_preprocessor.py
    ├── test_feature_extraction.py
    └── test_models.py
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/swarit930/arrhythmia-detection-pipeline.git
cd arrhythmia-detection-pipeline

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU support**: if a CUDA-capable GPU is available the training scripts will
> use it automatically. Install the CUDA build of PyTorch manually if needed:
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

---

## Quick start

### Run the demo (no data download required)

```bash
python scripts/demo.py --synthetic
```

This generates a synthetic ECG, detects R-peaks, segments beats, and saves
plots to `results/demo/`.

### Train the CNN on real MIT-BIH data

```bash
# Download data + train
python scripts/train_cnn.py --download --epochs 50

# Evaluate all models
python scripts/evaluate_all.py
```

### Train the baseline scikit-learn model

```bash
python scripts/train_baseline.py --download --model-type random_forest
```

---

## Configuration

All paths and hyper-parameters live in `config.py`. The key constants are:

| Constant | Default | Description |
|---|---|---|
| `DATA_DIR` | `data/` | MIT-BIH record storage |
| `MODEL_DIR` | `models/` | Saved checkpoints |
| `RESULTS_DIR` | `results/` | Figures and JSON reports |
| `SAMPLING_RATE` | `360` | MIT-BIH native sampling rate (Hz) |
| `BEAT_WINDOW_SAMPLES` | `129` | Samples per beat (360 ms × 360 Hz) |
| `AAMI_CLASSES` | `[N, S, V, F, Q]` | AAMI beat categories |
| `LEARNING_RATE` | `1e-3` | Initial learning rate |
| `BATCH_SIZE` | `256` | Mini-batch size |
| `EPOCHS` | `100` | Max training epochs |
| `EARLY_STOPPING_PATIENCE` | `10` | Early-stop patience |

Override paths via environment variables:

```bash
export ECG_DATA_DIR=/mnt/data/mitbih
```

---

## AAMI label mapping

The pipeline maps MIT-BIH beat annotation symbols to the five AAMI classes:

| AAMI | MIT-BIH symbols | Description |
|---|---|---|
| **N** | N, L, R, e, j | Normal / bundle-branch block |
| **S** | A, a, J, S | Supraventricular ectopic |
| **V** | V, E | Ventricular ectopic |
| **F** | F | Fusion beat |
| **Q** | /, f, Q, U | Paced / unclassifiable |

---

## Model architectures

### ECG_CNN

```
Input (batch, 1, 129)
  Conv1d(1→32, k=5) → BN → ReLU → MaxPool(2)
  Conv1d(32→64, k=5) → BN → ReLU → MaxPool(2)
  Conv1d(64→128, k=5) → BN → ReLU → MaxPool(2)
  GlobalAveragePool → Dropout(0.5)
  Linear(128→64) → ReLU → Linear(64→5)
```

### ECG_LSTM

```
Input (batch, seq_len=10, 129)
  Linear(129→64) → ReLU          # per-beat embedding
  BiLSTM(64, hidden=128, layers=2)  → output dim 256
  Additive Attention              → context (batch, 256)
  Dropout(0.5)
  Linear(256→128) → ReLU → Linear(128→5)
```

---

## Evaluation

Patient-aware splitting ensures **no patient appears in more than one split**,
preventing data leakage that inflates benchmark scores.

Reported metrics per model:
- Per-class F1, sensitivity (recall), specificity, PPV
- Macro / weighted F1
- AUC-ROC (one-vs-rest, macro)
- Confusion matrix (normalised)
- Precision-recall curves

---

## Running tests

```bash
# All tests (synthetic data only – no download required)
pytest tests/ -v

# Individual test modules
pytest tests/test_preprocessor.py -v
pytest tests/test_feature_extraction.py -v
pytest tests/test_models.py -v
```

---

## Explainability

```python
from src.inference import InferenceEngine

engine = InferenceEngine("models/cnn_best.pth", model_type="cnn")
engine.load_model()

pred_class, confidence, probs, attributions = engine.predict_with_explanation(beat)
```

Integrated Gradients attributions highlight which time-domain samples most
influenced the prediction. The `LSTMAttentionVisualizer` shows which beats in
a sequence the model focused on.

---

## License

MIT – see `LICENSE` for details.
