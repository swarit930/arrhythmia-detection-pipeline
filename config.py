"""
Configuration management for the arrhythmia detection pipeline.

Defines paths, signal processing parameters, AAMI label mappings,
model hyperparameters, and filter settings.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Directory paths (override via environment variables when needed)
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.environ.get("ECG_DATA_DIR", os.path.join(BASE_DIR, "data"))
MODEL_DIR: str = os.environ.get("ECG_MODEL_DIR", os.path.join(BASE_DIR, "models"))
RESULTS_DIR: str = os.environ.get("ECG_RESULTS_DIR", os.path.join(BASE_DIR, "results"))
LOG_DIR: str = os.environ.get("ECG_LOG_DIR", os.path.join(BASE_DIR, "logs"))

# ---------------------------------------------------------------------------
# Signal parameters
# ---------------------------------------------------------------------------
SAMPLING_RATE: int = 360  # Hz – native MIT-BIH sampling rate
BEAT_WINDOW_MS: int = 360  # milliseconds of signal centred on each R-peak
# Number of samples in one beat window: round(BEAT_WINDOW_MS / 1000 * SAMPLING_RATE)
BEAT_WINDOW_SAMPLES: int = 129  # 360 ms × 360 Hz / 1000 ≈ 129 samples
HALF_WINDOW: int = BEAT_WINDOW_SAMPLES // 2  # 64 samples either side of R-peak

# ---------------------------------------------------------------------------
# AAMI beat classes
# ---------------------------------------------------------------------------
AAMI_CLASSES: List[str] = ["N", "S", "V", "F", "Q"]
AAMI_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(AAMI_CLASSES)}
IDX_TO_AAMI_CLASS: Dict[int, str] = {i: c for i, c in enumerate(AAMI_CLASSES)}
NUM_CLASSES: int = len(AAMI_CLASSES)

# Mapping from MIT-BIH beat annotation symbols → AAMI class
MITBIH_TO_AAMI: Dict[str, str] = {
    # Normal beats → N
    "N": "N",   # Normal beat
    "L": "N",   # Left bundle branch block
    "R": "N",   # Right bundle branch block
    "e": "N",   # Atrial escape beat
    "j": "N",   # Nodal (junctional) escape beat
    # Supraventricular ectopic → S
    "A": "S",   # Atrial premature beat
    "a": "S",   # Aberrated atrial premature beat
    "J": "S",   # Nodal premature beat
    "S": "S",   # Supraventricular premature beat
    # Ventricular ectopic → V
    "V": "V",   # Premature ventricular contraction
    "E": "V",   # Ventricular escape beat
    # Fusion → F
    "F": "F",   # Fusion of ventricular and normal beat
    # Unknown / paced → Q
    "f": "Q",   # Fusion of paced and normal beat
    "Q": "Q",   # Unclassifiable beat
    "/": "Q",   # Paced beat
    "U": "Q",   # Unclassifiable beat
}

# ---------------------------------------------------------------------------
# MIT-BIH record list (all 48 records)
# ---------------------------------------------------------------------------
MITBIH_RECORDS: List[str] = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222", "223",
    "228", "230", "231", "232", "233", "234",
]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 256
EPOCHS: int = 100
EARLY_STOPPING_PATIENCE: int = 10
LR_SCHEDULER_PATIENCE: int = 5
LR_SCHEDULER_FACTOR: float = 0.5
WEIGHT_DECAY: float = 1e-4
DROPOUT_RATE: float = 0.5

# CNN specific
CNN_FILTERS: List[int] = [32, 64, 128]
CNN_KERNEL_SIZE: int = 5
CNN_POOL_SIZE: int = 2

# LSTM specific
LSTM_HIDDEN_SIZE: int = 128
LSTM_NUM_LAYERS: int = 2
LSTM_SEQ_LEN: int = 10  # number of consecutive beats per sequence
LSTM_BIDIRECTIONAL: bool = True

# Focal loss
FOCAL_LOSS_GAMMA: float = 2.0
FOCAL_LOSS_ALPHA: float = 1.0  # per-class weighting handled by compute_class_weights

# ---------------------------------------------------------------------------
# Filter configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class FilterConfig:
    """Bandpass filter settings for ECG pre-processing."""
    low_cutoff: float = 0.5   # Hz – high-pass corner (removes baseline wander)
    high_cutoff: float = 45.0  # Hz – low-pass corner (removes high-freq noise)
    filter_order: int = 4      # Butterworth filter order

    def validate(self, fs: int = SAMPLING_RATE) -> None:
        """Raise ValueError if settings are incompatible with sampling rate."""
        nyq = fs / 2.0
        if self.low_cutoff <= 0:
            raise ValueError("low_cutoff must be > 0")
        if self.high_cutoff >= nyq:
            raise ValueError(f"high_cutoff must be < Nyquist ({nyq} Hz)")
        if self.low_cutoff >= self.high_cutoff:
            raise ValueError("low_cutoff must be < high_cutoff")


DEFAULT_FILTER_CONFIG: FilterConfig = FilterConfig()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Patient-level split ratios
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.1
