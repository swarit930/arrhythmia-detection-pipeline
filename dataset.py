from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import config
from src.data_loader import ECGDataLoader
from src.preprocessor import ECGPreprocessor

DEFAULT_RR_INTERVAL_SECONDS = 0.8


class SignalAugment:
    """Beat-level ECG augmentation for robust minority-class generalization."""

    def __init__(
        self,
        scaling_range: Tuple[float, float] = (0.9, 1.1),
        baseline_wander_amplitude: float = 0.08,
        baseline_freq_range: Tuple[float, float] = (0.05, 0.5),
        stretch_range: Tuple[float, float] = (0.95, 1.05),
    ) -> None:
        self.scaling_range = scaling_range
        self.baseline_wander_amplitude = baseline_wander_amplitude
        self.baseline_freq_range = baseline_freq_range
        self.stretch_range = stretch_range

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        x = signal.astype(np.float32, copy=True)
        x = self._random_scaling(x)
        x = self._baseline_shift(x)
        x = self._time_stretch(x)
        return x.astype(np.float32)

    def _random_scaling(self, signal: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.scaling_range[0], self.scaling_range[1])
        return signal * np.float32(scale)

    def _baseline_shift(self, signal: np.ndarray) -> np.ndarray:
        n = signal.shape[0]
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        freq = np.random.uniform(self.baseline_freq_range[0], self.baseline_freq_range[1])
        phase = np.random.uniform(0.0, 2.0 * np.pi)
        amp = self.baseline_wander_amplitude * (np.std(signal) + 1e-6)
        wander = amp * np.sin(2.0 * np.pi * freq * t + phase)
        return signal + wander.astype(np.float32)

    def _time_stretch(self, signal: np.ndarray) -> np.ndarray:
        n = signal.shape[0]
        stretch = np.random.uniform(self.stretch_range[0], self.stretch_range[1])
        stretched_n = max(2, int(round(n * stretch)))
        src_idx = np.arange(n, dtype=np.float32)
        stretched_idx = np.linspace(0, n - 1, stretched_n, dtype=np.float32)
        stretched = np.interp(stretched_idx, src_idx, signal).astype(np.float32)
        out_idx = np.linspace(0, stretched_n - 1, n, dtype=np.float32)
        return np.interp(out_idx, np.arange(stretched_n, dtype=np.float32), stretched).astype(np.float32)


def compute_rr_features(r_peaks: np.ndarray, fs: float, local_window: int = 10) -> np.ndarray:
    """Return beat timing context in seconds as [pre_RR, post_RR, local_RR_avg].

    pre_RR:
        Interval from previous beat to current beat.
    post_RR:
        Interval from current beat to next beat.
    local_RR_avg:
        Mean RR interval over a rolling local window of recent beats.
    local_window:
        Number of recent beats used to compute local_RR_avg.
    """
    n = len(r_peaks)
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)

    r_peaks = r_peaks.astype(np.float64)
    rr_intervals = np.diff(r_peaks) / float(fs)  # length n-1
    global_rr = float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else DEFAULT_RR_INTERVAL_SECONDS

    pre_rr = np.full(n, global_rr, dtype=np.float32)
    post_rr = np.full(n, global_rr, dtype=np.float32)
    local_rr_avg = np.full(n, global_rr, dtype=np.float32)

    for i in range(n):
        if i > 0:
            pre_rr[i] = np.float32((r_peaks[i] - r_peaks[i - 1]) / float(fs))
        if i < n - 1:
            post_rr[i] = np.float32((r_peaks[i + 1] - r_peaks[i]) / float(fs))

        start = max(1, i - local_window + 1)
        if i >= 1:
            local_slice = (r_peaks[start : i + 1] - r_peaks[start - 1 : i]) / float(fs)
            local_rr_avg[i] = np.float32(np.mean(local_slice)) if len(local_slice) > 0 else np.float32(global_rr)

    return np.stack([pre_rr, post_rr, local_rr_avg], axis=1).astype(np.float32)


@dataclass
class PreparedData:
    beats: np.ndarray
    rr_features: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray


def collect_beats_with_rr_features(records: Sequence[str], data_dir: str) -> PreparedData:
    beats_all: List[np.ndarray] = []
    rr_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    rec_all: List[np.ndarray] = []

    for rec_name, signal, fields, annotation in ECGDataLoader.load_all_records(list(records), data_dir):
        if annotation is None:
            continue

        fs = fields["fs"]
        clean = ECGPreprocessor.preprocess_record(signal, fs=fs)

        ann_samples = np.asarray(annotation.sample)
        ann_symbols = np.asarray(annotation.symbol)

        mapped_peaks: List[int] = []
        mapped_labels: List[int] = []
        for sym, samp in zip(ann_symbols, ann_samples):
            aami_class = config.MITBIH_TO_AAMI.get(sym)
            if aami_class is None:
                continue
            mapped_peaks.append(int(samp))
            mapped_labels.append(int(config.AAMI_CLASS_TO_IDX[aami_class]))

        if len(mapped_peaks) == 0:
            continue

        r_peaks = np.asarray(mapped_peaks, dtype=np.int64)
        labels = np.asarray(mapped_labels, dtype=np.int64)
        rr_features_all = compute_rr_features(r_peaks, fs=float(fs), local_window=10)

        beats, valid_idx = ECGPreprocessor.segment_beats(clean, r_peaks, fs=fs)
        if len(beats) == 0:
            continue

        beats_all.append(beats.astype(np.float32))
        labels_all.append(labels[valid_idx].astype(np.int64))
        rr_all.append(rr_features_all[valid_idx].astype(np.float32))
        rec_all.append(np.full(len(valid_idx), rec_name))

    if len(beats_all) == 0:
        raise RuntimeError("No valid beats extracted from provided records.")

    return PreparedData(
        beats=np.concatenate(beats_all, axis=0),
        rr_features=np.concatenate(rr_all, axis=0),
        labels=np.concatenate(labels_all, axis=0),
        record_ids=np.concatenate(rec_all, axis=0),
    )


class ArrhythmiaDataset(Dataset):
    """Dataset yielding waveform, RR-context features, and label."""

    def __init__(
        self,
        beats: np.ndarray,
        rr_features: np.ndarray,
        labels: np.ndarray,
        record_ids: np.ndarray | None = None,
        augment: SignalAugment | None = None,
    ) -> None:
        self.beats = beats.astype(np.float32)
        self.rr_features = rr_features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.record_ids = record_ids
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        beat = self.beats[idx]
        if self.augment is not None:
            beat = self.augment(beat)
        beat_t = torch.from_numpy(beat).unsqueeze(0)  # (1, L)
        rr_t = torch.from_numpy(self.rr_features[idx])  # (3,)
        y_t = torch.tensor(self.labels[idx], dtype=torch.long)
        return beat_t, rr_t, y_t


def build_weighted_sampler(labels: np.ndarray, num_classes: int = config.NUM_CLASSES) -> WeightedRandomSampler:
    """Create inverse-frequency WeightedRandomSampler for class balancing."""
    labels = labels.astype(np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0.0] = 1.0
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]
    sample_weights_t = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=sample_weights_t, num_samples=len(sample_weights_t), replacement=True)
