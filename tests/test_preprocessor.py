"""
Unit tests for src/preprocessor.py.

All tests use synthetically generated ECG data so no real MIT-BIH download
is required.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import ECGPreprocessor

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FS = 360
DURATION_S = 10


def make_synthetic_ecg(fs: int = FS, duration_s: int = DURATION_S, seed: int = 0) -> np.ndarray:
    """Return a synthetic ECG-like signal with clear QRS peaks."""
    rng = np.random.RandomState(seed)
    n = duration_s * fs
    t = np.linspace(0, duration_s, n)
    # Heartbeat template at 72 bpm
    hr_hz = 1.2
    ecg = np.zeros(n)
    beat_period = int(fs / hr_hz)
    half = beat_period // 2

    t_beat = np.arange(-half, half + 1)
    qrs = (
        1.0 * np.exp(-0.5 * (t_beat / (0.02 * fs)) ** 2)
        + 0.2 * np.exp(-0.5 * ((t_beat - int(0.06 * fs)) / (0.03 * fs)) ** 2)
        - 0.15 * np.exp(-0.5 * ((t_beat + int(0.04 * fs)) / (0.02 * fs)) ** 2)
    )

    idx = half
    while idx + half < n:
        start = idx - half
        end = idx + half + 1
        ecg[start:end] += qrs
        idx += beat_period

    # Add noise and baseline wander
    noise = rng.normal(0, 0.03, n)
    wander = 0.15 * np.sin(2 * np.pi * 0.2 * t)
    return (ecg + noise + wander).astype(np.float32)


# ---------------------------------------------------------------------------
# test_bandpass_filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    def test_output_shape(self):
        """Filtered signal must have the same length as input."""
        signal = make_synthetic_ecg()
        filtered = ECGPreprocessor.bandpass_filter(signal, fs=FS)
        assert filtered.shape == signal.shape

    def test_output_dtype(self):
        """Filtered signal must be float32."""
        signal = make_synthetic_ecg()
        filtered = ECGPreprocessor.bandpass_filter(signal, fs=FS)
        assert filtered.dtype == np.float32

    def test_attenuates_high_frequency(self):
        """A 100 Hz sine should be attenuated by the 45 Hz low-pass."""
        n = 3600
        t = np.arange(n) / FS
        high_freq = np.sin(2 * np.pi * 100 * t).astype(np.float32)
        filtered = ECGPreprocessor.bandpass_filter(high_freq, fs=FS, high=45.0)
        # The output power should be much less than the input
        assert np.var(filtered) < 0.05 * np.var(high_freq)

    def test_passes_qrs_band(self):
        """A 10 Hz sine (within the passband) should pass through with little attenuation."""
        n = 3600
        t = np.arange(n) / FS
        mid_freq = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        filtered = ECGPreprocessor.bandpass_filter(mid_freq, fs=FS)
        # Power should be preserved reasonably well
        assert np.var(filtered) > 0.3 * np.var(mid_freq)


# ---------------------------------------------------------------------------
# test_baseline_wander_removal
# ---------------------------------------------------------------------------

class TestBaselineWanderRemoval:
    def test_output_shape(self):
        signal = make_synthetic_ecg()
        output = ECGPreprocessor.remove_baseline_wander(signal, fs=FS)
        assert output.shape == signal.shape

    def test_reduces_dc_offset(self):
        """After baseline removal the mean should be close to zero."""
        n = 3600
        dc = np.full(n, 0.5, dtype=np.float32)
        output = ECGPreprocessor.remove_baseline_wander(dc, fs=FS)
        # medfilt of a constant returns the constant; subtract → zero
        assert np.abs(np.mean(output)) < 1e-3

    def test_attenuates_low_frequency_wander(self):
        """A 0.1 Hz sine (baseline wander) should be suppressed."""
        n = FS * 20
        t = np.arange(n) / FS
        wander = 0.5 * np.sin(2 * np.pi * 0.1 * t).astype(np.float32)
        output = ECGPreprocessor.remove_baseline_wander(wander, fs=FS)
        assert np.std(output) < 0.3 * np.std(wander)


# ---------------------------------------------------------------------------
# test_normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_zero_mean(self):
        signal = make_synthetic_ecg()
        normed = ECGPreprocessor.normalize_signal(signal)
        assert abs(float(np.mean(normed))) < 1e-5

    def test_unit_std(self):
        signal = make_synthetic_ecg()
        normed = ECGPreprocessor.normalize_signal(signal)
        assert abs(float(np.std(normed)) - 1.0) < 1e-3

    def test_constant_signal(self):
        """A constant signal should return zeros without crashing."""
        signal = np.ones(500, dtype=np.float32)
        normed = ECGPreprocessor.normalize_signal(signal)
        assert np.all(normed == 0.0)

    def test_output_dtype(self):
        signal = make_synthetic_ecg()
        normed = ECGPreprocessor.normalize_signal(signal)
        assert normed.dtype == np.float32


# ---------------------------------------------------------------------------
# test_r_peak_detection
# ---------------------------------------------------------------------------

class TestRPeakDetection:
    def test_detects_peaks(self):
        """At least one R-peak should be detected in a 10-second synthetic ECG."""
        signal = make_synthetic_ecg()
        clean = ECGPreprocessor.preprocess_record(signal, fs=FS)
        r_peaks = ECGPreprocessor.detect_r_peaks(clean, fs=FS)
        # 10 s at 72 bpm → ~12 peaks; allow loose bounds
        assert len(r_peaks) >= 5

    def test_peaks_within_bounds(self):
        """All R-peak indices must be within the signal length."""
        signal = make_synthetic_ecg()
        clean = ECGPreprocessor.preprocess_record(signal, fs=FS)
        r_peaks = ECGPreprocessor.detect_r_peaks(clean, fs=FS)
        assert np.all(r_peaks >= 0)
        assert np.all(r_peaks < len(clean))

    def test_peaks_sorted(self):
        """R-peak indices should be monotonically increasing."""
        signal = make_synthetic_ecg()
        clean = ECGPreprocessor.preprocess_record(signal, fs=FS)
        r_peaks = ECGPreprocessor.detect_r_peaks(clean, fs=FS)
        if len(r_peaks) > 1:
            assert np.all(np.diff(r_peaks) > 0)

    def test_empty_signal(self):
        """A flat zero signal should not crash (may return 0 peaks)."""
        signal = np.zeros(3600, dtype=np.float32)
        r_peaks = ECGPreprocessor.detect_r_peaks(signal, fs=FS)
        assert isinstance(r_peaks, np.ndarray)


# ---------------------------------------------------------------------------
# test_beat_segmentation
# ---------------------------------------------------------------------------

class TestBeatSegmentation:
    def setup_method(self):
        signal = make_synthetic_ecg()
        self.clean = ECGPreprocessor.preprocess_record(signal, fs=FS)
        self.r_peaks = ECGPreprocessor.detect_r_peaks(self.clean, fs=FS)

    def test_beat_shape(self):
        """Each beat must have the expected window length (129 samples)."""
        beats, valid_idx = ECGPreprocessor.segment_beats(self.clean, self.r_peaks, fs=FS)
        if len(beats) > 0:
            assert beats.shape[1] == 129

    def test_valid_indices_length(self):
        """valid_indices length must equal number of returned beats."""
        beats, valid_idx = ECGPreprocessor.segment_beats(self.clean, self.r_peaks, fs=FS)
        assert len(beats) == len(valid_idx)

    def test_beats_not_empty(self):
        """Should extract at least a few beats from a 10-second signal."""
        beats, valid_idx = ECGPreprocessor.segment_beats(self.clean, self.r_peaks, fs=FS)
        assert len(beats) >= 3

    def test_no_boundary_beats(self):
        """Beats too close to the signal edges should be excluded."""
        n = len(self.clean)
        beats, valid_idx = ECGPreprocessor.segment_beats(self.clean, self.r_peaks, fs=FS)
        if len(beats) > 0:
            half = 64
            assert np.all(self.r_peaks[valid_idx] >= half)
            assert np.all(self.r_peaks[valid_idx] + half + 1 <= n)

    def test_custom_window(self):
        """A custom 200 ms window should produce beats of the correct length."""
        beats, _ = ECGPreprocessor.segment_beats(
            self.clean, self.r_peaks, fs=FS, window_ms=200
        )
        expected_len = 2 * int(100 * FS / 1000) + 1  # 2*36+1=73
        if len(beats) > 0:
            assert beats.shape[1] == expected_len
