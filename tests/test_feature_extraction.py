"""
Unit tests for src/feature_extractor.py.

All tests use synthetically generated beat data; no real ECG download required.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import FeatureExtractor, _longest_run

FS = 360
BEAT_LEN = 129


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_beats(n: int = 50, beat_len: int = BEAT_LEN, seed: int = 0) -> np.ndarray:
    """Return random synthetic beats shaped (n, beat_len)."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, beat_len).astype(np.float32)


def make_r_peaks(n: int = 50, fs: int = FS, start: int = 1000) -> np.ndarray:
    """Return evenly spaced R-peak indices simulating 72 bpm."""
    spacing = int(fs / 1.2)  # 72 bpm ≈ 300 samples at 360 Hz
    return np.array([start + i * spacing for i in range(n)], dtype=np.int64)


# ---------------------------------------------------------------------------
# test_rr_intervals
# ---------------------------------------------------------------------------

class TestRRIntervals:
    def test_basic_length(self):
        """RR intervals should have length n_peaks - 1."""
        r_peaks = make_r_peaks(10)
        rr = FeatureExtractor.compute_rr_intervals(r_peaks, fs=FS)
        assert len(rr) == 9

    def test_regular_spacing(self):
        """Evenly spaced peaks at 300 samples → RR ≈ 833 ms at 360 Hz."""
        r_peaks = make_r_peaks(5, fs=FS)
        rr = FeatureExtractor.compute_rr_intervals(r_peaks, fs=FS)
        expected = 300 / FS * 1000  # ms
        assert np.allclose(rr, expected, atol=1.0)

    def test_single_peak(self):
        """A single peak → empty RR array."""
        r_peaks = np.array([500], dtype=np.int64)
        rr = FeatureExtractor.compute_rr_intervals(r_peaks, fs=FS)
        assert len(rr) == 0

    def test_output_dtype(self):
        rr = FeatureExtractor.compute_rr_intervals(make_r_peaks(5), fs=FS)
        assert rr.dtype == np.float32

    def test_all_positive(self):
        """All RR intervals must be positive."""
        rr = FeatureExtractor.compute_rr_intervals(make_r_peaks(20), fs=FS)
        assert np.all(rr > 0)


# ---------------------------------------------------------------------------
# test_hrv_metrics
# ---------------------------------------------------------------------------

class TestHRVMetrics:
    def test_keys_present(self):
        rr = FeatureExtractor.compute_rr_intervals(make_r_peaks(20), fs=FS)
        metrics = FeatureExtractor.compute_hrv_metrics(rr)
        for key in ("sdnn", "rmssd", "pnn50", "mean_rr", "median_rr", "min_rr", "max_rr"):
            assert key in metrics

    def test_regular_rhythm_low_hrv(self):
        """Perfectly regular rhythm → SDNN=0, RMSSD=0, pNN50=0."""
        r_peaks = make_r_peaks(30)
        rr = FeatureExtractor.compute_rr_intervals(r_peaks, fs=FS)
        metrics = FeatureExtractor.compute_hrv_metrics(rr)
        assert metrics["sdnn"] < 1e-3
        assert metrics["rmssd"] < 1e-3
        assert metrics["pnn50"] == pytest.approx(0.0)

    def test_short_rr(self):
        """Single RR interval → all zeros (graceful fallback)."""
        metrics = FeatureExtractor.compute_hrv_metrics(np.array([800.0], dtype=np.float32))
        assert metrics["sdnn"] == 0.0

    def test_mean_rr_value(self):
        """Mean RR should match simple numpy mean."""
        rr = np.array([800.0, 820.0, 810.0, 790.0], dtype=np.float32)
        metrics = FeatureExtractor.compute_hrv_metrics(rr)
        assert abs(metrics["mean_rr"] - float(np.mean(rr))) < 1e-4


# ---------------------------------------------------------------------------
# test_morphological_features
# ---------------------------------------------------------------------------

class TestMorphologicalFeatures:
    def test_keys_present(self):
        beat = np.random.randn(BEAT_LEN).astype(np.float32)
        feats = FeatureExtractor.compute_morphological_features(beat)
        for key in (
            "qrs_duration", "peak_to_peak", "beat_energy", "max_slope",
            "r_amplitude", "beat_mean", "beat_std", "beat_skewness",
            "beat_kurtosis", "waveform_length",
        ):
            assert key in feats

    def test_constant_beat(self):
        """A constant beat (all zeros) should not crash and return zero energy."""
        beat = np.zeros(BEAT_LEN, dtype=np.float32)
        feats = FeatureExtractor.compute_morphological_features(beat)
        assert feats["beat_energy"] == pytest.approx(0.0)
        assert feats["peak_to_peak"] == pytest.approx(0.0)

    def test_peak_to_peak_positive(self):
        """Peak-to-peak amplitude must be non-negative."""
        for seed in range(10):
            beat = np.random.RandomState(seed).randn(BEAT_LEN).astype(np.float32)
            feats = FeatureExtractor.compute_morphological_features(beat)
            assert feats["peak_to_peak"] >= 0.0

    def test_r_amplitude_matches(self):
        """R amplitude should equal max absolute value."""
        beat = np.arange(BEAT_LEN, dtype=np.float32) - BEAT_LEN // 2
        feats = FeatureExtractor.compute_morphological_features(beat)
        expected = float(np.max(np.abs(beat)))
        assert feats["r_amplitude"] == pytest.approx(expected)

    def test_waveform_length_positive(self):
        beat = np.random.randn(BEAT_LEN).astype(np.float32)
        feats = FeatureExtractor.compute_morphological_features(beat)
        assert feats["waveform_length"] > 0.0


# ---------------------------------------------------------------------------
# test_delta_rr
# ---------------------------------------------------------------------------

class TestDeltaRR:
    def test_length(self):
        rr = np.array([800.0, 820.0, 810.0, 790.0], dtype=np.float32)
        delta = FeatureExtractor.compute_delta_rr(rr)
        assert len(delta) == len(rr) - 1

    def test_values(self):
        rr = np.array([800.0, 900.0, 700.0], dtype=np.float32)
        delta = FeatureExtractor.compute_delta_rr(rr)
        assert delta[0] == pytest.approx(100.0)
        assert delta[1] == pytest.approx(-200.0)

    def test_single_element(self):
        rr = np.array([800.0], dtype=np.float32)
        delta = FeatureExtractor.compute_delta_rr(rr)
        assert len(delta) == 0

    def test_empty(self):
        delta = FeatureExtractor.compute_delta_rr(np.array([], dtype=np.float32))
        assert len(delta) == 0


# ---------------------------------------------------------------------------
# test_full_feature_extraction
# ---------------------------------------------------------------------------

class TestFullFeatureExtraction:
    def test_returns_dataframe(self):
        beats = make_beats(20)
        r_peaks = make_r_peaks(20)
        df = FeatureExtractor.extract_all_features(beats, r_peaks, fs=FS)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        n = 30
        beats = make_beats(n)
        r_peaks = make_r_peaks(n)
        df = FeatureExtractor.extract_all_features(beats, r_peaks, fs=FS)
        assert len(df) == n

    def test_no_nan(self):
        beats = make_beats(20)
        r_peaks = make_r_peaks(20)
        df = FeatureExtractor.extract_all_features(beats, r_peaks, fs=FS)
        assert not df.isnull().any().any()

    def test_no_inf(self):
        beats = make_beats(20)
        r_peaks = make_r_peaks(20)
        df = FeatureExtractor.extract_all_features(beats, r_peaks, fs=FS)
        assert not np.isinf(df.values).any()

    def test_empty_beats(self):
        """Empty input should return an empty DataFrame without error."""
        df = FeatureExtractor.extract_all_features(
            np.empty((0, BEAT_LEN), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            fs=FS,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestLongestRun:
    def test_basic(self):
        assert _longest_run(np.array([1, 1, 0, 1, 1, 1, 0])) == 3

    def test_all_ones(self):
        assert _longest_run(np.ones(5, dtype=int)) == 5

    def test_all_zeros(self):
        assert _longest_run(np.zeros(5, dtype=int)) == 0

    def test_empty(self):
        assert _longest_run(np.array([], dtype=int)) == 0
