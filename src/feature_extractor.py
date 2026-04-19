"""
Hand-crafted ECG feature extraction for the baseline ML model.

:class:`FeatureExtractor` computes time-domain HRV metrics, morphological
beat descriptors, and delta-RR context features, returning a tidy
:class:`pandas.DataFrame` for downstream scikit-learn models.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_FS: int = 360


class FeatureExtractor:
    """Extracts hand-crafted features from segmented ECG beats.

    All methods are static; no instance state is required.
    """

    # ------------------------------------------------------------------ #
    #  RR-interval features                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_rr_intervals(
        r_peaks: np.ndarray,
        fs: int = _DEFAULT_FS,
    ) -> np.ndarray:
        """Compute RR intervals in milliseconds.

        Parameters
        ----------
        r_peaks:
            Sorted array of R-peak sample indices.
        fs:
            Sampling frequency in Hz.

        Returns
        -------
        np.ndarray
            RR intervals in ms (length = len(r_peaks) - 1).
        """
        if len(r_peaks) < 2:
            return np.array([], dtype=np.float32)
        rr_samples = np.diff(r_peaks.astype(np.float64))
        return (rr_samples / fs * 1000.0).astype(np.float32)

    @staticmethod
    def compute_hrv_metrics(rr_intervals: np.ndarray) -> dict:
        """Compute time-domain HRV statistics.

        Parameters
        ----------
        rr_intervals:
            Array of RR intervals in milliseconds.

        Returns
        -------
        dict with keys ``sdnn``, ``rmssd``, ``pnn50``, ``mean_rr``,
        ``median_rr``, ``min_rr``, ``max_rr``.
        """
        if len(rr_intervals) < 2:
            return {
                "sdnn": 0.0, "rmssd": 0.0, "pnn50": 0.0,
                "mean_rr": 0.0, "median_rr": 0.0,
                "min_rr": 0.0, "max_rr": 0.0,
            }
        diff_rr = np.diff(rr_intervals)
        sdnn = float(np.std(rr_intervals, ddof=1))
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
        pnn50 = float(np.mean(np.abs(diff_rr) > 50.0) * 100.0)
        return {
            "sdnn": sdnn,
            "rmssd": rmssd,
            "pnn50": pnn50,
            "mean_rr": float(np.mean(rr_intervals)),
            "median_rr": float(np.median(rr_intervals)),
            "min_rr": float(np.min(rr_intervals)),
            "max_rr": float(np.max(rr_intervals)),
        }

    # ------------------------------------------------------------------ #
    #  Morphological features                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_morphological_features(beat: np.ndarray) -> dict:
        """Extract morphological descriptors from a single beat waveform.

        Parameters
        ----------
        beat:
            1-D array representing one segmented beat (e.g. 129 samples).

        Returns
        -------
        dict with keys ``qrs_duration``, ``peak_to_peak``, ``beat_energy``,
        ``max_slope``, ``r_amplitude``, ``beat_mean``, ``beat_std``,
        ``beat_skewness``, ``beat_kurtosis``, ``waveform_length``.
        """
        n = len(beat)
        half = n // 2

        # QRS duration: consecutive samples above 50 % of peak amplitude
        r_amp = float(np.max(np.abs(beat)))
        threshold = 0.5 * r_amp
        above = (np.abs(beat) > threshold).astype(int)
        # Count longest run of above-threshold samples around the R-peak
        qrs_duration = _longest_run(above)

        peak_to_peak = float(np.max(beat) - np.min(beat))
        beat_energy = float(np.sum(beat ** 2))
        slope = np.abs(np.diff(beat))
        max_slope = float(np.max(slope)) if len(slope) > 0 else 0.0

        beat_mean = float(np.mean(beat))
        beat_std = float(np.std(beat))

        # Normalised central moments for shape description
        if beat_std > 1e-8:
            normalised = (beat - beat_mean) / beat_std
            beat_skewness = float(np.mean(normalised ** 3))
            beat_kurtosis = float(np.mean(normalised ** 4) - 3.0)
        else:
            beat_skewness = 0.0
            beat_kurtosis = 0.0

        # Waveform length (sum of absolute first differences)
        waveform_length = float(np.sum(np.abs(np.diff(beat))))

        return {
            "qrs_duration": qrs_duration,
            "peak_to_peak": peak_to_peak,
            "beat_energy": beat_energy,
            "max_slope": max_slope,
            "r_amplitude": r_amp,
            "beat_mean": beat_mean,
            "beat_std": beat_std,
            "beat_skewness": beat_skewness,
            "beat_kurtosis": beat_kurtosis,
            "waveform_length": waveform_length,
        }

    # ------------------------------------------------------------------ #
    #  Delta-RR context features                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_delta_rr(rr_intervals: np.ndarray) -> np.ndarray:
        """Compute successive differences of RR intervals (delta-RR).

        Parameters
        ----------
        rr_intervals:
            Array of RR intervals in ms.

        Returns
        -------
        np.ndarray
            Delta-RR values; length = max(0, len(rr_intervals) - 1).
        """
        if len(rr_intervals) < 2:
            return np.array([], dtype=np.float32)
        return np.diff(rr_intervals).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Per-beat feature vector                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_features_from_beat(
        beat: np.ndarray,
        rr_intervals: np.ndarray,
        idx: int,
    ) -> dict:
        """Combine all features for beat at position *idx*.

        Parameters
        ----------
        beat:
            Single beat waveform array.
        rr_intervals:
            All RR intervals for this record (ms).
        idx:
            Beat index within the record (0-based).

        Returns
        -------
        dict
            Flat dict of all feature values for this beat.
        """
        features: dict = {}

        # --- Morphological ---
        features.update(FeatureExtractor.compute_morphological_features(beat))

        # --- RR context ---
        # pre-RR: interval before this beat
        pre_rr = float(rr_intervals[idx - 1]) if idx > 0 and idx - 1 < len(rr_intervals) else 0.0
        # post-RR: interval after this beat
        post_rr = float(rr_intervals[idx]) if idx < len(rr_intervals) else 0.0
        # mean RR over a local window
        win_start = max(0, idx - 5)
        win_end = min(len(rr_intervals), idx + 5)
        local_rr = rr_intervals[win_start:win_end]
        local_mean_rr = float(np.mean(local_rr)) if len(local_rr) > 0 else 0.0

        features["pre_rr"] = pre_rr
        features["post_rr"] = post_rr
        features["local_mean_rr"] = local_mean_rr

        # Normalised RR (ratio to local mean)
        features["pre_rr_norm"] = pre_rr / local_mean_rr if local_mean_rr > 0 else 0.0
        features["post_rr_norm"] = post_rr / local_mean_rr if local_mean_rr > 0 else 0.0

        # delta-RR
        if idx > 0 and idx - 1 < len(rr_intervals) and idx < len(rr_intervals):
            delta = post_rr - pre_rr
        else:
            delta = 0.0
        features["delta_rr"] = delta

        return features

    # ------------------------------------------------------------------ #
    #  Full feature extraction for a set of beats                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_all_features(
        beats: np.ndarray,
        r_peaks: np.ndarray,
        fs: int = _DEFAULT_FS,
    ) -> pd.DataFrame:
        """Build a feature :class:`~pandas.DataFrame` for all beats in a record.

        Parameters
        ----------
        beats:
            Array of shape ``(n_beats, n_samples)`` with pre-processed beat
            waveforms.
        r_peaks:
            R-peak sample indices (length must match ``n_beats``).
        fs:
            Sampling frequency in Hz.

        Returns
        -------
        pd.DataFrame
            One row per beat, one column per feature.
        """
        if len(beats) == 0:
            logger.warning("No beats provided; returning empty DataFrame.")
            return pd.DataFrame()

        rr_intervals = FeatureExtractor.compute_rr_intervals(r_peaks, fs)

        rows: List[dict] = []
        for i, beat in enumerate(beats):
            row = FeatureExtractor.extract_features_from_beat(beat, rr_intervals, i)
            rows.append(row)

        df = pd.DataFrame(rows)
        # Replace any NaN/Inf that may arise from edge beats
        df.replace([np.inf, -np.inf], 0.0, inplace=True)
        df.fillna(0.0, inplace=True)
        return df


# ------------------------------------------------------------------ #
#  Module helpers                                                     #
# ------------------------------------------------------------------ #

def _longest_run(arr: np.ndarray) -> int:
    """Return the length of the longest contiguous run of ones in *arr*."""
    max_run = 0
    current = 0
    for v in arr:
        if v == 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run
