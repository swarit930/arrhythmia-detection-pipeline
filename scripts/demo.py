"""
demo.py – End-to-end demonstration of the arrhythmia detection pipeline.

Steps
-----
1. Generate (or load) a single ECG record.
2. Pre-process the signal.
3. Detect R-peaks and segment beats.
4. Predict each beat with the CNN model.
5. Visualise the ECG, beats, and an explanation for the most confident beat.

Usage
-----
    python scripts/demo.py [--model-path models/cnn_best.pth]
                           [--record 100]
                           [--data-dir data/]
                           [--output-dir results/demo/]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.preprocessor import ECGPreprocessor
from src.utils import setup_logging, ensure_dir, plot_ecg_signal, plot_beat

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arrhythmia detection demo.")
    parser.add_argument("--model-path", default=os.path.join(config.MODEL_DIR, "cnn_best.pth"))
    parser.add_argument("--record", default="100", help="MIT-BIH record name")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--output-dir", default=os.path.join(config.RESULTS_DIR, "demo"))
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Use a synthetic ECG instead of loading a real record (no wfdb required).",
    )
    return parser.parse_args()


def generate_synthetic_ecg(duration_s: int = 10, fs: int = 360) -> np.ndarray:
    """Generate a simple synthetic ECG-like signal for demonstration."""
    n = duration_s * fs
    t = np.linspace(0, duration_s, n)
    # Heartbeat template (QRS-like shape)
    hr_hz = 1.2  # 72 bpm
    ecg = np.zeros(n)
    beat_period = int(fs / hr_hz)
    half = beat_period // 2

    # Gaussian QRS complex template
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

    # Add light noise and baseline wander
    noise = np.random.RandomState(0).normal(0, 0.02, n)
    wander = 0.1 * np.sin(2 * np.pi * 0.25 * t)
    return (ecg + noise + wander).astype(np.float32)


def load_record_signal(record: str, data_dir: str) -> tuple[np.ndarray, int]:
    """Load an MIT-BIH record. Download if necessary."""
    from src.data_loader import ECGDataLoader

    record_path = os.path.join(data_dir, record)
    try:
        ECGDataLoader.download_record(record, data_dir)
        signal, fields, _ = ECGDataLoader.load_record(record, data_dir)
        return signal, fields["fs"]
    except Exception as exc:
        logger.warning("Could not load record %s (%s) – using synthetic signal.", record, exc)
        return generate_synthetic_ecg().reshape(-1, 1), config.SAMPLING_RATE


def run_demo(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    ensure_dir(args.output_dir)
    np.random.seed(config.RANDOM_SEED)

    # ------------------------------------------------------------------
    # Step 1: Load signal
    # ------------------------------------------------------------------
    if args.synthetic:
        logger.info("Generating synthetic ECG …")
        raw_signal = generate_synthetic_ecg().reshape(-1, 1)
        fs = config.SAMPLING_RATE
    else:
        logger.info("Loading record %s from %s …", args.record, args.data_dir)
        raw_signal, fs = load_record_signal(args.record, args.data_dir)

    logger.info("Signal shape: %s, fs=%d Hz", raw_signal.shape, fs)

    # ------------------------------------------------------------------
    # Step 2: Pre-process
    # ------------------------------------------------------------------
    logger.info("Pre-processing signal …")
    clean_signal = ECGPreprocessor.preprocess_record(raw_signal, fs=fs)
    plot_ecg_signal(
        clean_signal, fs=fs,
        title="Pre-processed ECG",
        save_path=os.path.join(args.output_dir, "ecg_clean.png"),
    )
    logger.info("Saved clean ECG plot.")

    # ------------------------------------------------------------------
    # Step 3: R-peak detection + beat segmentation
    # ------------------------------------------------------------------
    logger.info("Detecting R-peaks …")
    r_peaks = ECGPreprocessor.detect_r_peaks(clean_signal, fs=fs)
    logger.info("Found %d R-peaks.", len(r_peaks))

    plot_ecg_signal(
        clean_signal, fs=fs, r_peaks=r_peaks,
        title="ECG with R-peaks",
        save_path=os.path.join(args.output_dir, "ecg_rpeaks.png"),
    )

    beats, valid_idx = ECGPreprocessor.segment_beats(clean_signal, r_peaks, fs=fs)
    logger.info("Segmented %d beats.", len(beats))

    if len(beats) == 0:
        logger.error("No beats were segmented; cannot proceed with classification.")
        return

    # Plot a sample of beats
    n_show = min(5, len(beats))
    for i in range(n_show):
        plot_beat(
            beats[i],
            title=f"Beat {i}",
            save_path=os.path.join(args.output_dir, f"beat_{i}.png"),
        )

    # ------------------------------------------------------------------
    # Step 4: Classification
    # ------------------------------------------------------------------
    model_exists = os.path.exists(args.model_path)
    if model_exists:
        from src.inference import InferenceEngine

        logger.info("Loading CNN model from %s …", args.model_path)
        engine = InferenceEngine(args.model_path, model_type="cnn")
        engine.load_model()

        predictions = []
        for beat in beats:
            pred_class, conf, _ = engine.predict_beat(beat)
            predictions.append((pred_class, conf))

        class_names = config.AAMI_CLASSES
        for i, (cls, conf) in enumerate(predictions[:10]):
            logger.info("Beat %2d → %s (confidence=%.3f)", i, class_names[cls], conf)

        # ------------------------------------------------------------------
        # Step 5: Explanation for the most confident beat
        # ------------------------------------------------------------------
        best_idx = int(np.argmax([c for _, c in predictions]))
        best_beat = beats[best_idx]
        logger.info("Generating explanation for beat %d …", best_idx)

        pred_class, confidence, _, attributions = engine.predict_with_explanation(best_beat)

        from src.explainability import GradientExplainer
        explainer = GradientExplainer(engine.model, engine.device)
        explainer.visualize_explanation(
            best_beat,
            attributions,
            predicted_class=pred_class,
            confidence=confidence,
            class_names=class_names,
            save_path=os.path.join(args.output_dir, "explanation.png"),
        )
        logger.info("Explanation saved.")
    else:
        logger.warning(
            "Model checkpoint not found at %s. Skipping classification step. "
            "Run train_cnn.py first to generate a trained model.",
            args.model_path,
        )

    logger.info("Demo complete. Outputs saved to %s", args.output_dir)


if __name__ == "__main__":
    run_demo(parse_args())
