"""
General-purpose utilities for the arrhythmia detection pipeline.

Provides:
* :func:`setup_logging` – consistent log format across scripts.
* :func:`set_seed` – reproducibility across numpy / torch / random.
* :func:`ensure_dir` – create directories on demand.
* :func:`save_json` / :func:`load_json` – JSON helpers.
* :func:`plot_ecg_signal` / :func:`plot_beat` – quick visualisation helpers.
* :class:`AverageMeter` – tracks a running average (used in training loops).
* :class:`EarlyStopping` – patience-based early stopping.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Logging                                                                 #
# ======================================================================= #

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger with a standardised format.

    Parameters
    ----------
    log_level:
        Logging level string: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, …
    log_file:
        Optional file path; if provided a :class:`~logging.FileHandler` is
        added in addition to the console handler.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list = [logging.StreamHandler()]
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
    logger.debug("Logging configured at level %s.", log_level)


# ======================================================================= #
#  Reproducibility                                                         #
# ======================================================================= #

def set_seed(seed: int = 42) -> None:
    """Set random seeds for :mod:`random`, :mod:`numpy`, and *torch*.

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic operations (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.debug("Random seed set to %d.", seed)


# ======================================================================= #
#  File / directory utilities                                              #
# ======================================================================= #

def ensure_dir(path: str) -> None:
    """Create *path* (and any parents) if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to create.
    """
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Serialise *data* to a JSON file at *path*.

    Parameters
    ----------
    data:
        JSON-serialisable Python object.
    path:
        Destination file path.
    indent:
        JSON indentation width.
    """
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=_json_default)
    logger.debug("JSON saved to %s.", path)


def load_json(path: str) -> Any:
    """Load and return the contents of a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    Any
        Deserialised Python object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")


# ======================================================================= #
#  Plotting helpers                                                        #
# ======================================================================= #

def plot_ecg_signal(
    signal: np.ndarray,
    fs: int = 360,
    r_peaks: Optional[np.ndarray] = None,
    title: str = "ECG Signal",
    save_path: Optional[str] = None,
) -> None:
    """Plot a full ECG signal with optional R-peak markers.

    Parameters
    ----------
    signal:
        1-D array of ECG samples.
    fs:
        Sampling frequency in Hz (used to convert sample indices to seconds).
    r_peaks:
        Optional array of R-peak sample indices to annotate.
    title:
        Figure title.
    save_path:
        If provided, save the figure to this path.
    """
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(t, signal, linewidth=0.8, color="steelblue")
    if r_peaks is not None and len(r_peaks) > 0:
        ax.scatter(
            r_peaks / fs,
            signal[r_peaks],
            color="crimson",
            s=20,
            zorder=5,
            label="R-peaks",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(os.path.abspath(save_path)))
        fig.savefig(save_path, dpi=150)
        logger.info("ECG plot saved to %s.", save_path)
    plt.close(fig)


def plot_beat(
    beat: np.ndarray,
    title: str = "ECG Beat",
    save_path: Optional[str] = None,
) -> None:
    """Plot a single segmented beat waveform.

    Parameters
    ----------
    beat:
        1-D array of beat samples.
    title:
        Figure title.
    save_path:
        Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(beat, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(os.path.abspath(save_path)))
        fig.savefig(save_path, dpi=150)
        logger.info("Beat plot saved to %s.", save_path)
    plt.close(fig)


# ======================================================================= #
#  AverageMeter                                                            #
# ======================================================================= #

class AverageMeter:
    """Track the running mean of a scalar quantity (e.g. batch loss).

    Example
    -------
    >>> meter = AverageMeter()
    >>> meter.update(0.5, n=32)
    >>> meter.update(0.3, n=32)
    >>> meter.avg
    0.4
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the meter to an empty state."""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """Accumulate *val* weighted by *n* observations.

        Parameters
        ----------
        val:
            Value to accumulate (e.g. mean loss of a mini-batch).
        n:
            Number of observations the value represents (e.g. batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter(avg={self.avg:.4f}, count={self.count})"


# ======================================================================= #
#  EarlyStopping                                                           #
# ======================================================================= #

class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience:
        Number of epochs with no improvement before triggering stop.
    min_delta:
        Minimum absolute change to qualify as an improvement.
    verbose:
        Log a message whenever the counter increments.

    Attributes
    ----------
    should_stop : bool
        Set to ``True`` when training should stop.
    best_score : float
        Best validation score seen so far.
    counter : int
        Number of epochs since last improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter: int = 0
        self.best_score: float = float("inf")
        self.should_stop: bool = False

    def __call__(self, val_loss: float) -> None:
        """Update state based on the latest validation loss.

        Parameters
        ----------
        val_loss:
            Validation loss for the current epoch (lower is better).
        """
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.debug(
                    "EarlyStopping counter: %d / %d", self.counter, self.patience
                )
            if self.counter >= self.patience:
                self.should_stop = True

    def reset(self) -> None:
        """Reset to initial state (e.g. for a new training run)."""
        self.counter = 0
        self.best_score = float("inf")
        self.should_stop = False
