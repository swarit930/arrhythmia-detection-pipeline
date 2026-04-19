"""
ECG data loading utilities for the MIT-BIH Arrhythmia Database.

Provides :class:`ECGDataLoader` which wraps *wfdb* to download, read, and
iterate over records from the PhysioNet MIT-BIH dataset.
"""

from __future__ import annotations

import logging
import os
from typing import Generator, List, Optional, Tuple

import numpy as np
import wfdb

logger = logging.getLogger(__name__)


class ECGDataLoader:
    """Utility class for loading MIT-BIH Arrhythmia Database records.

    All methods are static/class-level so no instantiation is required,
    though you may instantiate the class if that is more convenient.
    """

    # PhysioNet database identifier used by wfdb
    DB_NAME: str = "mitdb"

    # ------------------------------------------------------------------ #
    #  Downloading                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def download_record(record_name: str, data_dir: str) -> None:
        """Download a single MIT-BIH record from PhysioNet.

        Parameters
        ----------
        record_name:
            Record identifier, e.g. ``"100"``.
        data_dir:
            Local directory where the files should be saved.

        Raises
        ------
        RuntimeError
            If the download fails for any reason.
        """
        os.makedirs(data_dir, exist_ok=True)
        # Check if already downloaded (header + data files)
        hea_path = os.path.join(data_dir, f"{record_name}.hea")
        dat_path = os.path.join(data_dir, f"{record_name}.dat")
        if os.path.exists(hea_path) and os.path.exists(dat_path):
            logger.debug("Record %s already present in %s – skipping download.", record_name, data_dir)
            return

        try:
            logger.info("Downloading record %s …", record_name)
            wfdb.dl_database(
                ECGDataLoader.DB_NAME,
                dl_dir=data_dir,
                records=[record_name],
                annotators=["atr"],
            )
            logger.info("Record %s downloaded successfully.", record_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to download record '{record_name}': {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Loading a single record                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def load_record(
        record_name: str,
        data_dir: str,
    ) -> Tuple[np.ndarray, dict, Optional[wfdb.Annotation]]:
        """Load a record's signal, metadata, and beat annotations.

        Parameters
        ----------
        record_name:
            Record identifier without extension, e.g. ``"100"``.
        data_dir:
            Directory containing the wfdb files.

        Returns
        -------
        signal : np.ndarray, shape (n_samples, n_leads)
            Raw ADC signal converted to physical units (mV).
        fields : dict
            Metadata from :func:`wfdb.rdrecord` (fs, units, sig_name, …).
        annotation : wfdb.Annotation or None
            Beat annotations; ``None`` if the annotation file is absent.

        Raises
        ------
        FileNotFoundError
            If the header file cannot be found in *data_dir*.
        """
        record_path = os.path.join(data_dir, record_name)
        hea_file = record_path + ".hea"
        if not os.path.exists(hea_file):
            raise FileNotFoundError(
                f"Header file not found: {hea_file}. "
                "Did you download the record first?"
            )

        try:
            record = wfdb.rdrecord(record_path)
            signal: np.ndarray = record.p_signal          # physical units
            fields: dict = {
                "fs": record.fs,
                "units": record.units,
                "sig_name": record.sig_name,
                "n_sig": record.n_sig,
                "sig_len": record.sig_len,
                "record_name": record.record_name,
                "base_date": record.base_date,
                "base_time": record.base_time,
            }
        except Exception as exc:
            logger.error("Error reading record %s: %s", record_name, exc)
            raise

        # Attempt to load annotation
        annotation: Optional[wfdb.Annotation] = None
        try:
            annotation = wfdb.rdann(record_path, "atr")
        except FileNotFoundError:
            logger.warning("Annotation file missing for record %s.", record_name)
        except Exception as exc:
            logger.warning("Could not load annotation for record %s: %s", record_name, exc)

        return signal, fields, annotation

    # ------------------------------------------------------------------ #
    #  Iterating over multiple records                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def load_all_records(
        record_list: List[str],
        data_dir: str,
    ) -> Generator[Tuple[str, np.ndarray, dict, Optional[wfdb.Annotation]], None, None]:
        """Yield (record_name, signal, fields, annotation) for every record.

        Parameters
        ----------
        record_list:
            List of record identifiers to load.
        data_dir:
            Directory containing the wfdb files.

        Yields
        ------
        record_name : str
        signal : np.ndarray
        fields : dict
        annotation : wfdb.Annotation or None
        """
        for record_name in record_list:
            try:
                signal, fields, annotation = ECGDataLoader.load_record(record_name, data_dir)
                yield record_name, signal, fields, annotation
            except Exception as exc:
                logger.error("Skipping record %s due to error: %s", record_name, exc)
                continue

    # ------------------------------------------------------------------ #
    #  Patient ID extraction                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_patient_ids(record_list: List[str]) -> List[str]:
        """Return the patient ID for each record.

        The MIT-BIH convention uses the first three digits of the record name
        as a patient identifier (e.g. record ``"100"`` → patient ``"100"``).

        Parameters
        ----------
        record_list:
            List of record name strings.

        Returns
        -------
        List[str]
            Patient ID corresponding to each entry in *record_list*.
        """
        return [ECGDataLoader._extract_patient_id(r) for r in record_list]

    @staticmethod
    def _extract_patient_id(record_name: str) -> str:
        """Extract patient ID from a record name (first 3 digit characters)."""
        digits = "".join(ch for ch in record_name if ch.isdigit())
        return digits[:3] if len(digits) >= 3 else digits

    # ------------------------------------------------------------------ #
    #  Convenience: download then load                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def download_and_load(
        record_name: str,
        data_dir: str,
    ) -> Tuple[np.ndarray, dict, Optional[wfdb.Annotation]]:
        """Download (if needed) and load a record in one call.

        Parameters
        ----------
        record_name:
            Record identifier.
        data_dir:
            Local storage directory.

        Returns
        -------
        Same as :meth:`load_record`.
        """
        ECGDataLoader.download_record(record_name, data_dir)
        return ECGDataLoader.load_record(record_name, data_dir)
