"""
Data loaders for MIMII and SKAB.

Owner: Taiwo

Both loaders should return raw 1D sequences plus enough metadata (timestamp
or index, and a normal/anomalous label if available) for `splits.py` to
build sequential temporal splits downstream. Keep this file free of any
split logic -- that lives in splits.py.

TODO(Taiwo):
- Decide on-disk layout for raw MIMII (audio, .wav) and SKAB (csv) once
  downloaded, and document expected paths here.
- Decide the sequence representation for MIMII: raw waveform vs. per-frame
  feature vectors (e.g. log-mel). This choice affects whether the "1D"
  framing holds -- flag to Samuel before committing.
"""

from dataclasses import dataclass
from typing import Iterator
import numpy as np


@dataclass
class SensorSequence:
    """One sequence/window of sensor data with its metadata."""
    values: np.ndarray       # shape (length,) or (length, channels)
    timestamp: float         # or sample index if no real timestamp
    machine_id: str          # e.g. MIMII machine type/ID, or SKAB run id
    is_anomalous: bool       # ground-truth label, where available


def load_mimii(root: str, machine_type: str | None = None) -> Iterator[SensorSequence]:
    """
    Load MIMII recordings.

    Args:
        root: path to the extracted MIMII dataset.
        machine_type: optional filter (e.g. "fan", "pump", "slider", "valve").

    Yields:
        SensorSequence objects in on-disk / recording order (do NOT shuffle
        here -- ordering is needed for temporal splitting).
    """
    raise NotImplementedError("Taiwo: implement MIMII loading")


def load_skab(root: str) -> Iterator[SensorSequence]:
    """
    Load SKAB multivariate time-series runs.

    Args:
        root: path to the extracted SKAB dataset.

    Yields:
        SensorSequence objects in original time order.
    """
    raise NotImplementedError("Taiwo: implement SKAB loading")


if __name__ == "__main__":
    # Quick manual smoke test while developing -- replace with real paths.
    pass
