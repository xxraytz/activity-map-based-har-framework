from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npz_sequence(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an activity-map sequence as channel-first (C, H, T)."""
    with np.load(filepath, allow_pickle=True) as npz:
        seq = npz["data"].astype(np.float32)
        labels = npz["labels"]

    if seq.ndim == 2:
        seq = seq[None, ...]
    if seq.ndim != 3:
        raise ValueError(f"{filepath}: expected data ndim 2 or 3, got shape {seq.shape}")
    if seq.shape[-1] != len(labels):
        raise ValueError(f"{filepath}: labels length {len(labels)} does not match data shape {seq.shape}")
    return seq, labels

