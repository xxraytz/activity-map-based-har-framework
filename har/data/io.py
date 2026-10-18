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


def load_npz_multilabel_sequence(
    filepath: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an activity-map sequence and independent per-class frame labels."""
    with np.load(filepath, allow_pickle=True) as npz:
        seq = npz["data"].astype(np.float32)
        labels_multihot = npz["labels_multihot"].astype(np.float32)
        action_ids = npz["action_ids"].astype(np.int64)

    if seq.ndim == 2:
        seq = seq[None, ...]
    if seq.ndim != 3:
        raise ValueError(f"{filepath}: expected data ndim 2 or 3, got shape {seq.shape}")
    if labels_multihot.ndim != 2:
        raise ValueError(
            f"{filepath}: expected labels_multihot shape (classes, frames), "
            f"got {labels_multihot.shape}"
        )
    if labels_multihot.shape != (len(action_ids), seq.shape[-1]):
        raise ValueError(
            f"{filepath}: labels_multihot shape {labels_multihot.shape} does not match "
            f"{len(action_ids)} action ids and data shape {seq.shape}"
        )
    return seq, labels_multihot, action_ids
