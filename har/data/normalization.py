from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .io import load_npz_sequence


def apply_sequence_normalization(seq: np.ndarray, normalize: Mapping[str, object] | None) -> np.ndarray:
    normalize = normalize or {}
    mode = normalize.get("mode")
    if not mode:
        return seq

    eps = float(normalize.get("eps", 1e-6))
    if mode == "per_channel_sequence":
        mean = seq.mean(axis=(1, 2), keepdims=True)
        std = seq.std(axis=(1, 2), keepdims=True)
        return (seq - mean) / np.maximum(std, eps)

    if mode == "trainset_channel_standard":
        stats = normalize.get("stats")
        if not stats:
            raise ValueError("trainset_channel_standard normalization requires fitted stats")
        mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
        if mean.shape[0] != seq.shape[0] or std.shape[0] != seq.shape[0]:
            raise ValueError(
                "Normalization stats channel count does not match sequence: "
                f"stats={mean.shape[0]}, seq={seq.shape[0]}"
            )
        return (seq - mean) / np.maximum(std, eps)

    if mode == "per_channel_window":
        raise ValueError("per_channel_window normalization is not supported")

    raise ValueError(f"Unknown normalization mode: {mode}")


def fit_trainset_channel_stats(recs: pd.DataFrame, eps: float = 1e-6) -> dict[str, object]:
    sums = None
    sums_sq = None
    counts = None

    for filepath in recs["filepath"]:
        seq, _ = load_npz_sequence(filepath)
        channel_sum = seq.sum(axis=(1, 2), dtype=np.float64)
        channel_sum_sq = np.square(seq, dtype=np.float64).sum(axis=(1, 2))
        channel_count = np.full(seq.shape[0], seq.shape[1] * seq.shape[2], dtype=np.float64)

        if sums is None:
            sums = channel_sum
            sums_sq = channel_sum_sq
            counts = channel_count
        else:
            if sums.shape[0] != seq.shape[0]:
                raise ValueError(
                    "All files in one manifest split must have the same channel count "
                    f"for trainset_channel_standard normalization; got {sums.shape[0]} and {seq.shape[0]}"
                )
            sums += channel_sum
            sums_sq += channel_sum_sq
            counts += channel_count

    if sums is None:
        raise ValueError("Cannot fit trainset_channel_standard stats on an empty train split")

    mean = sums / counts
    var = np.maximum(sums_sq / counts - np.square(mean), 0.0)
    std = np.sqrt(np.maximum(var, float(eps)))
    return {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "count": counts.astype(float).tolist(),
        "scope": "lopo_train",
    }


def resolve_normalization_config(
    normalize: Mapping[str, object] | None,
    train_recs: pd.DataFrame,
) -> dict[str, object]:
    resolved = dict(normalize or {})
    if resolved.get("mode") == "trainset_channel_standard" and not resolved.get("stats"):
        resolved["stats"] = fit_trainset_channel_stats(
            train_recs,
            eps=float(resolved.get("eps", 1e-6)),
        )
    return resolved


class SequenceNormalizer:
    def __init__(self, config: Mapping[str, object] | None = None):
        self.config = dict(config or {})

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        return apply_sequence_normalization(seq, self.config)
