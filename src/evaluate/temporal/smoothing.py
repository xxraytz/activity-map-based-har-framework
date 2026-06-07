from __future__ import annotations

from collections import Counter

import numpy as np

from .types import FrameTimeline


def available_smoothing_methods() -> tuple[str, ...]:
    return ("none", "moving_average", "ema", "median", "hysteresis")


def _renormalize(probs: np.ndarray) -> np.ndarray:
    denom = probs.sum(axis=1, keepdims=True)
    return probs / np.maximum(denom, 1e-12)


def smooth_moving_average(timeline: FrameTimeline, kernel_size: int) -> FrameTimeline:
    if kernel_size <= 1:
        return FrameTimeline("moving_average", timeline.probs.copy(), timeline.valid_mask.copy())
    if kernel_size % 2 == 0:
        raise ValueError("moving_average kernel_size must be odd")

    pad = kernel_size // 2
    padded = np.pad(timeline.probs, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(timeline.probs)
    for idx in range(timeline.probs.shape[0]):
        out[idx] = padded[idx:idx + kernel_size].mean(axis=0)
    return FrameTimeline("moving_average", _renormalize(out).astype(np.float32), timeline.valid_mask.copy())


def smooth_ema(timeline: FrameTimeline, alpha: float) -> FrameTimeline:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("ema alpha must be in (0, 1]")

    out = np.zeros_like(timeline.probs)
    if len(out) == 0:
        return FrameTimeline("ema", out, timeline.valid_mask.copy())
    out[0] = timeline.probs[0]
    for idx in range(1, len(out)):
        out[idx] = alpha * timeline.probs[idx] + (1.0 - alpha) * out[idx - 1]
    return FrameTimeline("ema", _renormalize(out).astype(np.float32), timeline.valid_mask.copy())


def smooth_median_classes(timeline: FrameTimeline, kernel_size: int) -> FrameTimeline:
    if kernel_size <= 1:
        return FrameTimeline("median", timeline.probs.copy(), timeline.valid_mask.copy())
    if kernel_size % 2 == 0:
        raise ValueError("median kernel_size must be odd")

    pred = timeline.pred
    pad = kernel_size // 2
    padded = np.pad(pred, (pad, pad), mode="edge")
    out_pred = np.zeros_like(pred)
    for idx in range(len(pred)):
        counts = Counter(padded[idx:idx + kernel_size])
        out_pred[idx] = counts.most_common(1)[0][0]

    probs = np.zeros_like(timeline.probs)
    probs[np.arange(len(out_pred)), out_pred] = 1.0
    return FrameTimeline("median", probs.astype(np.float32), timeline.valid_mask.copy())


def smooth_hysteresis(timeline: FrameTimeline, threshold: float, margin: float, decay: float) -> FrameTimeline:
    if not 0.0 <= decay <= 1.0:
        raise ValueError("hysteresis decay must be in [0, 1]")

    probs = timeline.probs
    if len(probs) == 0:
        return FrameTimeline("hysteresis", probs.copy(), timeline.valid_mask.copy())

    current = int(probs[0].argmax())
    evidence = np.zeros(probs.shape[1], dtype=np.float64)
    out_pred = np.zeros(probs.shape[0], dtype=np.int64)

    for idx, row in enumerate(probs):
        candidate = int(row.argmax())
        if candidate == current:
            evidence *= decay
        else:
            evidence *= decay
            evidence[candidate] += max(0.0, float(row[candidate] - row[current] + margin))
            if evidence[candidate] >= threshold:
                current = candidate
                evidence[:] = 0.0
        out_pred[idx] = current

    out = np.zeros_like(probs)
    out[np.arange(len(out_pred)), out_pred] = 1.0
    return FrameTimeline("hysteresis", out.astype(np.float32), timeline.valid_mask.copy())


def apply_smoothing(
    timeline: FrameTimeline,
    name: str,
    *,
    moving_average_kernel: int = 5,
    ema_alpha: float = 0.3,
    median_kernel: int = 5,
    hysteresis_threshold: float = 0.75,
    hysteresis_margin: float = 0.0,
    hysteresis_decay: float = 0.9,
) -> FrameTimeline:
    if name == "none":
        return FrameTimeline("none", timeline.probs.copy(), timeline.valid_mask.copy())
    if name == "moving_average":
        return smooth_moving_average(timeline, moving_average_kernel)
    if name == "ema":
        return smooth_ema(timeline, ema_alpha)
    if name == "median":
        return smooth_median_classes(timeline, median_kernel)
    if name == "hysteresis":
        return smooth_hysteresis(timeline, hysteresis_threshold, hysteresis_margin, hysteresis_decay)
    raise ValueError(f"Unknown smoothing method: {name}")

