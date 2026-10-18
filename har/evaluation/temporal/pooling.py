from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .types import FrameTimeline, WindowPredictions


PoolingFn = Callable[[WindowPredictions], FrameTimeline]


def _empty_scores(pred: WindowPredictions) -> tuple[np.ndarray, np.ndarray]:
    num_classes = pred.probs.shape[1]
    scores = np.zeros((pred.num_frames, num_classes), dtype=np.float64)
    weights = np.zeros(pred.num_frames, dtype=np.float64)
    return scores, weights


def _normalize_scores(name: str, scores: np.ndarray, weights: np.ndarray) -> FrameTimeline:
    valid = weights > 0
    probs = np.zeros_like(scores, dtype=np.float32)
    probs[valid] = (scores[valid] / weights[valid, None]).astype(np.float32)
    return FrameTimeline(name=name, probs=probs, valid_mask=valid)


def _triangular_weights(window_size: int) -> np.ndarray:
    center = (window_size - 1) / 2.0
    scale = (window_size + 1) / 2.0
    weights = 1.0 - np.abs(np.arange(window_size) - center) / scale
    return np.maximum(weights, 1e-6)


def pool_center(pred: WindowPredictions) -> FrameTimeline:
    scores, weights = _empty_scores(pred)
    center_offset = pred.window_size // 2
    centers = pred.starts + center_offset
    valid_centers = centers < pred.num_frames
    scores[centers[valid_centers]] += pred.probs[valid_centers]
    weights[centers[valid_centers]] += 1.0
    return _normalize_scores("center", scores, weights)


def pool_uniform(pred: WindowPredictions) -> FrameTimeline:
    scores, weights = _empty_scores(pred)
    for start, probs in zip(pred.starts, pred.probs):
        end = min(start + pred.window_size, pred.num_frames)
        scores[start:end] += probs
        weights[start:end] += 1.0
    return _normalize_scores("uniform", scores, weights)


def pool_center_weighted(pred: WindowPredictions) -> FrameTimeline:
    scores, weights = _empty_scores(pred)
    kernel = _triangular_weights(pred.window_size)
    for start, probs in zip(pred.starts, pred.probs):
        end = min(start + pred.window_size, pred.num_frames)
        local = kernel[:end - start]
        scores[start:end] += local[:, None] * probs[None, :]
        weights[start:end] += local
    return _normalize_scores("center_weighted", scores, weights)


def pool_max(pred: WindowPredictions) -> FrameTimeline:
    num_classes = pred.probs.shape[1]
    scores = np.full((pred.num_frames, num_classes), -np.inf, dtype=np.float64)
    valid = np.zeros(pred.num_frames, dtype=bool)
    for start, probs in zip(pred.starts, pred.probs):
        end = min(start + pred.window_size, pred.num_frames)
        scores[start:end] = np.maximum(scores[start:end], probs)
        valid[start:end] = True
    scores[~valid] = 0.0
    row_sum = scores[valid].sum(axis=1, keepdims=True)
    scores[valid] = scores[valid] / np.maximum(row_sum, 1e-12)
    return FrameTimeline("max", scores.astype(np.float32), valid)


def pool_topk_mean(pred: WindowPredictions, k: int = 3) -> FrameTimeline:
    num_classes = pred.probs.shape[1]
    buckets: list[list[np.ndarray]] = [[] for _ in range(pred.num_frames)]
    for start, probs in zip(pred.starts, pred.probs):
        end = min(start + pred.window_size, pred.num_frames)
        for frame_idx in range(start, end):
            buckets[frame_idx].append(probs)

    scores = np.zeros((pred.num_frames, num_classes), dtype=np.float32)
    valid = np.zeros(pred.num_frames, dtype=bool)
    for frame_idx, values in enumerate(buckets):
        if not values:
            continue
        stacked = np.stack(values)
        kk = min(k, stacked.shape[0])
        part = np.partition(stacked, kth=stacked.shape[0] - kk, axis=0)[-kk:]
        scores[frame_idx] = part.mean(axis=0)
        valid[frame_idx] = True
    return FrameTimeline(f"top{k}_mean", scores, valid)


def pool_vote(pred: WindowPredictions) -> FrameTimeline:
    num_classes = pred.probs.shape[1]
    scores = np.zeros((pred.num_frames, num_classes), dtype=np.float64)
    weights = np.zeros(pred.num_frames, dtype=np.float64)
    winners = pred.probs.argmax(axis=1)
    for start, winner in zip(pred.starts, winners):
        end = min(start + pred.window_size, pred.num_frames)
        scores[start:end, winner] += 1.0
        weights[start:end] += 1.0
    return _normalize_scores("vote", scores, weights)


def available_pooling_methods() -> tuple[str, ...]:
    return ("center", "uniform", "center_weighted", "max", "topk_mean", "vote")


def apply_pooling(pred: WindowPredictions, name: str, *, topk: int = 3) -> FrameTimeline:
    if name == "center":
        return pool_center(pred)
    if name == "uniform":
        return pool_uniform(pred)
    if name == "center_weighted":
        return pool_center_weighted(pred)
    if name == "max":
        return pool_max(pred)
    if name == "topk_mean":
        return pool_topk_mean(pred, k=topk)
    if name == "vote":
        return pool_vote(pred)
    raise ValueError(f"Unknown pooling method: {name}")
