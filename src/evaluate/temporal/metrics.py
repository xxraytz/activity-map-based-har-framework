from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .types import ClassMapping, FrameTimeline, SegmentMetricResult, TemporalMetricResult, WindowPredictions


def score_timeline(
    y_true: np.ndarray,
    timeline: FrameTimeline,
    mapping: ClassMapping,
    pooling: str,
    smoothing: str,
) -> tuple[TemporalMetricResult, np.ndarray, dict]:
    valid = timeline.valid_mask
    y_true_valid = y_true[valid]
    y_pred_valid = timeline.pred[valid]

    labels = list(range(len(mapping.class_names)))
    if y_true_valid.size == 0:
        result = TemporalMetricResult(pooling, smoothing, float("nan"), float("nan"), 0)
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        report = {}
        return result, cm, report

    result = TemporalMetricResult(
        pooling=pooling,
        smoothing=smoothing,
        accuracy=float(accuracy_score(y_true_valid, y_pred_valid)),
        f1_macro=float(f1_score(y_true_valid, y_pred_valid, labels=labels, average="macro", zero_division=0)),
        support=int(y_true_valid.size),
    )
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=labels)
    report = classification_report(
        y_true_valid,
        y_pred_valid,
        labels=labels,
        target_names=mapping.class_names,
        output_dict=True,
        zero_division=0,
    )
    return result, cm, report


def aggregate_results(results: list[TemporalMetricResult]) -> pd.DataFrame:
    rows = [
        {
            "pooling": item.pooling,
            "smoothing": item.smoothing,
            "accuracy": item.accuracy,
            "f1_macro": item.f1_macro,
            "support": item.support,
        }
        for item in results
    ]
    return pd.DataFrame(rows)


def _iter_label_segments(y_true: np.ndarray):
    if y_true.size == 0:
        return
    start = 0
    current = int(y_true[0])
    for idx in range(1, y_true.size):
        label = int(y_true[idx])
        if label == current:
            continue
        yield start, idx, current
        start = idx
        current = label
    yield start, y_true.size, current


def _majority_vote(labels: np.ndarray, num_classes: int) -> tuple[int, float]:
    counts = np.bincount(labels, minlength=num_classes)
    pred = int(counts.argmax())
    confidence = float(counts[pred] / max(1, labels.size))
    return pred, confidence


def _mean_probability(probs: np.ndarray) -> tuple[int, float]:
    mean_probs = probs.mean(axis=0)
    pred = int(mean_probs.argmax())
    return pred, float(mean_probs[pred])


def segment_decisions(
    pred: WindowPredictions,
    timeline: FrameTimeline,
    mapping: ClassMapping,
    pooling: str,
    smoothing: str,
    decision: str,
    min_confidence: float,
    min_coverage: float,
    min_segment_length: int,
) -> list[dict]:
    num_classes = len(mapping.class_names)
    rows: list[dict] = []

    for segment_id, (start, end, true_label) in enumerate(_iter_label_segments(pred.y_true)):
        length = end - start
        if length < min_segment_length:
            continue

        valid = timeline.valid_mask[start:end]
        coverage = float(valid.mean()) if length else 0.0
        segment_probs = timeline.probs[start:end][valid]
        segment_pred = timeline.pred[start:end][valid]

        is_unknown = bool(coverage < min_coverage or segment_probs.size == 0)
        confidence = 0.0
        predicted_label = -1

        if not is_unknown:
            if decision == "mean_prob":
                predicted_label, confidence = _mean_probability(segment_probs)
            elif decision == "majority_vote":
                predicted_label, confidence = _majority_vote(segment_pred, num_classes)
            else:
                raise ValueError(f"Unknown segment decision strategy: {decision}")
            if confidence < min_confidence:
                is_unknown = True
                predicted_label = -1

        rows.append({
            "filepath": str(pred.record.filepath),
            "actor_id": pred.record.actor_id,
            "seq_id": pred.record.seq_id,
            "segment_id": segment_id,
            "start": start,
            "end": end,
            "length": length,
            "coverage": coverage,
            "true_idx": true_label,
            "true_label": mapping.class_names[true_label],
            "pred_idx": predicted_label,
            "pred_label": "__unknown__" if is_unknown else mapping.class_names[predicted_label],
            "confidence": confidence,
            "correct": (not is_unknown) and predicted_label == true_label,
            "pooling": pooling,
            "smoothing": smoothing,
            "decision": decision,
            "min_confidence": min_confidence,
            "min_coverage": min_coverage,
        })

    return rows


def score_segment_rows(rows: list[dict], mapping: ClassMapping, pooling: str, smoothing: str, decision: str,
                       min_confidence: float, min_coverage: float) -> tuple[SegmentMetricResult, np.ndarray, dict]:
    labels = list(range(len(mapping.class_names)))
    if not rows:
        result = SegmentMetricResult(pooling, smoothing, decision, min_confidence, min_coverage, float("nan"), float("nan"), 0, 0)
        matrix = np.zeros((len(labels), len(labels) + 1), dtype=np.int64)
        return result, matrix, {}

    y_true = np.asarray([row["true_idx"] for row in rows], dtype=np.int64)
    y_pred = np.asarray([row["pred_idx"] for row in rows], dtype=np.int64)
    known_pred = np.where(y_pred < 0, len(labels), y_pred)

    matrix = confusion_matrix(y_true, known_pred, labels=labels + [len(labels)])[:len(labels)]
    f1_pred = np.where(y_pred < 0, -1, y_pred)
    result = SegmentMetricResult(
        pooling=pooling,
        smoothing=smoothing,
        decision=decision,
        min_confidence=min_confidence,
        min_coverage=min_coverage,
        accuracy=float(accuracy_score(y_true, f1_pred)),
        f1_macro=float(f1_score(y_true, f1_pred, labels=labels, average="macro", zero_division=0)),
        support=int(y_true.size),
        unknown=int((y_pred < 0).sum()),
    )
    report = classification_report(
        y_true,
        f1_pred,
        labels=labels,
        target_names=mapping.class_names,
        output_dict=True,
        zero_division=0,
    )
    return result, matrix, report


def aggregate_segment_results(results: list[SegmentMetricResult]) -> pd.DataFrame:
    rows = [
        {
            "pooling": item.pooling,
            "smoothing": item.smoothing,
            "decision": item.decision,
            "min_confidence": item.min_confidence,
            "min_coverage": item.min_coverage,
            "accuracy": item.accuracy,
            "f1_macro": item.f1_macro,
            "support": item.support,
            "unknown": item.unknown,
        }
        for item in results
    ]
    return pd.DataFrame(rows)


def write_confusion_matrix(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    df.to_csv(path)


def write_segment_confusion_matrix(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    df = pd.DataFrame(matrix, index=class_names, columns=class_names + ["__unknown__"])
    df.to_csv(path)


def write_classification_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True))


def per_sequence_row(
    pred: WindowPredictions,
    pooling: str,
    smoothing: str,
    result: TemporalMetricResult,
) -> dict:
    return {
        "filepath": str(pred.record.filepath),
        "actor_id": pred.record.actor_id,
        "seq_id": pred.record.seq_id,
        "pooling": pooling,
        "smoothing": smoothing,
        "accuracy": result.accuracy,
        "f1_macro": result.f1_macro,
        "support": result.support,
    }
