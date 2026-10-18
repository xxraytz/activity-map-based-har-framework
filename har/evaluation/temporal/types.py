from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClassMapping:
    class_names: list[str]
    cls2idx: dict[str, int]
    raw_to_canonical: dict[str, str]


@dataclass(frozen=True)
class SequenceRecord:
    filepath: Path
    actor_id: int
    seq_id: int


@dataclass(frozen=True)
class WindowPredictions:
    record: SequenceRecord
    y_true: np.ndarray
    starts: np.ndarray
    logits: np.ndarray
    probs: np.ndarray
    window_size: int

    @property
    def num_frames(self) -> int:
        return int(self.y_true.shape[0])


@dataclass(frozen=True)
class FrameTimeline:
    name: str
    probs: np.ndarray
    valid_mask: np.ndarray

    @property
    def pred(self) -> np.ndarray:
        return self.probs.argmax(axis=1)


@dataclass(frozen=True)
class TemporalMetricResult:
    pooling: str
    smoothing: str
    accuracy: float
    f1_macro: float
    support: int


@dataclass(frozen=True)
class SegmentMetricResult:
    pooling: str
    smoothing: str
    decision: str
    min_confidence: float
    min_coverage: float
    accuracy: float
    f1_macro: float
    support: int
    unknown: int
