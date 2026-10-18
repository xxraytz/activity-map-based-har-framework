from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from har.data.io import load_npz_sequence
from har.data.labels import LabelMapping, build_label_mapping, label_indices
from har.data.normalization import apply_sequence_normalization
from har.data.splits import split_manifest as split_shared_manifest
from .types import ClassMapping, SequenceRecord


def build_class_mapping(cfg: dict) -> ClassMapping:
    mapping = build_label_mapping(cfg["data"].get("class_names", {}))
    return ClassMapping(
        class_names=mapping.class_names,
        cls2idx=mapping.cls2idx,
        raw_to_canonical=mapping.raw_to_canonical,
    )


def canonical_frame_labels(labels, mapping: ClassMapping) -> np.ndarray:
    shared_mapping = LabelMapping(
        class_names=mapping.class_names,
        cls2idx=mapping.cls2idx,
        raw_to_canonical=mapping.raw_to_canonical,
    )
    return label_indices(labels, shared_mapping)


def split_manifest(cfg: dict, split: str) -> pd.DataFrame:
    mapping = build_label_mapping(cfg["data"].get("class_names", {}))
    return split_shared_manifest(cfg, split, mapping)


def records_from_manifest(df: pd.DataFrame) -> list[SequenceRecord]:
    return [
        SequenceRecord(filepath=Path(row.filepath), actor_id=int(row.actor_id), seq_id=int(row.seq_id))
        for row in df.itertuples(index=False)
    ]


def normalize_sequence(seq: np.ndarray, cfg: dict) -> np.ndarray:
    return apply_sequence_normalization(seq, cfg["data"].get("normalize"))


def load_sequence(record: SequenceRecord, cfg: dict, mapping: ClassMapping) -> tuple[np.ndarray, np.ndarray]:
    seq, labels = load_npz_sequence(record.filepath)
    seq = normalize_sequence(seq, cfg)
    y_true = canonical_frame_labels(labels, mapping)
    return seq, y_true
