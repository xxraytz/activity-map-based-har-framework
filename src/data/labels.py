from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class LabelMapping:
    class_names: list[str]
    cls2idx: dict[str, int]
    raw_to_canonical: dict[str, str]


def build_label_mapping(class_names_cfg: dict[object, object] | None) -> LabelMapping:
    raw_to_canonical = {str(raw): str(canonical) for raw, canonical in (class_names_cfg or {}).items()}
    class_names = list(dict.fromkeys(raw_to_canonical.values()))
    cls2idx = {name: idx for idx, name in enumerate(class_names)}
    return LabelMapping(class_names=class_names, cls2idx=cls2idx, raw_to_canonical=raw_to_canonical)


def canonical_names(labels: Iterable[object], mapping: LabelMapping) -> list[str]:
    return [mapping.raw_to_canonical.get(str(label), str(label)) for label in labels]


def label_indices(labels: Iterable[object], mapping: LabelMapping) -> np.ndarray:
    names = canonical_names(labels, mapping)
    try:
        return np.asarray([mapping.cls2idx[name] for name in names], dtype=np.int64)
    except KeyError as exc:
        raise KeyError(f"Unknown canonical class {exc!s}; class_names={mapping.class_names}") from exc


def majority_canonical(labels: Iterable[object], mapping: LabelMapping) -> str:
    names = canonical_names(labels, mapping)
    if not names:
        raise ValueError("Cannot compute majority label for an empty label sequence")
    return Counter(names).most_common(1)[0][0]


def window_distribution(labels: Iterable[object], mapping: LabelMapping) -> torch.Tensor:
    idxs = torch.as_tensor(label_indices(labels, mapping), dtype=torch.long)
    counts = torch.bincount(idxs, minlength=len(mapping.class_names)).float()
    total = counts.sum()
    if total <= 0:
        raise ValueError("Cannot build a target distribution for an empty window")
    return counts / total

