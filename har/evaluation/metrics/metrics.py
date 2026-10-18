# src/metrics/metrics.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class BaseMetric(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    def __repr__(self) -> str:
        return self.name()

    @abstractmethod
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        y_true: (N,) int
        y_pred: либо (N,) int (классы), либо (N, C) float (логиты/скор)
        """
        ...


def _to_hard_pred(y_pred: np.ndarray) -> np.ndarray:
    """Если переданы логиты/скор (N,C), берём argmax; если (N,), возвращаем как есть."""
    if y_pred.ndim == 2:
        return y_pred.argmax(axis=1)
    return y_pred


class Accuracy(BaseMetric):
    def name(self) -> str:
        return "accuracy"

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0:
            return float("nan")
        hard = _to_hard_pred(y_pred)
        return float((hard == y_true).mean())


class TopKAccuracy(BaseMetric):
    def __init__(self, k: int = 5) -> None:
        self.k = int(k)

    def name(self) -> str:
        return f"top_k_accuracy@{self.k}"

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0:
            return float("nan")
        if y_pred.ndim == 1:
            # если пришли уже жёсткие предсказания, top-k вырождается в top-1
            return float((y_pred == y_true).mean())
        # y_pred: (N, C) — логиты/скор
        # возьмём top-k индексы по оси классов
        # argsort по убыванию: берём последние k после сортировки по возрастанию -> используем argpartition
        N, C = y_pred.shape
        k = min(self.k, C)
        topk_idx = np.argpartition(y_pred, kth=C - k, axis=1)[:, -k:]
        # проверяем, входит ли true label в top-k
        hits = (topk_idx == y_true[:, None]).any(axis=1)
        return float(hits.mean())


class F1Macro(BaseMetric):
    def __init__(self) -> None:
        from sklearn.metrics import f1_score  # noqa: F401

    def name(self) -> str:
        return "f1_macro"

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import f1_score
        if y_true.size == 0:
            return float("nan")
        hard = _to_hard_pred(y_pred)
        return float(f1_score(y_true, hard, average="macro"))


class MultilabelF1Macro(BaseMetric):
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def name(self) -> str:
        return "multilabel_f1_macro"

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import f1_score

        probabilities = 1.0 / (1.0 + np.exp(-y_pred))
        hard = probabilities >= self.threshold
        return float(f1_score(y_true, hard, average="macro", zero_division=0))


class MultilabelMeanAveragePrecision(BaseMetric):
    def name(self) -> str:
        return "multilabel_map"

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import average_precision_score

        probabilities = 1.0 / (1.0 + np.exp(-y_pred))
        valid = y_true.sum(axis=0) > 0
        if not np.any(valid):
            return float("nan")
        return float(average_precision_score(y_true[:, valid], probabilities[:, valid], average="macro"))


# Регистр типов (без параметров)
_METRIC_REGISTRY: Dict[str, type[BaseMetric]] = {
    "accuracy": Accuracy,
    "f1_macro": F1Macro,
    "top_k_accuracy": TopKAccuracy,
    "multilabel_f1_macro": MultilabelF1Macro,
    "multilabel_map": MultilabelMeanAveragePrecision,
}


def _parse_metric_name(name: str) -> Tuple[str, Optional[int]]:
    """
    Разбирает имена вида 'top_k_accuracy@5' -> ('top_k_accuracy', 5).
    Без '@' -> (name, None).
    """
    if "@" in name:
        base, k = name.split("@", 1)
        try:
            return base, int(k)
        except ValueError:
            return base, None
    return name, None


def build_metric(name: str, cfg: Optional[dict] = None) -> BaseMetric:
    base, k = _parse_metric_name(name)
    if base not in _METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}")

    cls = _METRIC_REGISTRY[base]
    if base == "top_k_accuracy":
        if k is None and cfg is not None:
            k = int(cfg.get("metrics", {}).get("params", {}).get("top_k", 5))
        return cls(k=k or 5)  # type: ignore[arg-type]
    if base == "multilabel_f1_macro":
        threshold = 0.5
        if cfg is not None:
            threshold = float(cfg.get("metrics", {}).get("multilabel_threshold", 0.5))
        return cls(threshold=threshold)  # type: ignore[arg-type]
    return cls()
