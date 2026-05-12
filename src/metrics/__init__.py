# src/metrics/__init__.py
from .metrics import BaseMetric, F1Macro, build_metric
from .evaluate import evaluate_all, should_eval_metrics
