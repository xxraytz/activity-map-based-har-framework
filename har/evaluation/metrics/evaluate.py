# src/metrics/evaluate.py
from __future__ import annotations
from typing import Dict, List, Tuple
import os
import csv
import numpy as np
import torch

from .metrics import BaseMetric, build_metric
from tqdm.auto import tqdm

def _collect_logits_targets(model: torch.nn.Module,
                            loader: torch.utils.data.DataLoader,
                            device: torch.device,
                            task: str = "multiclass") -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает:
      logits: (N, C) float
      y_true: (N,) int  (argmax от soft-таргетов)
    """
    model.eval()
    logits_list, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)                                  # (B, C), логиты
            logits_list.append(out.detach().cpu().numpy())
            if task == "multilabel":
                targets.append(y.cpu().numpy())
            else:
                targets.append(y.argmax(dim=-1).cpu().numpy())  # hard target
    if not logits_list:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int64)
    return np.concatenate(logits_list), np.concatenate(targets)


def _ensure_metrics_csv(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "split", "metric", "value"])


def _log_metrics_csv(path: str, epoch: int, split: str, results: Dict[str, float]) -> None:
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for k, v in results.items():
            w.writerow([epoch, split, k, f"{v:.6f}" if isinstance(v, float) else v])


def build_metrics_from_config(cfg: dict) -> List[BaseMetric]:
    names = cfg.get("metrics", {}).get("names", [])
    return [build_metric(n, cfg=cfg) for n in names]


def should_eval_metrics(cfg: dict, epoch: int) -> bool:
    total_epochs = int(cfg.get("epochs", 1))
    every = int(cfg.get("metrics", {}).get("eval_every", 0))
    return (every and (epoch % every == 0)) or (epoch == total_epochs)


def evaluate_all(model: torch.nn.Module,
                 loaders: Dict[str, torch.utils.data.DataLoader],
                 device: torch.device,
                 cfg: dict,
                 run_dir: str,
                 epoch: int) -> Dict[str, Dict[str, float]]:
    metrics = build_metrics_from_config(cfg)
    if not metrics:
        return {}

    metrics_csv = os.path.join(run_dir, "metrics.csv")
    _ensure_metrics_csv(metrics_csv)

    results_all: Dict[str, Dict[str, float]] = {}
    # информационный заголовок
    metric_names = ", ".join(m.name() for m in metrics)
    print(f"[Epoch {epoch}] Evaluating metrics: {metric_names}")

    # внешний прогресс по сплитам
    for split_name, loader in list(loaders.items()):
        logits, y_true = _collect_logits_targets(model, loader, device, cfg.get("task", "multiclass"))
        split_results = {}
        # внутренний прогресс по метрикам
        for metric in metrics:
            v = metric.score(y_true, logits)
            split_results[metric.name()] = v
        results_all[split_name] = split_results

        # <-- фикс: передаём split_name
        _log_metrics_csv(metrics_csv, epoch, split_name, split_results)

        pretty = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in split_results.items())
        print(f"[Metrics @ epoch {epoch}] {split_name}: {pretty}")

    return results_all


def evaluate_split_metrics(model: torch.nn.Module,
                           loader: torch.utils.data.DataLoader,
                           device: torch.device,
                           cfg: dict) -> Dict[str, float]:
    metrics = build_metrics_from_config(cfg)
    if not metrics:
        return {}

    logits, y_true = _collect_logits_targets(model, loader, device, cfg.get("task", "multiclass"))
    return {metric.name(): metric.score(y_true, logits) for metric in metrics}
