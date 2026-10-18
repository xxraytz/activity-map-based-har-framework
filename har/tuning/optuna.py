# scripts/optuna_run.py
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import math
import os
import random
from typing import Tuple, Dict, Any

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from optuna.pruners import MedianPruner

from har.data.datasets import get_dataloaders
from har.models.factory import build_model
from har.training.engine import train_epoch, validate
from har.evaluation.metrics.metrics import build_metric


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = max(0.0, min(1.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def collect_logits_targets(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)                                  # (B, C)
            logits_list.append(out.detach().cpu().numpy())
            targets.append(y.argmax(dim=-1).cpu().numpy())
    if not logits_list:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int64)
    return np.concatenate(logits_list), np.concatenate(targets)


def objective(trial: optuna.Trial,
              base_cfg: Dict[str, Any],
              device: torch.device,
              max_epochs: int,
              optimize: str | None,
              metric_params: Dict[str, Any],
              repeats: int,
              base_seed: int,
              include_window_stride: bool,
              model_candidates: list[str] | None = None) -> float:
    # копируем конфиг
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base_cfg.items()}

    # фиксируем отсутствие предобучения
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["pretrained"] = False
    if model_candidates:
        chosen = trial.suggest_categorical("model_name", model_candidates)
        model_cfg["name"] = chosen
    # иначе используем то, что уже в cfg['model']['name']
    cfg["model"] = model_cfg

    # гиперпараметры поиска
    cfg["lr"] = trial.suggest_float("lr", 5e-7, 5e-3, log=True)
    cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

    if include_window_stride:
        data_cfg = dict(cfg.get("data", {}))
        data_cfg["window_stride"] = trial.suggest_categorical("window_stride", [1, 2, 4, 8])
        cfg["data"] = data_cfg

    total_epochs = min(int(cfg.get("epochs", 50)), int(max_epochs))

    vals = []
    for rep in range(repeats):
        set_seed(base_seed + rep)

        train_loader, val_loader, _ = get_dataloaders(cfg)
        model = build_model(cfg).to(device)

        opt = torch.optim.AdamW(model.parameters(),
                                lr=float(cfg["lr"]),
                                weight_decay=float(cfg["weight_decay"]))
        sch = make_scheduler(opt,
                             warmup_epochs=max(1, total_epochs // 10),
                             total_epochs=total_epochs)

        best_val = float("inf")
        best_metric = -float("inf")

        for ep in range(1, total_epochs + 1):
            _ = train_epoch(model, train_loader, opt, device, epoch=ep, epochs=total_epochs)
            val_loss = validate(model, val_loader, device)
            sch.step()

            global_step = rep * total_epochs + ep
            trial.report(val_loss, global_step)

            if trial.should_prune():
                raise optuna.TrialPruned()

            best_val = min(best_val, val_loss)

            if optimize and optimize != "val_loss":
                logits, y_true = collect_logits_targets(model, val_loader, device)
                metric = build_metric(optimize, cfg={"metrics": {"params": metric_params or {}}})
                mval = metric.score(y_true, logits)
                best_metric = max(best_metric, mval)

        vals.append(best_val if (not optimize or optimize == "val_loss") else best_metric)

    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--study-name", type=str, default="har_tuning")
    parser.add_argument("--storage", type=str, required=True,
                        help='Напр., "sqlite:///optuna.db" (обязательно для сохранения исследования)')
    parser.add_argument("--direction", type=str, default="minimize",
                        choices=["minimize", "maximize"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--optimize", type=str, default="val_loss",
                        help='Целевая: "val_loss" или метрика ("f1_macro", "accuracy", "top_k_accuracy@5")')
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--report-dir", type=str, default="optuna_reports")
    parser.add_argument("--include-window-stride", action="store_true")
    parser.add_argument(
        "--model-choices",
        type=str,
        default="",
        help="Список timm-моделей через запятую для перебора, например: "
            "'swin_tiny_patch4_window7_224,swin_small_patch4_window7_224,vit_tiny_patch16_224'"
    )

    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    cli_models = [m.strip() for m in args.model_choices.split(",") if m.strip()]
    cfg_models = []
    if isinstance(base_cfg.get("optuna", {}).get("model_names", None), list):
        cfg_models = [str(m) for m in base_cfg["optuna"]["model_names"]]
    model_candidates = cli_models or cfg_models  # приоритет у CLI


    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=args.direction,
        pruner=pruner,
    )

    def _objective(trial: optuna.Trial) -> float:
        torch.cuda.empty_cache()
        try:
            return objective(
                trial=trial,
                base_cfg=base_cfg,
                device=device,
                max_epochs=args.max_epochs,
                optimize=args.optimize,
                metric_params={"top_k": args.top_k},
                repeats=args.repeats,
                base_seed=args.seed,
                include_window_stride=args.include_window_stride,
                model_candidates=model_candidates if model_candidates else None,
            )
        finally:
            torch.cuda.empty_cache()

    study.optimize(_objective, n_trials=args.trials, gc_after_trial=True)

    # сохраняем CSV всех трейлов
    df = study.trials_dataframe(attrs=(
        "number", "value", "state", "params", "user_attrs", "datetime_start", "datetime_complete"
    ))
    csv_path = os.path.join(args.report_dir, f"{args.study_name}_trials.csv")
    df.to_csv(csv_path, index=False)

    # сохраняем лучшие параметры
    best_cfg_path = os.path.join(args.report_dir, f"best_params_{args.study_name}.yaml")
    with open(best_cfg_path, "w") as f:
        yaml.safe_dump(study.best_params, f)

    print("=== Best Trial ===")
    print("Value:", study.best_value)
    print("Params:", study.best_params)
    print(f"Saved: {csv_path}")
    print(f"Saved: {best_cfg_path}")


if __name__ == "__main__":
    main()
