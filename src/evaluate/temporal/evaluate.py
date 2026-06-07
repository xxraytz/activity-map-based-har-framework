from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from .data import build_class_mapping
from .inference import iter_window_predictions, load_model_from_run
from .metrics import (
    aggregate_results,
    aggregate_segment_results,
    per_sequence_row,
    score_timeline,
    score_segment_rows,
    segment_decisions,
    write_classification_report,
    write_confusion_matrix,
    write_segment_confusion_matrix,
)
from .pooling import apply_pooling
from .smoothing import apply_smoothing


def load_run_config(run_dir: str | Path) -> dict:
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def evaluate_temporal_pooling(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    split: str,
    device: str | None,
    checkpoint_name: str,
    batch_size: int | None,
    pooling_methods: list[str],
    smoothing_methods: list[str],
    topk: int,
    moving_average_kernel: int,
    ema_alpha: float,
    median_kernel: int,
    hysteresis_threshold: float,
    hysteresis_margin: float,
    hysteresis_decay: float,
    segment_decision_methods: list[str] | None = None,
    segment_min_confidence: float = 0.0,
    segment_min_coverage: float = 0.5,
    segment_min_length: int = 1,
) -> pd.DataFrame:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_run_config(run_dir)
    torch_device = torch.device(device or cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    inference_batch_size = int(batch_size or cfg.get("batch_size", 128))
    mapping = build_class_mapping(cfg)

    model = load_model_from_run(run_dir, cfg, torch_device, checkpoint_name)
    predictions = iter_window_predictions(model, cfg, split, torch_device, inference_batch_size)

    per_sequence_rows: list[dict] = []
    segment_rows: list[dict] = []
    aggregate_by_method: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: {
        "y_true": [],
        "y_pred": [],
    })
    segment_decision_methods = segment_decision_methods or ["mean_prob"]

    for pred in tqdm(predictions, desc="Pooling timelines", dynamic_ncols=True):
        for pooling_name in pooling_methods:
            pooled = apply_pooling(pred, pooling_name, topk=topk)
            for smoothing_name in smoothing_methods:
                smoothed = apply_smoothing(
                    pooled,
                    smoothing_name,
                    moving_average_kernel=moving_average_kernel,
                    ema_alpha=ema_alpha,
                    median_kernel=median_kernel,
                    hysteresis_threshold=hysteresis_threshold,
                    hysteresis_margin=hysteresis_margin,
                    hysteresis_decay=hysteresis_decay,
                )
                result, _, _ = score_timeline(pred.y_true, smoothed, mapping, pooling_name, smoothing_name)
                per_sequence_rows.append(per_sequence_row(pred, pooling_name, smoothing_name, result))

                valid = smoothed.valid_mask
                aggregate_by_method[(pooling_name, smoothing_name)]["y_true"].append(pred.y_true[valid])
                aggregate_by_method[(pooling_name, smoothing_name)]["y_pred"].append(smoothed.pred[valid])

                for decision_name in segment_decision_methods:
                    segment_rows.extend(segment_decisions(
                        pred=pred,
                        timeline=smoothed,
                        mapping=mapping,
                        pooling=pooling_name,
                        smoothing=smoothing_name,
                        decision=decision_name,
                        min_confidence=segment_min_confidence,
                        min_coverage=segment_min_coverage,
                        min_segment_length=segment_min_length,
                    ))

    aggregate_results_rows = []
    for (pooling_name, smoothing_name), values in aggregate_by_method.items():
        y_true = np.concatenate(values["y_true"]) if values["y_true"] else np.array([], dtype=np.int64)
        y_pred = np.concatenate(values["y_pred"]) if values["y_pred"] else np.array([], dtype=np.int64)
        pseudo_timeline = type("Timeline", (), {
            "valid_mask": np.ones_like(y_true, dtype=bool),
            "pred": y_pred,
        })()
        result, cm, report = score_timeline(y_true, pseudo_timeline, mapping, pooling_name, smoothing_name)
        aggregate_results_rows.append(result)

        stem = f"{pooling_name}__{smoothing_name}"
        write_confusion_matrix(output_dir / f"confusion_matrix__{stem}.csv", cm, mapping.class_names)
        write_classification_report(output_dir / f"classification_report__{stem}.json", report)

    metrics_df = aggregate_results(aggregate_results_rows).sort_values(
        ["f1_macro", "accuracy"],
        ascending=[False, False],
    )
    metrics_df.to_csv(output_dir / "temporal_metrics.csv", index=False)
    pd.DataFrame(per_sequence_rows).to_csv(output_dir / "per_sequence_metrics.csv", index=False)

    segment_results_rows = []
    segment_df = pd.DataFrame(segment_rows)
    if not segment_df.empty:
        group_columns = ["pooling", "smoothing", "decision", "min_confidence", "min_coverage"]
        for keys, group in segment_df.groupby(group_columns, sort=False):
            pooling_name, smoothing_name, decision_name, min_confidence, min_coverage = keys
            result, cm, report = score_segment_rows(
                group.to_dict("records"),
                mapping,
                pooling_name,
                smoothing_name,
                decision_name,
                float(min_confidence),
                float(min_coverage),
            )
            segment_results_rows.append(result)
            stem = f"{pooling_name}__{smoothing_name}__{decision_name}"
            write_segment_confusion_matrix(output_dir / f"segment_confusion_matrix__{stem}.csv", cm, mapping.class_names)
            write_classification_report(output_dir / f"segment_classification_report__{stem}.json", report)

    aggregate_segment_results(segment_results_rows).sort_values(
        ["f1_macro", "accuracy"],
        ascending=[False, False],
    ).to_csv(output_dir / "segment_metrics.csv", index=False)
    segment_df.to_csv(output_dir / "segment_decisions.csv", index=False)

    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint_name,
        "split": split,
        "pooling_methods": pooling_methods,
        "smoothing_methods": smoothing_methods,
        "class_names": mapping.class_names,
        "num_sequences": len(predictions),
        "batch_size": inference_batch_size,
        "segment_decision_methods": segment_decision_methods,
        "segment_min_confidence": segment_min_confidence,
        "segment_min_coverage": segment_min_coverage,
        "segment_min_length": segment_min_length,
    }
    (output_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))

    return metrics_df
