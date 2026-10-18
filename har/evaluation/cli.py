from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from har.paths import project_root

ROOT = project_root()

from har.evaluation.temporal.evaluate import evaluate_temporal_pooling


DEFAULT_EVAL_CONFIG = ROOT / "configs" / "evaluate" / "temporal_pooling.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Evaluation config must be a mapping, got {type(data).__name__}: {path}")
    return data


def _as_list(value: Any, *, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(f"{name} must be a string or a list, got {type(value).__name__}")


def _build_eval_kwargs(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    pooling = config.get("pooling", {})
    smoothing = config.get("smoothing", {})
    segment = config.get("segments", {})
    inference = config.get("inference", {})

    if not isinstance(pooling, dict) or not isinstance(smoothing, dict):
        raise TypeError("Config sections 'pooling' and 'smoothing' must be mappings")
    if not isinstance(segment, dict) or not isinstance(inference, dict):
        raise TypeError("Config sections 'segments' and 'inference' must be mappings")

    return {
        "split": args.split or str(config.get("split", "test")),
        "device": args.device or inference.get("device"),
        "checkpoint_name": args.checkpoint or str(inference.get("checkpoint", "best.pth")),
        "batch_size": args.batch_size if args.batch_size is not None else inference.get("batch_size"),
        "pooling_methods": _as_list(pooling.get("methods", ["max"]), name="pooling.methods"),
        "smoothing_methods": _as_list(smoothing.get("methods", ["moving_average"]), name="smoothing.methods"),
        "topk": int(pooling.get("topk", 3)),
        "moving_average_kernel": int(smoothing.get("moving_average_kernel", 5)),
        "ema_alpha": float(smoothing.get("ema_alpha", 0.3)),
        "median_kernel": int(smoothing.get("median_kernel", 5)),
        "hysteresis_threshold": float(smoothing.get("hysteresis_threshold", 0.75)),
        "hysteresis_margin": float(smoothing.get("hysteresis_margin", 0.0)),
        "hysteresis_decay": float(smoothing.get("hysteresis_decay", 0.9)),
        "segment_decision_methods": _as_list(segment.get("decision_methods", ["mean_prob"]), name="segments.decision_methods"),
        "segment_min_confidence": float(segment.get("min_confidence", 0.0)),
        "segment_min_coverage": float(segment.get("min_coverage", 0.5)),
        "segment_min_length": int(segment.get("min_length", 1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate temporal pooling and episode decisions for a trained HAR run.",
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing config.yaml and checkpoint.")
    parser.add_argument(
        "--eval-config",
        default=str(DEFAULT_EVAL_CONFIG),
        help="YAML config with pooling, smoothing, segment, and inference settings.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: <run-dir>/temporal_eval.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint filename from eval config.")
    parser.add_argument("--split", default=None, choices=("train", "val", "test", "all"), help="Override split from eval config.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    eval_config = _load_yaml(Path(args.eval_config))
    output_dir = Path(args.out_dir) if args.out_dir else run_dir / str(eval_config.get("output_dir", "temporal_eval"))

    metrics = evaluate_temporal_pooling(
        run_dir=run_dir,
        output_dir=output_dir,
        **_build_eval_kwargs(eval_config, args),
    )

    print(metrics.to_string(index=False))
    print(f"\nSaved temporal evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
