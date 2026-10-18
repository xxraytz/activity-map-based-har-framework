"""Evaluate all 33 TST video8 runs, preserving frame and video stages.

Every listed whole-video rule uses only predictions for that video. The rules
are reported side by side as diagnostics; this script does not choose a winner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from har.paths import project_root

ROOT = project_root()

from har.evaluation.temporal.inference import iter_window_predictions, load_model_from_run
from har.evaluation.temporal.pooling import apply_pooling
from har.evaluation.temporal.smoothing import apply_smoothing

NAME = "tst-video8-source-aware-5ch32"
CLASSES = ["no_class", "adl_sit", "adl_grasp", "adl_walk", "adl_lay",
           "fall_front", "fall_back", "fall_side", "fall_endupsit"]
SEEDS = (42, 43, 44)
METHODS = (
    "full_video_mean_prob",
    "top25_mean_prob",
    "majority_nonbackground",
    "hysteresis_majority_nonbackground",
)


def summarize_by_seed(rows: list[dict], value_columns: tuple[str, ...], group: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    groups = [(None, df)] if group is None else list(df.groupby(group, sort=False))
    summary = []
    for group_name, part in groups:
        item = {} if group is None else {group: group_name}
        for column in value_columns:
            values = part[column].to_numpy(dtype=float)
            item[f"{column}_mean"] = float(values.mean())
            item[f"{column}_std_across_3_seeds"] = float(values.std(ddof=1))
            item[f"{column}_min"] = float(values.min())
            item[f"{column}_max"] = float(values.max())
        summary.append(item)
    return pd.DataFrame(summary)


def frame_stage(out: Path, device_override: str | None = None) -> None:
    (out / "frames").mkdir(parents=True, exist_ok=True)
    per_run = []
    per_seed = []
    old_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for seed in SEEDS:
            seed_true, seed_pred = [], []
            (out / "frames" / f"seed-{seed}").mkdir(parents=True, exist_ok=True)
            for actor in range(1, 12):
                run_dir = ROOT / "runs" / NAME / f"lopo-{actor:02d}-seed-{seed}"
                if not (run_dir / "best.pth").exists():
                    raise FileNotFoundError(f"Missing trained checkpoint: {run_dir}")
                cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
                if int(cfg["seed"]) != seed or list(cfg["data"]["class_names"]) != CLASSES:
                    raise ValueError(f"Incorrect config for actor {actor}, seed {seed}")
                manifest = pd.read_parquet(ROOT / cfg["data"]["manifest"])
                truth = manifest.loc[manifest.split == "test"].set_index("filepath")
                if len(truth) != 24 or set(truth.actor_id) != {actor}:
                    raise ValueError(f"Invalid LOPO test split: actor {actor}")
                device = torch.device(device_override or (cfg["device"] if torch.cuda.is_available() else "cpu"))
                model = load_model_from_run(run_dir, cfg, device, "best.pth")
                predictions = iter_window_predictions(model, cfg, "test", device, int(cfg["batch_size"]))
                if len(predictions) != 24:
                    raise ValueError(f"Expected 24 videos, got {len(predictions)}")
                fold_true, fold_pred = [], []
                for pred in predictions:
                    info = truth.loc[str(pred.record.filepath)]
                    pooled = apply_pooling(pred, "max")
                    soft = apply_smoothing(pooled, "moving_average", moving_average_kernel=5)
                    hyst = apply_smoothing(pooled, "hysteresis",
                                           hysteresis_threshold=0.75,
                                           hysteresis_margin=0.0,
                                           hysteresis_decay=0.9)
                    valid = soft.valid_mask & hyst.valid_mask
                    target = pred.y_true[valid].astype(np.int64)
                    hard = soft.pred[valid].astype(np.int64)
                    if len(target) == 0:
                        raise ValueError(f"No valid frames: {info.source_key}")
                    fold_true.extend(target.tolist())
                    fold_pred.extend(hard.tolist())
                    np.savez_compressed(
                        out / "frames" / f"seed-{seed}" / f"{info.source_key}.npz",
                        probs=soft.probs[valid], y_true=target, y_pred=hard,
                        hyst_pred=hyst.pred[valid].astype(np.int64),
                        source_key=info.source_key, video_class=info.video_class,
                        actor_id=actor, seed=seed,
                    )
                seed_true.extend(fold_true)
                seed_pred.extend(fold_pred)
                per_run.append({
                    "seed": seed, "actor_id": actor, "frames": len(fold_true),
                    "accuracy": float(accuracy_score(fold_true, fold_pred)),
                    "f1_macro": float(f1_score(fold_true, fold_pred,
                                                labels=list(range(9)), average="macro", zero_division=0)),
                })
                print(f"seed={seed} actor={actor:02d} frame_accuracy={per_run[-1]['accuracy']:.4f}", flush=True)
            per_seed.append({
                "seed": seed, "frames": len(seed_true),
                "accuracy": float(accuracy_score(seed_true, seed_pred)),
                "f1_macro": float(f1_score(seed_true, seed_pred,
                                            labels=list(range(9)), average="macro", zero_division=0)),
            })
    finally:
        os.chdir(old_cwd)
    if len(per_run) != 33 or any(row["frames"] != 46410 for row in per_seed):
        raise ValueError("Incomplete 33-run frame evaluation")
    pd.DataFrame(per_run).to_csv(out / "frame_per_run.csv", index=False)
    pd.DataFrame(per_seed).to_csv(out / "frame_per_seed.csv", index=False)
    summarize_by_seed(per_seed, ("accuracy", "f1_macro")).to_csv(out / "frame_summary.csv", index=False)
    run_df = pd.DataFrame(per_run)
    pd.DataFrame([{
        "runs": 33,
        "accuracy_mean": float(run_df.accuracy.mean()),
        "accuracy_std_across_33_runs": float(run_df.accuracy.std(ddof=1)),
        "f1_macro_mean": float(run_df.f1_macro.mean()),
        "f1_macro_std_across_33_runs": float(run_df.f1_macro.std(ddof=1)),
    }]).to_csv(out / "frame_summary_across_33_runs.csv", index=False)
    (out / "frame_protocol.json").write_text(json.dumps({
        "seeds": SEEDS, "folds": 11, "classes": CLASSES,
        "windows": "32 frames, stride 1",
        "frame_pooling": "max probability per class across covering windows; row renormalized",
        "frame_smoothing": "moving average of 5 consecutive frames",
        "frame_metric_unit": "frame; per-seed support 46410",
        "hysteresis_diagnostic": {"threshold": 0.75, "margin": 0.0, "decay": 0.9},
    }, indent=2))


def nonbackground_vote(labels: np.ndarray, fallback: int) -> int:
    action_labels = labels[labels > 0]
    if len(action_labels) == 0:
        return fallback
    return int(np.bincount(action_labels, minlength=9)[1:].argmax()) + 1


def video_decisions(probs: np.ndarray, hyst_pred: np.ndarray) -> dict[str, int]:
    full_mean = int(np.argmax(probs[:, 1:].mean(axis=0))) + 1
    k = max(1, math.ceil(probs.shape[0] * 0.25))
    top25 = int(np.argmax(np.sort(probs[:, 1:], axis=0)[-k:].mean(axis=0))) + 1
    return {
        "full_video_mean_prob": full_mean,
        "top25_mean_prob": top25,
        "majority_nonbackground": nonbackground_vote(probs.argmax(axis=1), full_mean),
        "hysteresis_majority_nonbackground": nonbackground_vote(hyst_pred, full_mean),
    }


def video_stage(out: Path) -> None:
    if not (out / "frame_summary.csv").exists():
        raise FileNotFoundError("Run --stage frames first")
    rows = []
    per_run = []
    per_seed = []
    for seed in SEEDS:
        paths = sorted((out / "frames" / f"seed-{seed}").glob("*.npz"))
        if len(paths) != 264:
            raise ValueError(f"Expected 264 videos for seed {seed}, found {len(paths)}")
        for path in paths:
            with np.load(path, allow_pickle=False) as arr:
                probs = arr["probs"]
                hyst = arr["hyst_pred"]
                source_key = str(arr["source_key"])
                video_class = str(arr["video_class"])
                actor = int(arr["actor_id"])
            if probs.ndim != 2 or probs.shape[1] != 9:
                raise ValueError(f"Bad probabilities: {path}")
            decisions = video_decisions(probs, hyst)
            rows.append({
                "seed": seed, "actor_id": actor, "source_key": source_key,
                "true_class": video_class, "frames": len(probs),
                **{method: CLASSES[pred] for method, pred in decisions.items()},
            })
        seed_rows = [r for r in rows if r["seed"] == seed]
        if Counter(r["true_class"] for r in seed_rows) != Counter({name: 33 for name in CLASSES[1:]}):
            raise ValueError(f"Unbalanced test videos in seed {seed}")
        for method in METHODS:
            truth = [CLASSES.index(r["true_class"]) for r in seed_rows]
            guessed = [CLASSES.index(r[method]) for r in seed_rows]
            per_seed.append({
                "seed": seed, "method": method,
                "videos": len(truth), "correct": int(sum(a == b for a, b in zip(truth, guessed))),
                "accuracy": float(accuracy_score(truth, guessed)),
                "f1_macro": float(f1_score(truth, guessed, labels=list(range(1, 9)),
                                            average="macro", zero_division=0)),
            })
            for actor in range(1, 12):
                group = [r for r in seed_rows if r["actor_id"] == actor]
                yt = [CLASSES.index(r["true_class"]) for r in group]
                yp = [CLASSES.index(r[method]) for r in group]
                per_run.append({
                    "seed": seed, "actor_id": actor, "method": method,
                    "videos": len(group), "correct": int(sum(a == b for a, b in zip(yt, yp))),
                    "accuracy": float(accuracy_score(yt, yp)),
                    "f1_macro": float(f1_score(yt, yp, labels=list(range(1, 9)),
                                                average="macro", zero_division=0)),
                })
    pd.DataFrame(rows).to_csv(out / "video_predictions_all_methods.csv", index=False)
    pd.DataFrame(per_run).to_csv(out / "video_per_run.csv", index=False)
    pd.DataFrame(per_seed).to_csv(out / "video_per_seed.csv", index=False)
    summarize_by_seed(per_seed, ("accuracy", "f1_macro"), group="method").to_csv(
        out / "video_summary.csv", index=False)
    run_df = pd.DataFrame(per_run)
    run_summary = []
    for method, group in run_df.groupby("method", sort=False):
        run_summary.append({
            "method": method, "runs": len(group),
            "accuracy_mean": float(group.accuracy.mean()),
            "accuracy_std_across_33_runs": float(group.accuracy.std(ddof=1)),
            "f1_macro_mean": float(group.f1_macro.mean()),
            "f1_macro_std_across_33_runs": float(group.f1_macro.std(ddof=1)),
        })
    pd.DataFrame(run_summary).to_csv(out / "video_summary_across_33_runs.csv", index=False)
    (out / "video_protocol.json").write_text(json.dumps({
        "test_videos_per_seed": 264,
        "methods": {
            "full_video_mean_prob": "argmax over 8 action-class means of all frame probabilities; full-video extension of old episode mean_prob",
            "top25_mean_prob": "argmax over 8 action-class means of that class's highest 25% frame probabilities; original first-run rule",
            "majority_nonbackground": "argmax per frame, discard no_class frames, majority vote among 8 action classes; fallback to full_video_mean_prob",
            "hysteresis_majority_nonbackground": "max-pooled frame probabilities, hysteresis with old server parameters, then discard no_class and vote; fallback to full_video_mean_prob",
        },
        "ground_truth_boundaries_used": False,
        "selection_by_test_result": False,
        "interpretation": "All methods are diagnostics. The original top25 result remains the initial predefined video rule.",
    }, indent=2))
    print(pd.DataFrame(per_seed).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("frames", "videos"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / f"{NAME}-three-seeds")
    parser.add_argument("--device", help="Inference device, e.g. cpu or cuda:0")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "frames":
        frame_stage(args.output, args.device)
    else:
        video_stage(args.output)


if __name__ == "__main__":
    main()
