"""Score TST video8 in two explicit stages: frames, then whole videos.

The primary video rule is fixed in advance: for each non-background class,
average its highest 25% of frame probabilities, then choose the largest score.
Ground-truth event boundaries and source filenames are never inference inputs.
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from har.paths import project_root

ROOT = project_root()

from har.evaluation.temporal.inference import iter_window_predictions, load_model_from_run
from har.evaluation.temporal.pooling import apply_pooling
from har.evaluation.temporal.smoothing import apply_smoothing

NAME = "tst-video8-source-aware-5ch32"
CLASSES = ["no_class", "adl_sit", "adl_grasp", "adl_walk", "adl_lay",
           "fall_front", "fall_back", "fall_side", "fall_endupsit"]


def metrics(y_true: list[int], y_pred: list[int], labels: list[int], names: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "support": len(y_true),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0),
    }


def save_confusion(path: Path, y_true: list[int], y_pred: list[int], labels: list[int], names: list[str]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=names, columns=names).to_csv(path)


def frame_stage(output: Path) -> None:
    (output / "frames").mkdir(parents=True, exist_ok=True)
    all_true: list[int] = []
    all_pred: list[int] = []
    rows = []
    old_cwd = Path.cwd()
    os.chdir(ROOT)  # Manifest filepaths are relative to the HAR project.
    try:
        for actor in range(1, 12):
            run_dir = ROOT / "runs" / NAME / f"lopo-{actor:02d}-seed-42"
            if not (run_dir / "finished.ok").exists():
                raise FileNotFoundError(f"Training has not finished: {run_dir}")
            cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
            if list(cfg["data"]["class_names"]) != CLASSES:
                raise ValueError(f"Class order mismatch in fold {actor}")
            manifest = pd.read_parquet(ROOT / cfg["data"]["manifest"])
            truth = manifest.loc[manifest.split == "test"].set_index("filepath")
            if len(truth) != 24 or (truth.actor_id != actor).any():
                raise ValueError(f"Invalid test split in fold {actor}")
            device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
            model = load_model_from_run(run_dir, cfg, device, "best.pth")
            predictions = iter_window_predictions(model, cfg, "test", device, int(cfg["batch_size"]))
            if len(predictions) != 24:
                raise ValueError(f"Expected 24 test videos for actor {actor}, got {len(predictions)}")
            fold_true: list[int] = []
            fold_pred: list[int] = []
            for pred in predictions:
                source = truth.loc[str(pred.record.filepath)]
                timeline = apply_smoothing(
                    apply_pooling(pred, "max"), "moving_average", moving_average_kernel=5)
                valid = timeline.valid_mask
                y_true = pred.y_true[valid].astype(np.int64)
                y_pred = timeline.pred[valid].astype(np.int64)
                if len(y_true) == 0:
                    raise ValueError(f"No valid frames: {source.source_key}")
                fold_true.extend(y_true.tolist())
                fold_pred.extend(y_pred.tolist())
                np.savez_compressed(
                    output / "frames" / f"{source.source_key}.npz",
                    probs=timeline.probs[valid], y_true=y_true, y_pred=y_pred,
                    source_key=source.source_key, video_class=source.video_class,
                    actor_id=actor,
                )
                rows.append({
                    "actor_id": actor, "source_key": source.source_key,
                    "video_class": source.video_class, "frames": len(y_true),
                    "frame_accuracy": float((y_true == y_pred).mean()),
                })
            all_true.extend(fold_true)
            all_pred.extend(fold_pred)
            print(f"Actor {actor:02d}: frame accuracy {accuracy_score(fold_true, fold_pred):.4f}", flush=True)
    finally:
        os.chdir(old_cwd)
    if len(rows) != 264 or len({row["source_key"] for row in rows}) != 264:
        raise ValueError("Frame predictions do not cover 264 unique videos")
    result = metrics(all_true, all_pred, list(range(9)), CLASSES)
    result["method"] = "32-frame windows, max frame pooling, 5-frame moving average"
    (output / "frame_metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(rows).to_csv(output / "frame_per_video.csv", index=False)
    save_confusion(output / "frame_confusion.csv", all_true, all_pred, list(range(9)), CLASSES)
    print("Overall frame accuracy", result["accuracy"], "macro-F1", result["f1_macro"], flush=True)


def video_stage(output: Path) -> None:
    if not (output / "frame_metrics.json").exists():
        raise FileNotFoundError("Run --stage frames first")
    files = sorted((output / "frames").glob("*.npz"))
    if len(files) != 264:
        raise ValueError(f"Expected 264 frame prediction files, found {len(files)}")
    rows = []
    true: list[int] = []
    predicted: list[int] = []
    for path in files:
        with np.load(path, allow_pickle=False) as arr:
            probs = arr["probs"]
            source_key = str(arr["source_key"])
            video_class = str(arr["video_class"])
            actor_id = int(arr["actor_id"])
        if probs.ndim != 2 or probs.shape[1] != 9:
            raise ValueError(f"Invalid frame probabilities: {path}")
        k = max(1, math.ceil(probs.shape[0] * 0.25))
        class_scores = np.sort(probs[:, 1:], axis=0)[-k:].mean(axis=0)
        winner = int(class_scores.argmax()) + 1
        target = CLASSES.index(video_class)
        true.append(target)
        predicted.append(winner)
        rows.append({
            "actor_id": actor_id, "source_key": source_key,
            "video_class": video_class, "predicted_class": CLASSES[winner],
            "correct": winner == target, "frames": probs.shape[0],
            **{f"score_{name}": float(score) for name, score in zip(CLASSES[1:], class_scores)},
        })
    if Counter(row["video_class"] for row in rows) != Counter({name: 33 for name in CLASSES[1:]}):
        raise ValueError("Video classes are not balanced 33 each")
    result = metrics(true, predicted, list(range(1, 9)), CLASSES[1:])
    result["method"] = "top 25% mean of max-pooled, smoothed frame probabilities for each action class"
    result["background_excluded_from_decision"] = True
    (output / "video_metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(rows).to_csv(output / "video_predictions.csv", index=False)
    save_confusion(output / "video_confusion.csv", true, predicted, list(range(1, 9)), CLASSES[1:])
    print("Overall video accuracy", result["accuracy"], "macro-F1", result["f1_macro"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("frames", "videos"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / NAME)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "frames":
        frame_stage(args.output)
    else:
        video_stage(args.output)


if __name__ == "__main__":
    main()
