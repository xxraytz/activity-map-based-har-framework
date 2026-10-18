import argparse
import json
import os
import re
from collections import defaultdict
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd


ACTOR_RE = re.compile(r"Data(?P<actor>\d+)")


def actor_id_from_path(path: Path) -> int:
    match = ACTOR_RE.search(str(path))
    if not match:
        raise ValueError(f"Cannot extract actor id from {path}")
    return int(match.group("actor"))


def pad_data_prefix(name: str, width: int = 3) -> str:
    pattern = re.compile(r"(data)(\d+)", flags=re.IGNORECASE)
    return pattern.sub(lambda match: f"{match.group(1)}{int(match.group(2)):0{width}d}", name)


def flatten_annotations(segments: list) -> np.ndarray:
    segments = sorted(segments, key=lambda item: item[1][0])
    max_frame = max(end for _, (_, end) in segments)
    labels = np.empty(max_frame, dtype=object)
    for label, (start, end) in segments:
        labels[start - 1:end] = label
    return labels


def read_tst_skeleton(path: Path, coords: int = 3) -> np.ndarray:
    frames = []
    with path.open("r") as file:
        for line in file:
            if "frame" not in line.lower():
                continue
            joints = []
            for _ in range(17):
                values = [float(item) for item in file.readline().split(", ")[:coords]]
                joints.append(values)
            frames.append(joints)
    if not frames:
        raise ValueError(f"No skeleton frames found in {path}")
    return np.asarray(frames, dtype=np.float32)


def preprocess_skeleton(frames: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return frames

    processed = frames.copy()
    if mode in {"root_center", "root_center_scale"}:
        processed = processed - processed[:, :1, :]

    if mode == "root_center_scale":
        left_foot = processed[:, 12, :]
        right_foot = processed[:, 15, :]
        head = processed[:, 3, :]
        height = np.maximum(
            np.linalg.norm(head - left_foot, axis=1),
            np.linalg.norm(head - right_foot, axis=1),
        )
        height = np.maximum(height, 1e-6)
        processed = processed / height[:, None, None]

    return processed


def make_sequence_data(frames: np.ndarray, include_velocity: bool) -> np.ndarray:
    features = frames.reshape(frames.shape[0], -1)
    if include_velocity:
        velocity = np.zeros_like(features)
        velocity[1:] = features[1:] - features[:-1]
        features = np.concatenate([features, velocity], axis=1)
    return features.T[:, None, :].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create raw TST skeleton NPZ dataset for HAR baselines.")
    parser.add_argument("--source-dir", default="../datasets/TST Fall Detection")
    parser.add_argument("--annotations", default="../datasets/annotations_240823.json")
    parser.add_argument("--out-dir", default="data/tst-fall-detection-skeleton-xyz")
    parser.add_argument("--coords", type=int, default=3, choices=[2, 3])
    parser.add_argument(
        "--preprocess",
        default="root_center_scale",
        choices=["raw", "root_center", "root_center_scale"],
    )
    parser.add_argument("--include-velocity", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.annotations).open("r") as file:
        annotations = json.load(file)

    rows = []
    counters = defaultdict(lambda: count(1))
    skeleton_files = sorted(source_dir.rglob("FileSkeleton.txt"))
    if not skeleton_files:
        raise FileNotFoundError(f"No FileSkeleton.txt files found under {source_dir}")

    for skeleton_file in skeleton_files:
        sequence_name = skeleton_file.parent.parent.name.lower()
        key = pad_data_prefix(sequence_name)
        if key not in annotations:
            raise KeyError(f"Missing annotation key {key} for {skeleton_file}")

        frames = read_tst_skeleton(skeleton_file, coords=args.coords)
        frames = preprocess_skeleton(frames, args.preprocess)
        data = make_sequence_data(frames, include_velocity=args.include_velocity)
        labels = flatten_annotations(annotations[key])

        if data.shape[-1] != len(labels):
            length = min(data.shape[-1], len(labels))
            data = data[:, :length]
            labels = labels[:length]

        actor = actor_id_from_path(skeleton_file)
        seq_id = next(counters[actor])
        filename = f"actor_{actor:04d}_seq_{seq_id:05d}.npz"
        filepath = out_dir / filename
        np.savez(filepath, data=data, labels=labels)
        rows.append({"filepath": str(filepath), "actor_id": actor, "seq_id": seq_id})

    manifest = pd.DataFrame(rows).sort_values(["actor_id", "seq_id"]).reset_index(drop=True)
    manifest.to_parquet(out_dir / "manifest.parquet", index=False)

    metadata = {
        "source_dir": str(source_dir),
        "annotations": str(args.annotations),
        "coords": args.coords,
        "preprocess": args.preprocess,
        "include_velocity": args.include_velocity,
        "num_sequences": len(rows),
        "feature_count": int(data.shape[0]),
    }
    with (out_dir / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved {len(rows)} sequences to {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.parquet'}")


if __name__ == "__main__":
    main()
