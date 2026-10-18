from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


NAME_RE = re.compile(r"^A(?P<origin>\d+)N(?P<subject>\d+)-(?P<view>[LMR])$")


def parse_split_file(path: Path) -> dict[str, str]:
    split = {}
    current = None
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value:
            continue
        lower = value.lower()
        if lower.startswith("training"):
            current = "train"
            continue
        if lower.startswith(("validataion", "validation", "test")):
            current = "test"
            continue
        if current:
            for stem in value.split(","):
                if stem.strip():
                    split[stem.strip()] = current
    return split


def parse_labels(path: Path) -> list[tuple[int, int, int, int]]:
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        values = [int(value) for value in line.split(",")]
        if len(values) == 3:
            values.append(-1)
        result.append(tuple(values[:4]))
    return result


def load_skeleton(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] < 150:
        raise ValueError(f"{path}: expected at least 150 values per frame, got {rows.shape}")
    return rows[:, :150].reshape(-1, 2, 25, 3)


def collect_samples(root: Path, split_map: dict[str, str], action_to_idx: dict[int, int]):
    samples = {"train": [], "test": []}
    skipped = []
    for skeleton_path in sorted((root / "skeleton-data").glob("*.txt")):
        stem = skeleton_path.stem
        if not NAME_RE.match(stem):
            raise ValueError(f"Unexpected PKU filename: {skeleton_path.name}")
        split = split_map.get(stem)
        if split not in samples:
            raise ValueError(f"{stem}: missing official split membership")
        frame_count = 0
        has_tracked_body = False
        with skeleton_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                frame_count += 1
                if not has_tracked_body:
                    row = np.fromstring(line, sep=" ", count=150)
                    has_tracked_body = row.size == 150 and np.any(row != 0)
        if not has_tracked_body:
            skipped.append((stem, -1, "all_zero_body_video"))
            continue
        for segment_idx, (action_id, start, end, confidence) in enumerate(
            parse_labels(root / "label" / skeleton_path.name)
        ):
            start = max(0, start)
            end = min(frame_count, end)
            if end <= start:
                skipped.append((stem, segment_idx, "empty_interval"))
                continue
            samples[split].append(
                (skeleton_path, stem, segment_idx, action_to_idx[action_id], start, end, confidence)
            )
    return samples, skipped


def write_split(samples, data_path: Path, label_path: Path, max_frames: int) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data = open_memmap(
        data_path,
        dtype="float32",
        mode="w+",
        shape=(len(samples), 3, max_frames, 25, 2),
    )
    names, labels = [], []
    cached_path = None
    cached_video = None
    for idx, (path, stem, segment_idx, label, start, end, _confidence) in enumerate(samples):
        if path != cached_path:
            cached_video = load_skeleton(path)
            cached_path = path
        clip = cached_video[start:end]
        if clip.shape[0] > max_frames:
            indices = np.linspace(0, clip.shape[0] - 1, max_frames).round().astype(np.int64)
            clip = clip[indices]
        # PKU rows are (T,M,V,C); graph models use (C,T,V,M).
        data[idx, :, : clip.shape[0]] = clip.transpose(3, 0, 2, 1)
        names.append(f"{stem}_seg{segment_idx:03d}")
        labels.append(label)
        if (idx + 1) % 500 == 0:
            print(f"{data_path.name}: {idx + 1}/{len(samples)}", flush=True)
    data.flush()
    with label_path.open("wb") as handle:
        pickle.dump((names, labels), handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PKU-MMD v2 graph-model tensor caches.")
    parser.add_argument("--phase2-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=800)
    args = parser.parse_args()

    action_ids = sorted(
        {
            action_id
            for path in (args.phase2_root / "label").glob("*.txt")
            for action_id, _start, _end, _confidence in parse_labels(path)
        }
    )
    action_to_idx = {action_id: idx for idx, action_id in enumerate(action_ids)}
    metadata = {"action_ids": action_ids, "num_classes": len(action_ids), "max_frames": args.max_frames}

    for protocol, split_file in {
        "csub": "cross_subject_v2.txt",
        "cview": "cross_view_v2.txt",
    }.items():
        output = args.output_dir / protocol
        samples, skipped = collect_samples(
            args.phase2_root,
            parse_split_file(args.phase2_root / "splits" / split_file),
            action_to_idx,
        )
        for split in ("train", "test"):
            write_split(
                samples[split],
                output / f"{split}_data.npy",
                output / f"{split}_label.pkl",
                args.max_frames,
            )
        (output / "metadata.json").write_text(
            json.dumps({**metadata, "protocol": protocol, "counts": {k: len(v) for k, v in samples.items()}, "skipped": skipped}, indent=2)
        )


if __name__ == "__main__":
    main()
