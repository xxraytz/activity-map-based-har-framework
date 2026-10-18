"""Create source-aware TST eight-action frame labels and 11 LOPO manifests.

The activity maps are copied without changing a single input value. Original
video identity is reconstructed from actor, skeleton length, and old labels.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from har.paths import project_root

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit


CLASS_ORDER = (
    "no_class",
    "adl_sit",
    "adl_grasp",
    "adl_walk",
    "adl_lay",
    "fall_front",
    "fall_back",
    "fall_side",
    "fall_endupsit",
)

POSITIVE_LABELS = {
    "adl_sit": {"stand-to-sit", "sitting", "falling"},
    "adl_grasp": {"grasp", "ungrasp"},
    "adl_walk": {"walking"},
    "adl_lay": {"stand-to-sit", "sit-to-lay", "laying", "falling"},
    "fall_front": {"falling", "post-fall-laying"},
    "fall_back": {"falling", "post-fall-laying"},
    "fall_side": {"falling", "post-fall-laying"},
    "fall_endupsit": {"falling", "sitting"},
}


def original_records(root: Path, annotations: dict) -> list[dict]:
    records = []
    for skeleton in sorted(root.rglob("FileSkeleton.txt")):
        folder = skeleton.parent.parent.name
        match = re.fullmatch(r"Data(\d+)_(ADL|Fall)_(\w+)_(\d+)", folder, re.I)
        if match is None:
            raise ValueError(f"Unexpected TST folder: {folder}")
        actor = int(match.group(1))
        video_class = f"{match.group(2).lower()}_{match.group(3).lower()}"
        source_key = f"data{actor:03d}_{video_class}_{match.group(4)}"
        if source_key not in annotations:
            raise ValueError(f"Missing source annotation: {source_key}")
        n_skeleton = sum(line.startswith("frame ") for line in skeleton.open())
        parts = sorted(annotations[source_key], key=lambda pair: pair[1][0])
        n_annotated = max(int(bounds[1]) for _, bounds in parts)
        old_labels = [None] * n_annotated
        for label, bounds in parts:
            start, end = map(int, bounds)
            for frame in range(start - 1, end):
                if old_labels[frame] is not None:
                    raise ValueError(f"Overlapping annotation: {source_key} frame {frame + 1}")
                old_labels[frame] = str(label)
        if any(label is None for label in old_labels):
            raise ValueError(f"Annotation gap: {source_key}")
        if n_skeleton > n_annotated:
            raise ValueError(f"Skeleton longer than annotations: {source_key}")
        records.append({
            "source_key": source_key,
            "video_class": video_class,
            "actor_id": actor,
            "skeleton_frames": n_skeleton,
            "annotated_frames": n_annotated,
            "old_labels": old_labels,
            "skeleton_file": str(skeleton),
        })
    if len(records) != 264:
        raise ValueError(f"Expected 264 TST sources, found {len(records)}")
    if Counter(r["video_class"] for r in records) != Counter({name: 33 for name in CLASS_ORDER[1:]}):
        raise ValueError("Source class counts differ from 33 per class")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=project_root())
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-name", default="tst-video8-source-aware-5ch32")
    parser.add_argument("--maps-dir", type=Path, default=Path("data/tst-fall-detection-3d-parts-32"))
    parser.add_argument("--template", type=Path, default=Path("configs/tst-video8-template.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    source_dir = project / args.maps_dir
    source_manifest = pd.read_parquet(source_dir / "manifest.parquet")
    annotations = json.loads(args.annotations.read_text())
    originals = original_records(args.source_root, annotations)
    by_actor = {actor: [r for r in originals if r["actor_id"] == actor] for actor in range(1, 12)}
    output_dir = project / "data" / args.out_name
    config_dir = project / "configs" / args.out_name
    template = yaml.safe_load((project / args.template).read_text())
    rows = []
    counts = Counter()
    mismatches = []

    for item in source_manifest.itertuples(index=False):
        old_path = project / item.filepath
        with np.load(old_path, allow_pickle=True) as npz:
            data = npz["data"]
            old_labels = [str(label) for label in npz["labels"]]
        n = len(old_labels)
        if data.shape != (5, 32, n):
            raise ValueError(f"Unexpected map shape: {old_path}: {data.shape}")
        matches = [r for r in by_actor[int(item.actor_id)]
                   if r["skeleton_frames"] == n and r["old_labels"][:n] == old_labels]
        if len(matches) != 1:
            mismatches.append({"filepath": item.filepath, "matches": [r["source_key"] for r in matches]})
            continue
        source = matches[0]
        video_class = source["video_class"]
        selected = POSITIVE_LABELS[video_class]
        new_labels = np.asarray([video_class if label in selected else "no_class" for label in old_labels])
        if not (new_labels == video_class).any():
            raise ValueError(f"No target frames: {source['source_key']}")
        new_relpath = f"data/{args.out_name}/{old_path.name}"
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(project / new_relpath, data=data, labels=new_labels)
        rows.append({
            "filepath": new_relpath,
            "actor_id": int(item.actor_id),
            "seq_id": int(item.seq_id),
            "source_key": source["source_key"],
            "video_class": video_class,
            "source_skeleton": source["skeleton_file"],
            "n_frames": n,
            "n_positive": int((new_labels == video_class).sum()),
            "n_background": int((new_labels == "no_class").sum()),
            "annotation_truncated": source["annotated_frames"] > n,
        })
        counts[video_class] += 1

    if mismatches:
        raise ValueError(f"Unmatched maps: {mismatches}")
    if len(rows) != 264 or len({r["source_key"] for r in rows}) != 264:
        raise ValueError("NPZ to source mapping is not one-to-one")
    if counts != Counter({name: 33 for name in CLASS_ORDER[1:]}):
        raise ValueError(f"Uneven class counts: {counts}")
    manifest = pd.DataFrame(rows)
    print("Source mapping:", len(rows), "unique video classes:", dict(counts))
    print("Truncated annotations:", int(manifest.annotation_truncated.sum()))
    print("Target/background frames:", int(manifest.n_positive.sum()), int(manifest.n_background.sum()))
    print("Minimum positive frames:", int(manifest.n_positive.min()))
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_dir / "provenance.parquet", index=False)
    manifest.to_csv(output_dir / "provenance.csv", index=False)
    (output_dir / "label_policy.json").write_text(json.dumps({
        "class_order": CLASS_ORDER,
        "positive_raw_labels": {key: sorted(value) for key, value in POSITIVE_LABELS.items()},
        "rule": "Each source-conditioned positive raw label becomes its source video class; all other frames become no_class.",
    }, indent=2))
    for actor in range(1, 12):
        fold = manifest.copy()
        fold["split"] = "test"
        other = fold[fold.actor_id != actor]
        stratifier = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, val_idx = next(stratifier.split(other, other.video_class))
        fold.loc[other.iloc[train_idx].index, "split"] = "train"
        fold.loc[other.iloc[val_idx].index, "split"] = "val"
        assert len(fold[fold.split == "test"]) == 24
        assert Counter(fold[fold.split == "test"].video_class) == Counter({name: 3 for name in CLASS_ORDER[1:]})
        fold.to_parquet(output_dir / f"manifest-lopo-{actor:02d}.parquet", index=False)
        fold_dir = config_dir / f"lopo-{actor:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for seed in (42, 43, 44):
            cfg = yaml.safe_load(yaml.safe_dump(template))
            cfg["seed"] = seed
            cfg["num_classes"] = len(CLASS_ORDER)
            cfg["data"]["seq_dir"] = f"data/{args.out_name}"
            cfg["data"]["manifest"] = f"data/{args.out_name}/manifest-lopo-{actor:02d}.parquet"
            cfg["data"]["split_mode"] = "manifest"
            cfg["data"]["class_names"] = {name: name for name in CLASS_ORDER}
            cfg["data"].pop("lopo", None)
            cfg["metrics"]["eval_every"] = 0  # final checkpoint still receives complete evaluation
            cfg["device"] = args.device
            (fold_dir / f"seed-{seed}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Wrote new dataset and 33 LOPO configs to", output_dir, config_dir)


if __name__ == "__main__":
    main()
