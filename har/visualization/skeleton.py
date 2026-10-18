"""Draw real TST skeleton poses for the ten illustrated held-out videos.

Run inside the TSU container with the original TST source data available:

python scripts/illustrate_tst_skeleton_examples.py \
  --provenance data/tst-video8-source-aware-5ch32/provenance.csv \
  --examples reports/tst-video8-source-aware-5ch32-three-seeds/illustrations/examples.csv \
  --output-dir reports/tst-video8-source-aware-5ch32-three-seeds/illustrations
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Joint order follows FileSkeleton.txt: torso; left and right arms; left and
# right legs. Joint 16 is the additional upper-torso point.
EDGES = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6),
         (2, 7), (7, 8), (8, 9), (0, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (1, 16)]
FRACTIONS = (0.05, 0.275, 0.5, 0.725, 0.95)


def load_skeleton(path: Path) -> np.ndarray:
    frames = []
    with path.open() as stream:
        for line in stream:
            if not line.lower().startswith("frame "):
                continue
            joints = [tuple(float(value) for value in stream.readline().split(", ")[:3])
                      for _ in range(17)]
            frames.append(joints)
    result = np.asarray(frames, dtype=np.float32)
    if result.ndim != 3 or result.shape[1:] != (17, 3):
        raise ValueError(f"Unexpected skeleton shape: {path}: {result.shape}")
    return result


def draw_group(rows: list[dict], source_files: dict[str, Path], output: Path,
               title: str) -> None:
    fig, axes = plt.subplots(len(rows), len(FRACTIONS), figsize=(16, 15))
    for row_index, row in enumerate(rows):
        poses = load_skeleton(source_files[row["source_key"]])
        n = len(poses)
        assert n == int(row["frames"]), (row["source_key"], n, row["frames"])
        # Keep one coordinate system within each video to make motion visible.
        selected = np.array([min(n - 1, round((n - 1) * f)) for f in FRACTIONS])
        xy_all = poses[:, :, :2]
        x_lo, x_hi = np.nanpercentile(xy_all[:, :, 0], [0.5, 99.5])
        y_lo, y_hi = np.nanpercentile(xy_all[:, :, 1], [0.5, 99.5])
        cx, cy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
        span = max(x_hi - x_lo, y_hi - y_lo) * 1.2
        for col_index, frame in enumerate(selected):
            ax = axes[row_index, col_index]
            xy = poses[frame, :, :2]
            for a, b in EDGES:
                ax.plot(xy[[a, b], 0], xy[[a, b], 1], color="#263e58",
                        linewidth=2.2, solid_capstyle="round")
            ax.scatter(xy[:, 0], xy[:, 1], s=13, color="#1c85ad", zorder=3)
            ax.scatter(xy[3, 0], xy[3, 1], s=35, color="#de6654", zorder=4)
            ax.set_xlim(cx - span / 2, cx + span / 2)
            ax.set_ylim(cy - span / 2, cy + span / 2)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#dbe2e8")
            ax.set_title(f"кадр {frame + 1}/{n}", fontsize=10, pad=5)
        axes[row_index, 0].set_ylabel(
            f"{row['source_key']}\nистина: {row['true_class']}\nответ: {row['predicted_class']}",
            fontsize=10, rotation=0, ha="right", va="center", labelpad=10)
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.text(0.5, 0.01,
             "Пять моментов каждого исходного скелетного ролика (проекция XY). "
             "Класс видео решается по всем кадрам, как на графиках вероятностей и голосов.",
             ha="center", fontsize=10, color="#526273")
    fig.subplots_adjust(left=0.23, right=0.98, top=0.96, bottom=0.04,
                        hspace=0.45, wspace=0.08)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    with args.provenance.open(newline="") as stream:
        sources = {row["source_key"]: Path(row["source_skeleton"])
                   for row in csv.DictReader(stream)}
    with args.examples.open(newline="") as stream:
        examples = list(csv.DictReader(stream))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind, title in [("good", "Пять правильных примеров: исходные скелетные кадры"),
                        ("bad", "Пять ошибочных примеров: исходные скелетные кадры")]:
        rows = [row for row in examples if row["group"] == kind]
        assert len(rows) == 5
        draw_group(rows, sources, args.output_dir / f"{kind}_skeletons.png", title)


if __name__ == "__main__":
    main()
