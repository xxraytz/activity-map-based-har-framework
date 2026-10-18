"""Make whole-video 8-class confusion matrices from saved test predictions.

Run in an environment with matplotlib, for example the TSU training container:

python scripts/plot_tst_video8_confusion.py \
  --predictions reports/tst-video8-source-aware-5ch32-three-seeds/video_predictions_all_methods.csv \
  --output-dir reports/tst-video8-source-aware-5ch32-three-seeds
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


CLASSES = (
    "adl_sit", "adl_grasp", "adl_walk", "adl_lay",
    "fall_front", "fall_back", "fall_side", "fall_endupsit",
)
PREDICTION_COLUMN = "hysteresis_majority_nonbackground"
SEEDS = (42, 43, 44)


def matrix(rows: list[dict]) -> np.ndarray:
    result = np.zeros((8, 8), dtype=int)
    for row in rows:
        result[CLASSES.index(row["true_class"]),
               CLASSES.index(row[PREDICTION_COLUMN])] += 1
    return result


def save_csv(path: Path, values: np.ndarray) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\predicted", *CLASSES, "total"])
        for name, counts in zip(CLASSES, values):
            writer.writerow([name, *map(int, counts), int(counts.sum())])
        writer.writerow(["total", *map(int, values.sum(axis=0)), int(values.sum())])


def draw_panel(ax, values: np.ndarray, title: str) -> None:
    proportions = values / values.sum(axis=1, keepdims=True)
    ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(8), labels=CLASSES, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(8), labels=CLASSES, fontsize=9)
    ax.set_xlabel("Предсказанный класс", fontsize=11)
    ax.set_ylabel("Истинный класс", fontsize=11)
    ax.set_title(title, fontsize=13, pad=11)
    for i in range(8):
        for j in range(8):
            count = int(values[i, j])
            ax.text(j, i, str(count) if count else "·", ha="center", va="center",
                    color="white" if proportions[i, j] > 0.55 else "#1f3145",
                    fontsize=10 if count else 8,
                    fontweight="bold" if count else "normal")
            if i != j and count:
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                       fill=False, edgecolor="#c56b32", linewidth=2))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)
    ax.tick_params(length=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.predictions.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 264 * 3
    assert set(int(row["seed"]) for row in rows) == set(SEEDS)
    assert set(row["true_class"] for row in rows) == set(CLASSES)
    assert set(row[PREDICTION_COLUMN] for row in rows) <= set(CLASSES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    for seed in SEEDS:
        subset = [row for row in rows if int(row["seed"]) == seed]
        assert len(subset) == len({row["source_key"] for row in subset}) == 264
        assert Counter(row["true_class"] for row in subset) == Counter({name: 33 for name in CLASSES})
        values = matrix(subset)
        save_csv(args.output_dir / f"video_confusion_seed-{seed}.csv", values)
        panels.append((f"Seed {seed}: {np.trace(values)}/264 верно", values))
    pooled = sum((values for _, values in panels), np.zeros((8, 8), dtype=int))
    assert int(np.trace(pooled)) == 739
    save_csv(args.output_dir / "video_confusion_three_seeds.csv", pooled)

    fig, axes = plt.subplots(2, 2, figsize=(18, 17))
    draw_panel(axes[0, 0], pooled,
               "Три seed вместе: 739/792 верно (те же 264 видео × 3)")
    for ax, (title, values) in zip(axes.flat[1:], panels):
        draw_panel(ax, values, title)
    fig.suptitle("Матрица ошибок по целым видео — гистерезис + голосование", fontsize=17, y=0.99)
    fig.text(0.5, 0.015,
             "Строки — истинный класс, столбцы — ответ модели. Числа — ролики; цвет — доля внутри строки; "
             "оранжевая рамка — ненулевая ошибка. no_class не бывает итоговым классом видео.",
             ha="center", fontsize=10, color="#4f5d6d")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.075,
                        wspace=0.27, hspace=0.43)
    fig.savefig(args.output_dir / "video_confusion_three_seeds.png", dpi=165)
    plt.close(fig)
    print("Pooled correct", int(np.trace(pooled)), "/", int(pooled.sum()))


if __name__ == "__main__":
    main()
