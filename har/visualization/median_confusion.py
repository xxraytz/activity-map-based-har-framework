"""Plot one publication-ready video confusion matrix for the median seed.

Run in the TSU training container, where scikit-learn and matplotlib exist:

python scripts/plot_tst_video8_confusion_median_seed.py \
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
from sklearn.metrics import ConfusionMatrixDisplay


CLASSES = (
    "adl_sit", "adl_grasp", "adl_walk", "adl_lay",
    "fall_front", "fall_back", "fall_side", "fall_endupsit",
)
PREDICTION_COLUMN = "hysteresis_majority_nonbackground"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.predictions.open(newline="") as stream:
        all_rows = list(csv.DictReader(stream))

    by_seed: dict[int, list[dict]] = {}
    for row in all_rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    assert set(by_seed) == {42, 43, 44}
    for seed, rows in by_seed.items():
        assert len(rows) == len({row["source_key"] for row in rows}) == 264, seed
        assert Counter(row["true_class"] for row in rows) == Counter({cls: 33 for cls in CLASSES}), seed
        assert set(row[PREDICTION_COLUMN] for row in rows) <= set(CLASSES), seed

    # Median by whole-video accuracy; deterministic seed number breaks a tie.
    ranking = sorted(by_seed, key=lambda seed: (
        sum(row["true_class"] == row[PREDICTION_COLUMN] for row in by_seed[seed]), seed
    ))
    seed = ranking[len(ranking) // 2]
    rows = by_seed[seed]
    y_true = [row["true_class"] for row in rows]
    y_pred = [row[PREDICTION_COLUMN] for row in rows]
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    assert seed == 42 and correct == 247

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    fig, ax = plt.subplots(figsize=(11.8, 10.6), facecolor="white")
    display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        labels=CLASSES,
        display_labels=CLASSES,
        cmap=plt.cm.Blues,
        normalize=None,
        values_format="d",
        colorbar=True,
        ax=ax,
    )
    assert display.confusion_matrix.shape == (8, 8)
    assert int(display.confusion_matrix.sum()) == 264
    assert int(display.confusion_matrix.trace()) == correct

    ax.set_title(f"Матрица ошибок по видео · seed {seed}", pad=22, fontweight="semibold")
    ax.set_xlabel("Предсказанный класс", labelpad=15)
    ax.set_ylabel("Истинный класс", labelpad=15)
    ax.set_xticklabels(CLASSES, rotation=42, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(CLASSES)
    ax.tick_params(axis="both", which="both", length=0, pad=8)
    for item in display.text_.flat:
        item.set_fontsize(14)
        item.set_fontweight("semibold")
    display.im_.colorbar.ax.tick_params(labelsize=11)
    display.im_.colorbar.set_label("Количество видео", fontsize=12, labelpad=12)
    fig.text(0.50, 0.915,
             f"{correct}/264 верно · accuracy {correct / 264:.2%} · 33 видео на класс",
             ha="center", va="center", fontsize=12, color="#526273")
    fig.subplots_adjust(left=0.215, right=0.91, bottom=0.205, top=0.865)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.output_dir / f"video_confusion_median_seed-{seed}"
    fig.savefig(base.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"Median seed: {seed}; correct: {correct}/264; saved: {base}.png and .pdf")


if __name__ == "__main__":
    main()
