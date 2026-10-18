"""Illustrate actual held-out TST videos and the final one-video decision.

Run in the TSU container, where numpy and matplotlib are installed:

python scripts/illustrate_tst_video8_decisions.py \
  --frames-dir reports/tst-video8-source-aware-5ch32-three-seeds/frames/seed-42 \
  --output-dir reports/tst-video8-source-aware-5ch32-three-seeds/illustrations
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


CLASSES = ["no_class", "adl_sit", "adl_grasp", "adl_walk", "adl_lay",
           "fall_front", "fall_back", "fall_side", "fall_endupsit"]
COLORS = ["#b8c2cc", "#9b59b6", "#159895", "#d6a800", "#e47f28",
          "#7755bf", "#2e78c4", "#cb4e8a", "#d54b50"]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(-0.5, 9.5), CMAP.N)


@dataclass
class Example:
    key: str
    kind: str
    note: str
    probs: np.ndarray
    truth_frames: np.ndarray
    argmax_frames: np.ndarray
    hyst_frames: np.ndarray
    true_class: int
    predicted_class: int
    votes: np.ndarray
    fallback: bool
    seed: int


SPECS = {
    "good": [
        ("data002_adl_grasp_2", "обычное действие: grasp"),
        ("data004_adl_walk_2", "долгая ходьба"),
        ("data003_fall_front_2", "падение вперёд"),
        ("data004_fall_back_1", "падение назад; гистерезис исправил ответ"),
        ("data003_fall_endupsit_2", "падение с посадкой"),
    ],
    "bad": [
        ("data001_adl_grasp_1", "неверный тип ADL"),
        ("data001_adl_sit_1", "ложное падение"),
        ("data001_fall_back_3", "падение пропущено"),
        ("data001_fall_front_2", "неверное направление падения"),
        ("data006_fall_side_1", "падение пропущено"),
    ],
}


def load_example(frames_dir: Path, key: str, kind: str, note: str) -> Example:
    with np.load(frames_dir / f"{key}.npz", allow_pickle=False) as arr:
        probs = arr["probs"].copy()
        y_true = arr["y_true"].astype(int)
        y_pred = arr["y_pred"].astype(int)
        hyst_pred = arr["hyst_pred"].astype(int)
        true_class = CLASSES.index(str(arr["video_class"]))
        seed = int(arr["seed"])
    votes = np.bincount(hyst_pred, minlength=9)
    fallback = not bool(votes[1:].sum())
    predicted_class = int(np.argmax(probs[:, 1:].mean(axis=0)) + 1) if fallback else int(np.argmax(votes[1:]) + 1)
    assert (predicted_class == true_class) == (kind == "good"), key
    assert set(np.unique(y_true)).issubset({0, true_class}), key
    return Example(key, kind, note, probs, y_true, y_pred, hyst_pred,
                   true_class, predicted_class, votes, fallback, seed)


def draw_row(ax_prob, ax_strip, ax_vote, ex: Example, row_number: int) -> None:
    n = len(ex.probs)
    x = np.arange(1, n + 1)
    true = ex.true_class
    predicted = ex.predicted_class
    competitor = predicted if true != predicted else int(np.argmax(np.where(np.arange(9) == true, -1, ex.votes)))
    if competitor == 0:
        competitor = int(np.argsort(ex.votes[1:])[-2] + 1)
    ax_prob.plot(x, ex.probs[:, true], color="#147d53", linewidth=2.1,
                 label=f"истинный {CLASSES[true]}")
    if competitor != true:
        ax_prob.plot(x, ex.probs[:, competitor], color="#c84d35", linewidth=1.7,
                     label=f"конкурент {CLASSES[competitor]}")
    ax_prob.plot(x, ex.probs[:, 0], color="#7d8895", linewidth=1.1,
                 alpha=0.8, label="no_class")
    ax_prob.set(xlim=(1, n), ylim=(-0.02, 1.02), xlabel="Кадр", ylabel="Балл",
                title=f"{row_number}. {ex.key} — {ex.note}")
    ax_prob.grid(alpha=0.18)
    ax_prob.legend(loc="upper left", fontsize=8, framealpha=0.9)

    timeline = np.stack([ex.truth_frames, ex.argmax_frames, ex.hyst_frames])
    ax_strip.imshow(timeline, cmap=CMAP, norm=NORM, interpolation="nearest", aspect="auto",
                    extent=(0.5, n + 0.5, 2.5, -0.5))
    ax_strip.set(xlim=(0.5, n + 0.5), ylim=(2.5, -0.5),
                 yticks=[0, 1, 2], yticklabels=["Истинная разметка", "Argmax + MA5", "Гистерезис"],
                 xlabel="Кадр", title="Как меняется метка во времени")
    ax_strip.axhline(0.5, color="white", linewidth=2)
    ax_strip.axhline(1.5, color="white", linewidth=2)

    order = sorted(range(1, 9), key=lambda cls: (-ex.votes[cls], cls))
    labels = [CLASSES[cls] for cls in order]
    vals = [int(ex.votes[cls]) for cls in order]
    bar_colors = ["#188350" if cls == predicted and predicted == true else
                  "#d44736" if cls == predicted else "#bac4cf" for cls in order]
    bars = ax_vote.barh(range(8), vals, color=bar_colors, height=0.72)
    for bar, cls in zip(bars, order):
        if cls == true:
            bar.set_edgecolor("#153b70")
            bar.set_linewidth(2)
    ax_vote.invert_yaxis()
    ax_vote.set(yticks=range(8), yticklabels=labels, xlabel="Голоса кадров",
                title=f"Итог: {CLASSES[predicted]} {'✓' if true == predicted else '✗'}")
    ax_vote.set_xlim(0, max(vals) * 1.17 + 1)
    ax_vote.grid(axis="x", alpha=0.17)
    ax_vote.set_axisbelow(True)
    for bar, count in zip(bars, vals):
        if count:
            ax_vote.text(count + 0.5, bar.get_y() + bar.get_height() / 2,
                         str(count), va="center", fontsize=8)
    ax_vote.text(0.99, 0.04, f"no_class: {ex.votes[0]} отброшено",
                 transform=ax_vote.transAxes, ha="right", va="bottom", fontsize=8,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    if ex.fallback:
        ax_vote.text(0.5, 0.5, "Все кадры = no_class\nРезерв: средняя вероятность\nпо всему видео",
                     transform=ax_vote.transAxes, ha="center", va="center", fontsize=10,
                     bbox={"facecolor": "#fff7dc", "edgecolor": "#c8a75a", "alpha": 0.95})


def draw_sheet(examples: list[Example], title: str, path: Path) -> None:
    rows = len(examples)
    fig, axes = plt.subplots(rows, 3, figsize=(17.5, 5.7 if rows == 1 else 3.35 * rows + 2.0),
                             gridspec_kw={"width_ratios": [1.45, 1.13, 0.92]})
    if rows == 1:
        axes = axes.reshape(1, 3)
    for i, ex in enumerate(examples):
        draw_row(*axes[i], ex, i + 1)
    fig.suptitle(title, fontsize=17, y=0.99)
    fig.legend(handles=[Patch(color=COLORS[i], label=CLASSES[i]) for i in range(9)],
               loc="upper center", bbox_to_anchor=(0.5, 0.92 if rows == 1 else 0.973), ncol=9,
               fontsize=8, frameon=False)
    fig.text(0.5, 0.06 if rows == 1 else 0.012,
             "Голосуют метки после гистерезиса; no_class не голосует. Если голосов нет, берём среднюю вероятность. "
             "Синяя рамка — истинный класс; истинная временная разметка показана только для проверки.",
             ha="center", fontsize=9, color="#465367")
    fig.subplots_adjust(left=0.10, right=0.965,
                        top=0.76 if rows == 1 else 0.92,
                        bottom=0.19 if rows == 1 else 0.055,
                        hspace=0.68 if rows > 1 else 0.35, wspace=0.45)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for kind, label in [("good", "Правильные ответы на 5 тестовых роликах"),
                        ("bad", "Ошибки на 5 тестовых роликах")]:
        examples = [load_example(args.frames_dir, key, kind, note)
                    for key, note in SPECS[kind]]
        draw_sheet(examples, label, args.output_dir / f"{kind}_overview.png")
        for i, ex in enumerate(examples, 1):
            draw_sheet([ex], f"Тестовый ролик: {ex.key}",
                       args.output_dir / f"{kind}_{i:02d}_{ex.key}.png")
            records.append({"group": kind, "number": i, "source_key": ex.key,
                            "note": ex.note, "seed": ex.seed, "frames": len(ex.probs),
                            "true_class": CLASSES[ex.true_class],
                            "predicted_class": CLASSES[ex.predicted_class],
                            "background_frames_ignored": int(ex.votes[0]),
                            "fallback_to_mean_probability": ex.fallback,
                            "winner_votes": int(ex.votes[ex.predicted_class]),
                            "true_class_votes": int(ex.votes[ex.true_class]),
                            "correct": ex.predicted_class == ex.true_class})
    with (args.output_dir / "examples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for row in records:
        print(row)


if __name__ == "__main__":
    main()
