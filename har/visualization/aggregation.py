"""Draw the window-to-frame step and a real whole-video counterexample.

Run in the TSU container with numpy and matplotlib:
  python scripts/illustrate_tst_video8_aggregation.py \
    --predictions /home/2025/human-action-recognition/reports/tst-video8-source-aware-5ch32-three-seeds/frames/seed-42/data004_fall_back_1.npz \
    --output-dir /home/2025/human-action-recognition/reports/tst-video8-source-aware-5ch32-three-seeds
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


CLASSES = ["no_class", "adl_sit", "adl_grasp", "adl_walk", "adl_lay",
           "fall_front", "fall_back", "fall_side", "fall_endupsit"]
COLORS = {"no_class": "#7b8794", "adl_lay": "#ed8936", "fall_back": "#3178c6"}


def smooth_ma5(scores: np.ndarray) -> np.ndarray:
    padded = np.pad(scores, ((2, 2), (0, 0)), mode="edge")
    out = np.stack([padded[t:t + 5].mean(axis=0) for t in range(len(scores))])
    return out / out.sum(axis=1, keepdims=True)


def draw_window_demo(out: Path) -> None:
    # Three illustrative 4-frame windows; the experiment uses length 32.
    windows = [(0, 4, np.array([0.8, 0.2])),
               (2, 6, np.array([0.3, 0.7])),
               (3, 7, np.array([0.2, 0.8]))]
    scores = np.zeros((7, 2))
    for frame in range(7):
        covering = [p for start, end, p in windows if start <= frame < end]
        scores[frame] = np.max(covering, axis=0)
    frame_probs = scores / scores.sum(axis=1, keepdims=True)
    smoothed = smooth_ma5(frame_probs)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), height_ratios=[1, 1.5], sharex=True)
    for idx, (start, end, p) in enumerate(windows):
        y = 2 - idx
        axes[0].barh(y, end - start, left=start + 0.5, height=0.55,
                     color=["#2b6cb0", "#62a3d6", "#99c7e7"][idx], edgecolor="#1a365d")
        axes[0].text(start + 0.65, y, f"W{idx + 1}: P(A)={p[0]:.1f}, P(B)={p[1]:.1f}",
                     va="center", fontsize=10, color="white" if idx == 0 else "#17324d")
    axes[0].set(yticks=[0, 1, 2], yticklabels=["W3", "W2", "W1"],
                xlim=(0.5, 7.5), ylim=(-0.65, 2.7), title="1. Перекрывающиеся окна дают ответы для одних и тех же кадров")
    axes[0].axvline(4, color="#a61b29", linestyle=":", linewidth=1.8)
    axes[0].grid(axis="x", alpha=0.2)

    x = np.arange(1, 8)
    axes[1].plot(x, frame_probs[:, 0], "o-", color="#2b6cb0", label="A после max + нормирования", linewidth=2)
    axes[1].plot(x, frame_probs[:, 1], "o-", color="#dd6b20", label="B после max + нормирования", linewidth=2)
    axes[1].plot(x, smoothed[:, 0], "--", color="#2b6cb0", label="A после среднего по 5 кадрам", linewidth=2)
    axes[1].plot(x, smoothed[:, 1], "--", color="#dd6b20", label="B после среднего по 5 кадрам", linewidth=2)
    axes[1].scatter([4], [frame_probs[3, 0]], s=160, facecolors="none", edgecolors="#a61b29", linewidths=2)
    axes[1].annotate("Кадр 4: max A=0.8 и max B=0.8\nиз разных окон → после нормирования 0.5 / 0.5",
                     xy=(4, 0.5), xytext=(4.5, 0.88), fontsize=9,
                     arrowprops={"arrowstyle": "->", "color": "#a61b29"})
    axes[1].set(xlabel="Номер кадра", ylabel="Вероятность / нормированный балл",
                xticks=x, ylim=(-0.03, 1.08), title="2. Покадровый max по классам, затем скользящее среднее")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="lower left", fontsize=9, ncol=2)
    fig.text(0.5, 0.01, "Игрушечный пример: длина окна 4 кадра для наглядности; в эксперименте — 32 кадра, шаг 1.",
             ha="center", fontsize=9, color="#4a5568")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out / "01_window_overlap.png", dpi=180)
    plt.close(fig)


def draw_real_video(predictions: Path, out: Path) -> None:
    with np.load(predictions, allow_pickle=False) as arr:
        probs = arr["probs"]
        y_true = arr["y_true"].astype(int)
        y_pred = arr["y_pred"].astype(int)
        hyst_pred = arr["hyst_pred"].astype(int)
        source = str(arr["source_key"])
        seed = int(arr["seed"])
    n = len(probs)
    x = np.arange(1, n + 1)
    onset = int(np.flatnonzero(y_true == CLASSES.index("fall_back"))[0]) + 1
    k = math.ceil(n * 0.25)
    mean = probs.mean(axis=0)
    top = np.sort(probs, axis=0)[-k:].mean(axis=0)
    raw_counts = np.bincount(y_pred, minlength=9)
    hyst_counts = np.bincount(hyst_pred, minlength=9)

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1], height_ratios=[2.2, 1], hspace=0.32, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    for name in ("no_class", "adl_lay", "fall_back"):
        ax.plot(x, probs[:, CLASSES.index(name)], color=COLORS[name], label=name, linewidth=1.8)
    ax.axvspan(onset - 0.5, n + 0.5, color="#3178c6", alpha=0.07)
    ax.axvline(onset - 0.5, color="#3178c6", linestyle=":", linewidth=1.5)
    ax.text(104, 0.97, "Синяя область: истинный fall_back\n(только для проверки)", va="top", fontsize=9, color="#25598f")
    ax.set(xlim=(1, n), ylim=(0, 1.02), xlabel="Кадр", ylabel="Покадровая вероятность",
           title=f"Реальный ролик {source}, seed {seed}: две конкурирующие гипотезы")
    ax.grid(alpha=0.17)
    ax.legend(loc="upper left", ncol=3)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    for row, (pred, label) in enumerate([(y_pred, "Argmax после MA5"), (hyst_pred, "После гистерезиса")]):
        for name in ("no_class", "adl_lay", "fall_back"):
            mask = pred == CLASSES.index(name)
            ax2.bar(x[mask], np.ones(mask.sum()) * 0.7, bottom=row, width=1.0,
                    color=COLORS[name], align="center", linewidth=0)
        other = ~np.isin(pred, [CLASSES.index(a) for a in COLORS])
        ax2.bar(x[other], np.ones(other.sum()) * 0.7, bottom=row, width=1.0,
                color="#b8c2cc", align="center", linewidth=0)
    ax2.set(yticks=[0.35, 1.35], yticklabels=["Argmax + MA5", "Гистерезис"],
            ylim=(0, 1.8), xlabel="Кадр", title="Жёсткие покадровые ответы")
    ax2.axvline(onset - 0.5, color="#3178c6", linestyle=":", linewidth=1.2)
    ax2.legend(handles=[Patch(color=COLORS[name], label=name) for name in COLORS],
               loc="upper right", ncol=3, fontsize=9)

    ax3 = fig.add_subplot(gs[0, 1])
    labels = ["Среднее по всем кадрам", "Среднее верхних 25%"]
    positions = np.arange(2)
    w = 0.34
    ax3.bar(positions - w / 2, [mean[4], top[4]], width=w, color=COLORS["adl_lay"], label="adl_lay")
    ax3.bar(positions + w / 2, [mean[6], top[6]], width=w, color=COLORS["fall_back"], label="fall_back")
    ax3.set(xticks=positions, xticklabels=labels, ylim=(0, 0.9), ylabel="Баллы по видео",
            title="Мягкие правила выбирают adl_lay")
    ax3.tick_params(axis="x", labelrotation=15)
    ax3.grid(axis="y", alpha=0.18)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    labels = ["Обычное\nголосование", "После\nгистерезиса"]
    ax4.bar(positions - w / 2, [raw_counts[4], hyst_counts[4]], width=w, color=COLORS["adl_lay"])
    ax4.bar(positions + w / 2, [raw_counts[6], hyst_counts[6]], width=w, color=COLORS["fall_back"])
    ax4.set(xticks=positions, xticklabels=labels, ylabel="Число кадров с меткой",
            title="Гистерезис выбирает fall_back")
    ax4.grid(axis="y", alpha=0.18)
    fig.suptitle("Четыре правила для одного видео: разные ответы из одних и тех же вероятностей", fontsize=15)
    fig.text(0.5, 0.015,
             "Имя файла и синяя область разметки показаны только для объяснения ошибки; алгоритмы их не получают.",
             ha="center", fontsize=9, color="#4a5568")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / "02_real_video_methods.png", dpi=180)
    plt.close(fig)
    print("source", source, "seed", seed, "frames", n, "onset", onset,
          "mean", mean[[4, 6]], "top25", top[[4, 6]],
          "hard_counts", raw_counts[[4, 6]], "hyst_counts", hyst_counts[[4, 6]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_window_demo(args.output_dir)
    draw_real_video(args.predictions, args.output_dir)


if __name__ == "__main__":
    main()
