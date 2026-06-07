from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.models import build_model

from .data import build_class_mapping, load_sequence, records_from_manifest, split_manifest
from .types import SequenceRecord, WindowPredictions


def load_model_from_run(run_dir: str | Path, cfg: dict, device: torch.device, checkpoint_name: str = "best.pth") -> torch.nn.Module:
    checkpoint_path = Path(run_dir) / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_sequence(
    model: torch.nn.Module,
    record: SequenceRecord,
    cfg: dict,
    device: torch.device,
    batch_size: int,
) -> WindowPredictions:
    mapping = build_class_mapping(cfg)
    seq, y_true = load_sequence(record, cfg, mapping)
    window_size = int(cfg["data"]["window_size"])
    stride = int(cfg["data"]["window_stride"])
    starts = np.arange(0, seq.shape[-1] - window_size + 1, stride, dtype=np.int64)

    if starts.size == 0:
        logits = np.zeros((0, len(mapping.class_names)), dtype=np.float32)
        probs = logits.copy()
        return WindowPredictions(record, y_true, starts, logits, probs, window_size)

    logits_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset:offset + batch_size]
            windows = np.stack([seq[:, :, start:start + window_size] for start in batch_starts])
            x = torch.from_numpy(windows).to(device=device, dtype=torch.float32)
            logits = model(x).detach().cpu().numpy()
            logits_chunks.append(logits)

    logits = np.concatenate(logits_chunks, axis=0).astype(np.float32)
    return WindowPredictions(
        record=record,
        y_true=y_true,
        starts=starts,
        logits=logits,
        probs=_softmax(logits).astype(np.float32),
        window_size=window_size,
    )


def iter_window_predictions(
    model: torch.nn.Module,
    cfg: dict,
    split: str,
    device: torch.device,
    batch_size: int,
) -> list[WindowPredictions]:
    df = split_manifest(cfg, split)
    records = records_from_manifest(df)
    return [
        predict_sequence(model, record, cfg, device, batch_size)
        for record in tqdm(records, desc=f"Temporal inference [{split}]", dynamic_ncols=True)
    ]

