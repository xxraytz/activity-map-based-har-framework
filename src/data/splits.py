from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .io import load_npz_sequence
from .labels import LabelMapping, majority_canonical


def read_manifest(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(cfg["data"]["manifest"])


def make_lopo_splits(cfg: dict, mapping: LabelMapping) -> dict[str, pd.DataFrame]:
    df = read_manifest(cfg)
    leave = int(cfg["data"]["lopo"])

    test_df = df[df.actor_id == leave].reset_index(drop=True)
    other = df[df.actor_id != leave].sample(frac=1, random_state=42).reset_index(drop=True)
    other_major = other["filepath"].apply(lambda filepath: majority_canonical(load_npz_sequence(filepath)[1], mapping))

    val_frac = float(cfg["data"]["val_frac"])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    train_idx, val_idx = next(sss.split(other, other_major))

    return {
        "train": other.iloc[train_idx].reset_index(drop=True),
        "val": other.iloc[val_idx].reset_index(drop=True),
        "test": test_df,
    }


def split_manifest(cfg: dict, split: str, mapping: LabelMapping) -> pd.DataFrame:
    if split == "all":
        return read_manifest(cfg).reset_index(drop=True)

    splits = make_lopo_splits(cfg, mapping)
    try:
        return splits[split]
    except KeyError as exc:
        raise ValueError(f"Unknown split: {split}") from exc

