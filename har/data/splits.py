from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

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


def make_manifest_splits(cfg: dict, mapping: LabelMapping) -> dict[str, pd.DataFrame]:
    df = read_manifest(cfg)
    split_column = cfg["data"].get("split_column", "split")
    if split_column not in df.columns:
        raise KeyError(f"Manifest split column {split_column!r} is missing")

    split_values = df[split_column].astype(str).str.lower()
    train_df = df[split_values == "train"].reset_index(drop=True)
    val_df = df[split_values == "val"].reset_index(drop=True)
    test_df = df[split_values == "test"].reset_index(drop=True)

    if train_df.empty:
        raise ValueError(f"Manifest split column {split_column!r} has no train rows")
    if test_df.empty:
        raise ValueError(f"Manifest split column {split_column!r} has no test rows")

    if val_df.empty:
        train_major = train_df["filepath"].apply(
            lambda filepath: majority_canonical(load_npz_sequence(filepath)[1], mapping)
        )
        val_frac = float(cfg["data"].get("val_frac", 0.1))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
        try:
            train_idx, val_idx = next(sss.split(train_df, train_major))
        except ValueError:
            indices = list(range(len(train_df)))
            train_idx, val_idx = train_test_split(indices, test_size=val_frac, random_state=42)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        train_df = train_df.iloc[train_idx].reset_index(drop=True)

    return {"train": train_df, "val": val_df, "test": test_df}


def make_splits(cfg: dict, mapping: LabelMapping) -> dict[str, pd.DataFrame]:
    split_mode = cfg["data"].get("split_mode", "lopo")
    if split_mode == "lopo":
        return make_lopo_splits(cfg, mapping)
    if split_mode == "manifest":
        return make_manifest_splits(cfg, mapping)
    raise ValueError(f"Unknown split_mode: {split_mode}")


def split_manifest(cfg: dict, split: str, mapping: LabelMapping) -> pd.DataFrame:
    if split == "all":
        return read_manifest(cfg).reset_index(drop=True)

    splits = make_splits(cfg, mapping)
    try:
        return splits[split]
    except KeyError as exc:
        raise ValueError(f"Unknown split: {split}") from exc
