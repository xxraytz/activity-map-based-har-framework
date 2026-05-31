# src/datasets.py

from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit


class SeqWindow(Dataset):
    def __init__(self, recs, win, stride, orig_to_merged, cls2idx, class_names):
        self.recs = recs.reset_index(drop=True)
        self.win = win
        self.stride = stride
        self.orig_to_merged = orig_to_merged
        self.cls2idx = cls2idx
        self.class_names = class_names
        self.map = []
        for i, r in self.recs.iterrows():
            labs = np.load(r.filepath, allow_pickle=True)['labels']
            M = len(labs)
            assert M == np.load(r.filepath, allow_pickle=True)['data'].shape[1], \
                f'The number if elements {M} and {np.load(r.filepath, allow_pickle=True)["data"].shape} are unmatched'
            
            for s in range(0, M - self.win + 1, self.stride):
                self.map.append((i, s))

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx):
        rec_idx, start = self.map[idx]
        r = self.recs.iloc[rec_idx]
        npz = np.load(r.filepath, allow_pickle=True)
        seq = npz['data'].astype(np.float32)
        
        seq = seq if seq.ndim > 2 else seq[None, ...]
        assert seq.ndim == 3, f'The {seq} tensor should have ndim == 3 or ndim == 2'

        labs = npz['labels']
        window = torch.tensor(seq[:, :, start:start + self.win])
        wl = labs[start:start + self.win]
        merged = [self.orig_to_merged.get(str(l), str(l)) for l in wl]
        idxs = [self.cls2idx[m] for m in merged]
        cnt = torch.bincount(torch.tensor(idxs), minlength=len(self.class_names)).float()

        assert window.shape[-1] == self.win
        return window, cnt / cnt.sum()


def get_dataloaders(cfg):
    seed = cfg.get('seed')
    df = pd.read_parquet(cfg['data']['manifest'])
    leave = cfg['data']['lopo']
    win = cfg['data']['window_size']
    stride = cfg['data']['window_stride']
    class_map = cfg['data'].get('class_names', {})
    class_names = list(dict.fromkeys(class_map.values()))
    cls2idx = {c: i for i, c in enumerate(class_names)}

    # LOPO split
    test_df = df[df.actor_id == leave]
    other = df[df.actor_id != leave].sample(frac=1, random_state=42)
    
    # --- доминирующий канонический класс каждой записи (по полному файлу)
    def majority_canonical(filepath: str) -> str:
        npz = np.load(filepath, allow_pickle=True)
        labs = npz['labels']
        merged = [class_map.get(str(l), str(l)) for l in labs]
        return Counter(merged).most_common(1)[0][0]
    
    other_major = other['filepath'].apply(majority_canonical)

    # --- стратификация other -> train/val
    val_frac = cfg['data']['val_frac']
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    train_idx, val_idx = next(sss.split(other, other_major))
    train_df = other.iloc[train_idx].reset_index(drop=True)
    val_df   = other.iloc[val_idx].reset_index(drop=True)

    splits = {
        'train': train_df,
        'val':   val_df,
        'test':  test_df,
    }

    def make_loader(split):
        dataset = SeqWindow(
            recs=splits[split],
            win=win,
            stride=stride,
            orig_to_merged=class_map,
            cls2idx=cls2idx,
            class_names=class_names
        )
        generator = None
        worker_init_fn = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))

            def worker_init_fn(worker_id):
                worker_seed = int(seed) + worker_id
                np.random.seed(worker_seed)
                torch.manual_seed(worker_seed)

        return DataLoader(
            dataset,
            batch_size=cfg['batch_size'],
            shuffle=(split == 'train'),
            num_workers=cfg['num_workers'],
            generator=generator,
            worker_init_fn=worker_init_fn,
        )

    return make_loader('train'), make_loader('val'), make_loader('test')
