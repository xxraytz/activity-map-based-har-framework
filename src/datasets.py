import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.io import load_npz_sequence
from src.data.labels import build_label_mapping, window_distribution
from src.data.normalization import SequenceNormalizer, fit_trainset_channel_stats, resolve_normalization_config
from src.data.splits import make_lopo_splits


class SeqWindow(Dataset):
    def __init__(self, recs, win, stride, label_mapping, normalizer=None):
        self.recs = recs.reset_index(drop=True)
        self.win = int(win)
        self.stride = int(stride)
        self.label_mapping = label_mapping
        self.normalizer = normalizer or SequenceNormalizer()
        self.map = []

        for i, r in self.recs.iterrows():
            seq, labels = load_npz_sequence(r.filepath)
            num_frames = len(labels)
            assert num_frames == seq.shape[-1], \
                f'The number if elements {num_frames} and {seq.shape} are unmatched'

            for s in range(0, num_frames - self.win + 1, self.stride):
                self.map.append((i, s))

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx):
        rec_idx, start = self.map[idx]
        r = self.recs.iloc[rec_idx]
        seq, labels = load_npz_sequence(r.filepath)
        seq = self.normalizer(seq)

        window = torch.from_numpy(seq[:, :, start:start + self.win])
        target = window_distribution(labels[start:start + self.win], self.label_mapping)
        assert window.shape[-1] == self.win
        return window, target


def get_dataloaders(cfg):
    seed = cfg.get('seed')
    win = cfg['data']['window_size']
    stride = cfg['data']['window_stride']
    label_mapping = build_label_mapping(cfg['data'].get('class_names', {}))
    splits = make_lopo_splits(cfg, label_mapping)
    normalize = resolve_normalization_config(cfg['data'].get('normalize'), splits['train'])
    cfg['data']['normalize'] = normalize
    normalizer = SequenceNormalizer(normalize)

    def make_loader(split):
        dataset = SeqWindow(
            recs=splits[split],
            win=win,
            stride=stride,
            label_mapping=label_mapping,
            normalizer=normalizer,
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
            drop_last=(split == 'train'),
        )

    return make_loader('train'), make_loader('val'), make_loader('test')
