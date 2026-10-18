import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from har.data.io import load_npz_multilabel_sequence, load_npz_sequence
from har.data.labels import build_label_mapping, window_distribution
from har.data.normalization import SequenceNormalizer, fit_trainset_channel_stats, resolve_normalization_config
from har.data.splits import make_splits


class SeqWindow(Dataset):
    def __init__(
        self,
        recs,
        win,
        stride,
        label_mapping,
        normalizer=None,
        task="multiclass",
        multilabel_aggregation="any",
        multilabel_coverage_threshold=0.5,
    ):
        self.recs = recs.reset_index(drop=True)
        self.win = int(win)
        self.stride = int(stride)
        if self.win <= 0 or self.stride <= 0:
            raise ValueError("window_size and window_stride must be positive")
        self.label_mapping = label_mapping
        self.normalizer = normalizer or SequenceNormalizer()
        self.task = task
        self.multilabel_aggregation = multilabel_aggregation
        self.multilabel_coverage_threshold = float(multilabel_coverage_threshold)
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
        if self.task == "multilabel":
            seq, labels_multihot, action_ids = load_npz_multilabel_sequence(r.filepath)
            labels = None
        else:
            seq, labels = load_npz_sequence(r.filepath)
        seq = self.normalizer(seq)

        window = torch.from_numpy(seq[:, :, start:start + self.win])
        if self.task == "multilabel":
            frame_targets = labels_multihot[:, start:start + self.win]
            if self.multilabel_aggregation == "any":
                raw_target = frame_targets.max(axis=1)
            elif self.multilabel_aggregation == "center":
                raw_target = frame_targets[:, self.win // 2]
            elif self.multilabel_aggregation == "coverage_threshold":
                raw_target = (
                    frame_targets.mean(axis=1) >= self.multilabel_coverage_threshold
                ).astype(np.float32)
            else:
                raise ValueError(
                    f"Unknown multilabel aggregation: {self.multilabel_aggregation}"
                )

            target = torch.zeros(len(self.label_mapping.class_names), dtype=torch.float32)
            for raw_idx, action_id in enumerate(action_ids):
                canonical = self.label_mapping.raw_to_canonical.get(str(action_id), str(action_id))
                if canonical in self.label_mapping.cls2idx:
                    target[self.label_mapping.cls2idx[canonical]] = float(raw_target[raw_idx])
        else:
            target = window_distribution(labels[start:start + self.win], self.label_mapping)
        assert window.shape[-1] == self.win
        return window, target


def get_dataloaders(cfg):
    seed = cfg.get('seed')
    win = cfg['data']['window_size']
    stride = cfg['data']['window_stride']
    label_mapping = build_label_mapping(cfg['data'].get('class_names', {}))
    task = cfg.get('task', 'multiclass')
    multilabel_cfg = cfg['data'].get('multilabel', {}) or {}
    splits = make_splits(cfg, label_mapping)
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
            task=task,
            multilabel_aggregation=multilabel_cfg.get('aggregation', 'any'),
            multilabel_coverage_threshold=multilabel_cfg.get('coverage_threshold', 0.5),
        )
        if not len(dataset):
            raise ValueError(f"No complete windows in {split}; check sequence lengths and window_size")
        if split == "train" and len(dataset) < cfg["batch_size"]:
            raise ValueError("Training windows fewer than batch_size with drop_last=True; reduce batch_size")
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
