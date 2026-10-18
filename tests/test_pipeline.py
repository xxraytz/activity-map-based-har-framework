"""Training and evaluation pipeline checks; no real dataset required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from har.training.runner import main, resolve_monitor
from har.cli import main as cli
from har.data.datasets import SeqWindow
from har.data.labels import build_label_mapping


class PipelineChecks(unittest.TestCase):
    def test_test_split_cannot_select_checkpoint(self):
        for metric in ('accuracy', 'loss'):
            with self.assertRaises(ValueError):
                resolve_monitor({'training': {'monitor': {'split': 'test', 'metric': metric}}})

    def test_missing_evaluation_checkpoint_fails_before_data_access(self):
        with tempfile.TemporaryDirectory() as temp, patch('har.training.runner.get_dataloaders') as loader:
            with self.assertRaises(FileNotFoundError):
                main({'mode': 'eval'}, temp, resume=True)
            loader.assert_not_called()

    def test_invalid_window_parameters(self):
        for win, stride in [(0, 1), (32, 0), (-1, 1)]:
            with self.assertRaises(ValueError):
                SeqWindow(pd.DataFrame(), win, stride, build_label_mapping({'a': 'a'}))

    def test_train_checkpoint_and_evaluate(self):
        """One CPU epoch through the real ResNet18/data/checkpoint/metric path."""
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for index, split in enumerate(('train', 'val', 'test')):
                path = root / f'{split}.npz'
                values = np.random.default_rng(index).normal(size=(5, 32, 36)).astype('float32')
                np.savez_compressed(path, data=values, labels=np.array(['a'] * 18 + ['b'] * 18))
                rows.append({'filepath': str(path), 'actor_id': index, 'seq_id': index, 'split': split})
            manifest = root / 'manifest.parquet'
            pd.DataFrame(rows).to_parquet(manifest, index=False)
            cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/tst-video8-template.yaml').read_text())
            cfg.update(device='cpu', epochs=1, batch_size=2, num_workers=0, num_classes=2)
            cfg['data'].update(manifest=str(manifest), class_names={'a': 'a', 'b': 'b'})
            run = root / 'run'
            run.mkdir()
            main(cfg, str(run), resume=False)
            self.assertTrue((run / 'best.pth').is_file())
            self.assertTrue((run / 'last.pth').is_file())
            saved = yaml.safe_load((run / 'config.yaml').read_text())
            self.assertIn('stats', saved['data']['normalize'])
            saved['mode'] = 'eval'
            main(saved, str(run), resume=True)
            # Resume training through the new command entry point.
            saved.update(mode='train', epochs=2)
            config = root / 'experiment.yaml'
            config.write_text(yaml.safe_dump(saved, sort_keys=False))
            cli(['train', '--config', str(config), '--run-dir', str(run), '--resume'])
            checkpoint = torch.load(run / 'last.pth', weights_only=False, map_location='cpu')
            self.assertEqual(checkpoint['epoch'], 2)
            cli(['evaluate', 'temporal', '--run-dir', str(run), '--device', 'cpu'])
            self.assertTrue((run / 'temporal_eval/temporal_metrics.csv').is_file())
            # One real Optuna trial with persistent storage and exported parameters.
            report = root / 'hpo'
            cli(['tune', '--config', str(config), '--storage', f'sqlite:///{root / "study.db"}',
                 '--study-name', 'smoke', '--trials', '1', '--max-epochs', '1',
                 '--report-dir', str(report), '--device', 'cpu'])
            self.assertTrue((report / 'best_params_smoke.yaml').is_file())
            trials = pd.read_csv(report / 'smoke_trials.csv')
            self.assertEqual(trials['state'].tolist(), ['COMPLETE'])



if __name__ == '__main__':
    unittest.main()
