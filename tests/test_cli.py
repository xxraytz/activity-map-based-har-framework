"""Public command routing and legacy API compatibility."""
import contextlib
import importlib
import io
import sys
import unittest
from unittest.mock import patch

from har.cli import COMMANDS, Command, main


def leaves(tree, prefix=()):
    for name, value in tree.items():
        path = (*prefix, name)
        if isinstance(value, Command):
            yield path, value
        else:
            yield from leaves(value, path)


class CommandChecks(unittest.TestCase):
    def test_all_commands_have_safe_help(self):
        for path, command in leaves(COMMANDS):
            with self.subTest(command=path), contextlib.redirect_stdout(io.StringIO()):
                # Even orchestration commands must not start work for --help.
                with self.assertRaises(SystemExit) as result:
                    main([*path, '--help'])
                self.assertEqual(result.exception.code, 0)

    def test_dispatch_preserves_arguments_and_restores_argv_on_failure(self):
        original = sys.argv
        observed = []
        def fail():
            observed.extend(sys.argv[1:])
            raise RuntimeError('test')
        with patch('har.training.runner.cli', side_effect=fail):
            with self.assertRaises(RuntimeError):
                main(['train', '--config', 'a path/config.yaml', '--resume'])
        self.assertEqual(observed, ['--config', 'a path/config.yaml', '--resume'])
        self.assertIs(sys.argv, original)

    def test_old_imports_resolve_to_the_same_implementation(self):
        for old, new in [
            ('main', 'har.training.runner'),
            ('src.models', 'har.models.factory'),
            ('src.datasets', 'har.data.datasets'),
            ('src.train', 'har.training.engine'),
            ('src.data.labels', 'har.data.labels'),
            ('src.evaluate.temporal.types', 'har.evaluation.temporal.types'),
            ('scripts.optuna_run', 'har.tuning.optuna'),
            ('scripts.prepare_data', 'har.data.preparation.tst'),
            ('scripts.evaluate', 'har.evaluation.tst'),
        ]:
            with self.subTest(module=old):
                self.assertIs(importlib.import_module(old), importlib.import_module(new))

    def test_invalid_command_fails(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            main(['not-a-command'])
        self.assertEqual(result.exception.code, 2)
