"""Unified command dispatch with lazy imports and preserved command arguments."""
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    module: str
    description: str
    entry: str = "main"


COMMANDS = {
    "train": Command("har.training.runner", "Train or resume a configured model", "cli"),
    "tune": Command("har.tuning.optuna", "Run Optuna hyperparameter/architecture search"),
    "prepare": {
        "manifest": Command("har.data.manifest", "Build an NPZ sequence manifest"),
        "tst": Command("har.data.preparation.tst", "Build TST source-aware labels and LOPO configs"),
        "tst-skeleton": Command("har.data.preparation.tst_skeleton", "Build TST skeleton features"),
        "pku": Command("har.data.preparation.pku", "Build PKU graph data"),
    },
    "evaluate": {
        "temporal": Command("har.evaluation.cli", "Evaluate windows, frames and annotated segments"),
        "tst": Command("har.evaluation.tst", "Evaluate three-seed TST frames/videos"),
        "tst-single": Command("har.evaluation.tst_single_seed", "Evaluate the original single-seed TST protocol"),
    },
    "plot": {
        "confusion": Command("har.visualization.confusion", "Plot pooled video confusion matrices"),
        "median-confusion": Command("har.visualization.median_confusion", "Plot the median-seed confusion matrix"),
        "aggregation": Command("har.visualization.aggregation", "Illustrate overlapping predictions"),
        "decisions": Command("har.visualization.decisions", "Illustrate video decisions"),
        "skeleton": Command("har.visualization.skeleton", "Illustrate skeleton examples"),
    },
    "experiment": {
        "tst-worker": Command("har.experiments.tst.worker", "Run the existing TST GPU fold worker"),
        "tst-finish": Command("har.experiments.tst.finish", "Wait for and evaluate original TST runs"),
        "tst-finish-three": Command("har.experiments.tst.finish_three_seeds", "Wait for and evaluate 33 TST runs"),
    },
}


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    commands = COMMANDS
    path = []
    while isinstance(commands, dict):
        parser = argparse.ArgumentParser(prog="har" + (" " + " ".join(path) if path else ""),
                                         description="Human action recognition framework")
        choices = parser.add_subparsers(dest="command", required=True)
        for name, command in commands.items():
            choices.add_parser(name, help=command.description if isinstance(command, Command) else f"{name.title()} commands", add_help=False)
        if not args or args[0] in {"-h", "--help"}:
            parser.print_help()
            if not args:
                raise SystemExit(2)
            return
        name = args.pop(0)
        if name not in commands:
            parser.error(f"unknown command: {name}")
        path.append(name)
        commands = commands[name]
    original_argv = sys.argv
    try:
        sys.argv = ["har " + " ".join(path), *args]
        module = importlib.import_module(commands.module)
        getattr(module, commands.entry)()
    finally:
        sys.argv = original_argv
