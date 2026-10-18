"""Run assigned TST video8 LOPO folds sequentially on one GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


from har.paths import project_root

ROOT = project_root()
NAME = "tst-video8-source-aware-5ch32"


def write_status(path: Path, **status: object) -> None:
    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(status, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    experiment_dir = ROOT / "runs" / NAME
    experiment_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = "-".join(map(str, args.seeds))
    status_path = experiment_dir / f"worker-gpu{args.gpu}-seeds-{seed_tag}.json"
    folds = list(range(args.gpu, 12, 3))
    for actor in folds:
        for seed in args.seeds:
            run_dir = experiment_dir / f"lopo-{actor:02d}-seed-{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config = ROOT / "configs" / NAME / f"lopo-{actor:02d}" / f"seed-{seed}.yaml"
            checkpoint = run_dir / "best.pth"
            if checkpoint.exists() and (run_dir / "finished.ok").exists():
                print(f"Fold {actor}, seed {seed}: already finished", flush=True)
                continue
            cmd = [
                sys.executable, "-u", "-m", "har", "train",
                "--config", str(config),
                "--run-dir", str(run_dir),
                "--device", f"cuda:{args.gpu}",
            ]
            if (run_dir / "last.pth").exists():
                cmd.append("--resume")
            write_status(status_path, state="running", gpu=args.gpu, actor=actor,
                         seed=seed, folds=folds, seeds=args.seeds, command=cmd)
            print("Running", " ".join(cmd), flush=True)
            with (run_dir / "console.log").open("a") as log:
                log.write(f"\nSTART {datetime.now(timezone.utc).isoformat()} {' '.join(cmd)}\n")
                log.flush()
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                write_status(status_path, state="failed", gpu=args.gpu, actor=actor,
                             seed=seed, exit_code=result.returncode, folds=folds, seeds=args.seeds)
                print(f"Fold {actor}, seed {seed} failed with exit code {result.returncode}", flush=True)
                raise SystemExit(result.returncode)
            (run_dir / "finished.ok").write_text(datetime.now(timezone.utc).isoformat() + "\n")
            write_status(status_path, state="fold_finished", gpu=args.gpu, actor=actor,
                         seed=seed, folds=folds, seeds=args.seeds)
            print(f"Fold {actor}, seed {seed} complete", flush=True)
    write_status(status_path, state="complete", gpu=args.gpu, folds=folds, seeds=args.seeds)


if __name__ == "__main__":
    main()
