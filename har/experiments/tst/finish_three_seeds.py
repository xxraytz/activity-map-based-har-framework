"""Wait for 33 TST video8 runs and evaluate all video rules."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


from har.paths import project_root

ROOT = project_root()
NAME = "tst-video8-source-aware-5ch32"
RUNS = ROOT / "runs" / NAME
REPORT = ROOT / "reports" / f"{NAME}-three-seeds"
STATUS = RUNS / "evaluation-three-seeds-status.json"


def state(status: str, **fields: object) -> None:
    STATUS.write_text(json.dumps({
        "state": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }, indent=2) + "\n")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    while True:
        done = [(actor, seed) for seed in (42, 43, 44) for actor in range(1, 12)
                if (RUNS / f"lopo-{actor:02d}-seed-{seed}" / "finished.ok").exists()]
        failed = []
        for path in RUNS.glob("worker-gpu*-seeds-43-44.json"):
            worker = json.loads(path.read_text())
            if worker.get("state") == "failed":
                failed.append(worker)
        if failed:
            state("failed_training", finished=len(done), failures=failed)
            return
        if len(done) == 33:
            break
        state("waiting_for_training", finished=len(done), remaining=33 - len(done))
        time.sleep(30)
    REPORT.mkdir(parents=True, exist_ok=True)
    for stage in ("frames", "videos"):
        state(f"evaluating_{stage}")
        command = [sys.executable, "-u", "-m", "har", "evaluate", "tst",
                   "--stage", stage, "--output", str(REPORT)]
        with (RUNS / f"evaluation-three-seeds-{stage}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            state("failed_evaluation", stage=stage, exit_code=result.returncode)
            return
    state("complete", report=str(REPORT))


if __name__ == "__main__":
    main()
