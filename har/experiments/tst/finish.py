"""Wait for 11 LOPO runs, then score frames before videos."""

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
REPORT = ROOT / "reports" / NAME


def state(status: str, **fields: object) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "evaluation-status.json").write_text(json.dumps({
        "state": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }, indent=2) + "\n")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    while True:
        complete = [actor for actor in range(1, 12)
                    if (RUNS / f"lopo-{actor:02d}-seed-42" / "finished.ok").exists()]
        failed = []
        for path in RUNS.glob("worker-gpu*.json"):
            status = json.loads(path.read_text())
            if status.get("state") == "failed":
                failed.append(status)
        if failed:
            state("failed_training", finished=complete, failures=failed)
            return
        if len(complete) == 11:
            break
        state("waiting_for_training", finished=complete, remaining=11 - len(complete))
        time.sleep(30)
    for stage in ("frames", "videos"):
        state(f"evaluating_{stage}")
        REPORT.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", "-m", "har", "evaluate", "tst-single",
                   "--stage", stage, "--output", str(REPORT)]
        with (RUNS / f"evaluation-{stage}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            state("failed_evaluation", stage=stage, exit_code=result.returncode)
            return
    state("complete", frame_metrics=str(REPORT / "frame_metrics.json"),
          video_metrics=str(REPORT / "video_metrics.json"))


if __name__ == "__main__":
    main()
