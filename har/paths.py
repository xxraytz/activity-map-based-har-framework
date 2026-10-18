"""Resolve the experiment workspace without depending on module nesting."""
import os
from pathlib import Path


def project_root() -> Path:
    """Prefer an explicit workspace; preserve source-checkout defaults."""
    override = os.environ.get("HAR_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[1]
    if (checkout / "configs").is_dir():
        return checkout
    return Path.cwd()
