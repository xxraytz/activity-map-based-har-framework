"""Compatibility entry point; implementation lives in har.experiments.tst.worker."""
import importlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_module = importlib.import_module("har.experiments.tst.worker")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
