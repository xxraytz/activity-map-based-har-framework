"""Compatibility entry point; implementation lives in har.visualization.decisions."""
import importlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_module = importlib.import_module("har.visualization.decisions")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
