"""Compatibility entry point; implementation lives in har.training.runner."""
import importlib
import sys
_module = importlib.import_module("har.training.runner")
if __name__ == "__main__":
    _module.cli()
else:
    sys.modules[__name__] = _module
