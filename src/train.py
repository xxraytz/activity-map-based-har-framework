"""Compatibility entry point; implementation lives in har.training.engine."""
import importlib
import sys
_module = importlib.import_module("har.training.engine")
sys.modules[__name__] = _module
