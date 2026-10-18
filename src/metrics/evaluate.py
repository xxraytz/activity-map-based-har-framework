"""Compatibility entry point; implementation lives in har.evaluation.metrics.evaluate."""
import importlib
import sys
_module = importlib.import_module("har.evaluation.metrics.evaluate")
sys.modules[__name__] = _module
