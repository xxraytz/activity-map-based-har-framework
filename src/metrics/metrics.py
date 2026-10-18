"""Compatibility entry point; implementation lives in har.evaluation.metrics.metrics."""
import importlib
import sys
_module = importlib.import_module("har.evaluation.metrics.metrics")
sys.modules[__name__] = _module
