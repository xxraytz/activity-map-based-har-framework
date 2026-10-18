"""Compatibility entry point; implementation lives in har.evaluation.temporal.metrics."""
import importlib
import sys
_module = importlib.import_module("har.evaluation.temporal.metrics")
sys.modules[__name__] = _module
