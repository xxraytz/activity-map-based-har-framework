"""Compatibility entry point; implementation lives in har.evaluation.temporal.smoothing."""
import importlib
import sys
_module = importlib.import_module("har.evaluation.temporal.smoothing")
sys.modules[__name__] = _module
