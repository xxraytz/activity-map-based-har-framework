"""Compatibility entry point; implementation lives in har.evaluation.temporal.types."""
import importlib
import sys
_module = importlib.import_module("har.evaluation.temporal.types")
sys.modules[__name__] = _module
