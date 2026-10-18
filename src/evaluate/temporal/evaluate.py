"""Compatibility entry point; implementation lives in har.evaluation.temporal.evaluate."""
import importlib
import sys
_module = importlib.import_module("har.evaluation.temporal.evaluate")
sys.modules[__name__] = _module
