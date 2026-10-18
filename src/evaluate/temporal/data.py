"""Compatibility entry point; implementation lives in har.evaluation.temporal.data."""
import importlib
import sys
_module = importlib.import_module("har.evaluation.temporal.data")
sys.modules[__name__] = _module
