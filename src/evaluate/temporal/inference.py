"""Compatibility entry point; implementation lives in har.evaluation.temporal.inference."""
import importlib
import sys
_module = importlib.import_module("har.evaluation.temporal.inference")
sys.modules[__name__] = _module
