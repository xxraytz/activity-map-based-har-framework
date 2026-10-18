"""Compatibility entry point; implementation lives in har.evaluation.temporal.pooling."""
import importlib
import sys
_module = importlib.import_module("har.evaluation.temporal.pooling")
sys.modules[__name__] = _module
