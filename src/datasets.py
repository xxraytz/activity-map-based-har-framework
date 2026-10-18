"""Compatibility entry point; implementation lives in har.data.datasets."""
import importlib
import sys
_module = importlib.import_module("har.data.datasets")
sys.modules[__name__] = _module
