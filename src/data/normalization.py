"""Compatibility entry point; implementation lives in har.data.normalization."""
import importlib
import sys
_module = importlib.import_module("har.data.normalization")
sys.modules[__name__] = _module
