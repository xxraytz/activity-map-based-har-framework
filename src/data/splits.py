"""Compatibility entry point; implementation lives in har.data.splits."""
import importlib
import sys
_module = importlib.import_module("har.data.splits")
sys.modules[__name__] = _module
