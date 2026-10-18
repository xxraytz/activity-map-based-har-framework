"""Compatibility entry point; implementation lives in har.data.labels."""
import importlib
import sys
_module = importlib.import_module("har.data.labels")
sys.modules[__name__] = _module
