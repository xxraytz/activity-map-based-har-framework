"""Compatibility entry point; implementation lives in har.data.io."""
import importlib
import sys
_module = importlib.import_module("har.data.io")
sys.modules[__name__] = _module
