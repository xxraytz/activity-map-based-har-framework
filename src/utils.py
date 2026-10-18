"""Compatibility entry point; implementation lives in har.utils."""
import importlib
import sys
_module = importlib.import_module("har.utils")
sys.modules[__name__] = _module
