"""Compatibility entry point; implementation lives in har.models.factory."""
import importlib
import sys
_module = importlib.import_module("har.models.factory")
sys.modules[__name__] = _module
