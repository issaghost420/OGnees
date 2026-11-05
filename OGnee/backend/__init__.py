#!/usr/bin/env python3
"""
OGnee Backend

OGneey trust progression:
  OGreenie 🌱 → OSeemie 👀 → OGneey 🧞 → OFreeney 🧞‍♂️✨

Submodules are loaded lazily (PEP 562) so a bare ``import backend`` does
not pull in requests/prompt_toolkit or open network sessions, and the
package namespace stays clean (no wildcard-import clobbering — the old
``from .logging import *`` both shadowed stdlib logging and silently
replaced config.get_shadow_path with shadow.py's divergent copy).
"""

import importlib

__version__ = "1.1.0"
__author__ = "EdgeCase"

_SUBMODULES = (
    "config",
    "context",
    "display",
    "session_log",
    "shadow",
    "gates",
    "commands",
    "tools",
    "output_buffer",
    "mcp",
    "ssh",
    "startup",
    "agent",
)

__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    """Lazily import submodules on first attribute access."""
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
