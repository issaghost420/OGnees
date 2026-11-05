#!/usr/bin/env python3
"""
shadow.py - Shadow (staging) file operations

Writes to protected paths are staged under SHADOW_ROOT for user review.
The real<->shadow path mapping lives in config.py (get_shadow_path /
get_real_path) — there is exactly ONE mapping, and it is idempotent, so
staging a file that is already in the shadow tree never doubles the path.
"""

import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .config import SHADOW_ROOT, get_shadow_path, get_real_path


def shadow_write(real_path: str, content: str) -> Tuple[bool, str]:
    """Write *content* to the shadow staging path for *real_path*."""
    try:
        shadow_path = get_shadow_path(real_path)
        shadow_path.parent.mkdir(parents=True, exist_ok=True)
        shadow_path.write_text(content)
        return True, str(shadow_path)
    except Exception as e:
        return False, str(e)


def shadow_read(real_path: str) -> Optional[str]:
    """Read the staged shadow copy of *real_path*, if one exists."""
    shadow_path = get_shadow_path(real_path)
    if shadow_path.exists():
        return shadow_path.read_text()
    return None


def has_shadow_version(real_path: str) -> bool:
    """True if *real_path* has a staged shadow copy."""
    return get_shadow_path(real_path).exists()


def list_shadow_changes() -> List[Dict[str, Any]]:
    """List all staged shadow files with their real (target) paths."""
    if not SHADOW_ROOT.exists():
        return []
    changes = []
    for shadow_file in SHADOW_ROOT.rglob("*"):
        if shadow_file.is_file():
            real_path = get_real_path(shadow_file)
            changes.append({
                "shadow_path": str(shadow_file),
                "real_path": str(real_path),
                "size": shadow_file.stat().st_size,
                "modified": datetime.fromtimestamp(shadow_file.stat().st_mtime),
                "is_new": not real_path.exists(),
            })
    return sorted(changes, key=lambda x: x["modified"], reverse=True)


def apply_shadow_change(real_path: str, backup: bool = True) -> Tuple[bool, str]:
    """Copy the staged shadow copy of *real_path* onto the real file.

    The backup of a pre-existing target is ``<name>.bak`` — built with
    ``with_name(name + ".bak")`` because ``Path.with_suffix`` raises
    ValueError on dotfiles like ``.bashrc``.
    """
    shadow_path = get_shadow_path(real_path)
    if not shadow_path.exists():
        return False, "No shadow file"
    real = get_real_path(shadow_path)
    try:
        if backup and real.exists():
            shutil.copy2(real, real.with_name(real.name + ".bak"))
        real.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shadow_path, real)
        shadow_path.unlink()
        return True, f"Applied: {real}"
    except Exception as e:
        return False, str(e)


def apply_shadow_changes(paths: List[str] = None) -> List[Tuple[str, bool, str]]:
    """Apply staged changes for *paths* (all staged changes when None).

    Returns a list of ``(real_path, success, message)`` triples.
    """
    if paths is None:
        targets = [c["real_path"] for c in list_shadow_changes()]
    else:
        targets = [str(get_real_path(get_shadow_path(p))) for p in paths]
    results = []
    for target in targets:
        ok, msg = apply_shadow_change(target)
        results.append((target, ok, msg))
    return results


def clear_shadow(paths: List[str] = None) -> int:
    """Delete staged shadow files.

    With no argument, clears the whole shadow tree. With *paths*, removes
    only the shadow copies of those paths (honored, not ignored).
    """
    count = 0
    if paths is None:
        if not SHADOW_ROOT.exists():
            return 0
        for f in SHADOW_ROOT.rglob("*"):
            if f.is_file():
                f.unlink()
                count += 1
        return count
    for p in paths:
        shadow_path = get_shadow_path(p)
        # Never delete anything that resolved outside the shadow tree.
        if shadow_path.is_file() and shadow_path.is_relative_to(SHADOW_ROOT):
            shadow_path.unlink()
            count += 1
    return count
