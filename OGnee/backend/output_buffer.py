#!/usr/bin/env python3
"""
output_buffer.py — Session-scoped storage for large tool outputs

When a tool returns more than MAX_TOOL_RESULT_CHARS, the full output is
stashed here and the LLM receives a smart preview + a reference handle.
The model can then call `read_output(ref=..., offset=...)` to paginate
through the result in structured chunks.

Design goals:
  - LLM never gets invalid JSON / truncated diffs / chopped listings
  - One round-trip per chunk, no polling magic
  - Bounded memory: store at most 100 stashes, drop oldest
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# Maximum stashed outputs per session (memory bound)
_MAX_STASHED = 100
# Maximum chars before stashing (imported by tools.py)
MAX_TOOL_RESULT_CHARS = 20000

@dataclass
class StashedOutput:
    """A single stashed tool result."""
    ref: str
    server: str
    tool: str
    iteration: int
    full_text: str
    content_type: str          # "json", "listing", "diff", "text", "csv", "code"
    total_lines: int
    total_chars: int


class ToolOutputBuffer:
    """Session-scoped store for large tool outputs."""

    def __init__(self, max_stashed: int = _MAX_STASHED):
        self._outputs: Dict[str, StashedOutput] = {}
        self._counter = 0
        self._max_stashed = max_stashed
        # Track insertion order for LRU eviction
        self._refs_lru: List[str] = []

    # ------------------------------------------------------------------
    # Core: store / retrieve / drop
    # ------------------------------------------------------------------

    def store(self, server: str, tool: str, iteration: int,
              full_output: str) -> Tuple[str, str]:
        """Store full output, return (reference_handle, preview_text).

        The preview is a structure-aware truncation (see _build_preview).
        """
        ref = f"out_{self._counter:04d}"
        self._counter += 1

        # Evict oldest if at limit
        if len(self._refs_lru) >= self._max_stashed:
            oldest = self._refs_lru.pop(0)
            self._outputs.pop(oldest, None)

        content_type = _detect_content_type(full_output)
        so = StashedOutput(
            ref=ref,
            server=server,
            tool=tool,
            iteration=iteration,
            full_text=full_output,
            content_type=content_type,
            total_lines=full_output.count('\n'),
            total_chars=len(full_output),
        )
        self._outputs[ref] = so
        self._refs_lru.append(ref)

        preview = _build_preview(full_output, content_type)
        return ref, preview

    def read(self, ref: str, offset: int = 0,
             max_chars: int = 4000) -> Tuple[str, bool, int]:
        """Read a chunk. Returns (chunk, has_more, next_offset).

        `offset` is character offset into the full text.
        `max_chars` is the soft limit for this chunk.
        """
        so = self._outputs.get(ref)
        if not so:
            return f"[ERROR: Unknown output reference '{ref}']", False, 0

        full = so.full_text
        total = len(full)

        if offset >= total:
            return "[END: No more data]", False, offset

        # Try to snap to a line boundary for text / diff / listing
        chunk_end = min(offset + max_chars, total)
        if so.content_type in ("text", "diff", "listing", "csv", "code"):
            # Walk back from chunk_end to find a newline
            newline_pos = full.rfind('\n', offset, chunk_end)
            if newline_pos != -1:
                chunk_end = newline_pos + 1  # include the newline

        chunk = full[offset:chunk_end]
        next_off = chunk_end
        has_more = next_off < total
        return chunk, has_more, next_off

    def search(self, ref: str, query: str, context_lines: int = 2) -> List[Tuple[int, str]]:
        """Find lines matching query with line numbers and surrounding context."""
        so = self._outputs.get(ref)
        if not so:
            return []

        lines = so.full_text.split('\n')
        matches = []
        matched_line_nums = set()

        for i, line in enumerate(lines):
            if query.lower() in line.lower():
                matched_line_nums.add(i)

        # Build context windows
        result = []
        for ln in sorted(matched_line_nums):
            start = max(0, ln - context_lines)
            end = min(len(lines), ln + context_lines + 1)
            for j in range(start, end):
                marker = ">>>" if j == ln else "   "
                result.append((j + 1, f"{marker} {lines[j]}"))
            result.append((0, "---"))  # separator (line 0 is a sentinel)
        return result

    def metadata(self, ref: str) -> Optional[Dict]:
        """Return metadata for a stashed output."""
        so = self._outputs.get(ref)
        if not so:
            return None
        return {
            "ref": so.ref,
            "server": so.server,
            "tool": so.tool,
            "content_type": so.content_type,
            "total_lines": so.total_lines,
            "total_chars": so.total_chars,
            "iterations_ago": self._counter - int(so.ref.split('_')[1]) - 1,
        }

    def list_refs(self) -> List[Dict]:
        """List all active references."""
        return [self.metadata(r) for r in self._refs_lru if r in self._outputs]

    def drop(self, ref: str) -> bool:
        """Manually drop a stashed output to free memory."""
        if ref in self._outputs:
            del self._outputs[ref]
            self._refs_lru = [r for r in self._refs_lru if r != ref]
            return True
        return False


# =============================================================================
# CONTENT TYPE DETECTION
# =============================================================================

def _detect_content_type(text: str) -> str:
    """Heuristically detect what kind of data this is."""
    stripped = text.strip()

    # JSON: starts with { or [ — try full parse with a size guard
    if stripped.startswith('{') or stripped.startswith('['):
        if len(stripped) < 1_000_000:  # Don't attempt parse on multi-megabyte blobs
            try:
                json.loads(stripped)
                return "json"
            except (json.JSONDecodeError, ValueError):
                pass

    # Diff: starts with diff --git or contains hunk headers
    if stripped.startswith("diff --git") or "@@ -" in text[:500]:
        return "diff"

    # CSV / table: contains many commas in first few lines
    lines = stripped.split('\n')[:5]
    comma_lines = sum(1 for l in lines if l.count(',') >= 2)
    if comma_lines >= 3:
        return "csv"

    # File listing: lines start with common path prefixes or -rw-
    listing_patterns = [r'^\s*[-dl]', r'^\s*\d+\.\s+[/~\w]', r'^\s*[~/\w].*\.(py|js|ts|json|md|txt)\b']
    matches = sum(1 for pat in listing_patterns for l in lines if re.search(pat, l))
    if matches >= 3:
        return "listing"

    # Code: contains common syntax patterns
    code_markers = ['def ', 'class ', 'import ', 'function', 'const ', 'let ', 'var ']
    if any(m in text[:2000] for m in code_markers):
        return "code"

    return "text"


# =============================================================================
# SMART PREVIEW BUILDERS
# =============================================================================

def _build_preview(text: str, content_type: str, max_chars: int = 8000) -> str:
    """Build a structure-aware preview that never breaks mid-structure."""
    total = len(text)
    lines = text.split('\n')

    if total <= max_chars:
        return text

    if content_type == "json":
        return _preview_json(text, max_chars)
    elif content_type == "diff":
        return _preview_diff(text, max_chars)
    elif content_type == "listing":
        return _preview_listing(lines, max_chars)
    elif content_type == "csv":
        return _preview_csv(lines, max_chars)
    elif content_type == "code":
        return _preview_code(lines, max_chars)
    else:
        return _preview_text(lines, max_chars)


def _preview_json(text: str, max_chars: int) -> str:
    """Preview JSON: keep schema, truncate large arrays/objects."""
    try:
        data = json.loads(text)
        preview = _truncate_json_value(data, max_items=10, max_depth=3, max_string=200)
        preview_str = json.dumps(preview, indent=2, ensure_ascii=False)
        if len(preview_str) > max_chars:
            preview_str = preview_str[:max_chars]
            # Snap to last valid JSON structure
            last_brace = preview_str.rfind('}')
            last_bracket = preview_str.rfind(']')
            snap = max(last_brace, last_bracket)
            if snap > max_chars * 0.8:
                preview_str = preview_str[:snap + 1]
        total_items = _count_json_items(data)
        return (
            f"{preview_str}\n\n"
            f"[PREVIEW: JSON with {total_items} total items. "
            f"Use read_output(ref=..., offset=0) to paginate raw text.]"
        )
    except (json.JSONDecodeError, ValueError):
        # Fallback to text preview if JSON is malformed
        return _preview_text(text.split('\n'), max_chars)


def _preview_diff(text: str, max_chars: int) -> str:
    """Preview diff: keep first N complete hunks."""
    hunks = text.split('\ndiff --git')
    preview_parts = [hunks[0]] if hunks else []
    current_len = len(preview_parts[0]) if preview_parts else 0

    for hunk in hunks[1:]:
        candidate = '\ndiff --git' + hunk
        if current_len + len(candidate) > max_chars * 0.95:
            break
        preview_parts.append(candidate)
        current_len += len(candidate)

    total_hunks = len(hunks)
    shown_hunks = len(preview_parts)
    preview = ''.join(preview_parts)
    return (
        f"{preview}\n\n"
        f"[PREVIEW: {shown_hunks} of {total_hunks} hunks shown. "
        f"Use read_output(ref=..., offset={len(preview)}) for next chunk.]"
    )


def _preview_listing(lines: List[str], max_chars: int) -> str:
    """Preview file listing: first N entries with count."""
    preview_lines = []
    current_len = 0
    for line in lines:
        if current_len + len(line) + 1 > max_chars * 0.95:
            break
        preview_lines.append(line)
        current_len += len(line) + 1

    shown = len(preview_lines)
    total = len(lines)
    preview = '\n'.join(preview_lines)
    return (
        f"{preview}\n\n"
        f"[PREVIEW: {shown} of {total} entries shown. "
        f"Use read_output(ref=..., offset={current_len}) for more.]"
    )


def _preview_csv(lines: List[str], max_chars: int) -> str:
    """Preview CSV: header + first N rows."""
    if not lines:
        return ""
    header = lines[0]
    preview = [header]
    current_len = len(header) + 1

    for row in lines[1:]:
        if current_len + len(row) + 1 > max_chars * 0.95:
            break
        preview.append(row)
        current_len += len(row) + 1

    shown_rows = len(preview) - 1  # minus header
    total_rows = len(lines) - 1
    return (
        f"{'\n'.join(preview)}\n\n"
        f"[PREVIEW: header + {shown_rows} of {total_rows} rows. "
        f"Use read_output(ref=..., offset={current_len}) for more.]"
    )


def _preview_code(lines: List[str], max_chars: int) -> str:
    """Preview code: first N complete lines, snap at function boundary if possible."""
    preview_lines = []
    current_len = 0
    for line in lines:
        if current_len + len(line) + 1 > max_chars * 0.95:
            break
        preview_lines.append(line)
        current_len += len(line) + 1

    # Try to snap to a blank line or function boundary for clean break
    while preview_lines and preview_lines[-1].strip():
        if len(preview_lines) > 1:
            preview_lines.pop()
            current_len = sum(len(l) + 1 for l in preview_lines)
        else:
            break

    shown = len(preview_lines)
    total = len(lines)
    return (
        f"{'\n'.join(preview_lines)}\n\n"
        f"[PREVIEW: first {shown} of {total} lines. "
        f"Use read_output(ref=..., offset={current_len}) for more.]"
    )


def _preview_text(lines: List[str], max_chars: int) -> str:
    """Preview plain text: snap to paragraph boundary."""
    preview_lines = []
    current_len = 0
    for line in lines:
        if current_len + len(line) + 1 > max_chars * 0.95:
            break
        preview_lines.append(line)
        current_len += len(line) + 1

    # Snap to last blank line for clean break
    while preview_lines and preview_lines[-1].strip():
        if len(preview_lines) > 1:
            preview_lines.pop()
            current_len = sum(len(l) + 1 for l in preview_lines)
        else:
            break

    shown = len(preview_lines)
    total = len(lines)
    if shown == 0 and lines:
        # Single oversized line (or every line too long): show the first
        # max_chars of raw text so the preview is never empty.
        head = lines[0][: int(max_chars * 0.95)]
        return (
            f"{head}\n\n"
            f"[PREVIEW: first {len(head)} chars of a {total}-line output. "
            f"Use read_output(ref=..., offset={len(head)}) for more.]"
        )
    return (
        f"{'\n'.join(preview_lines)}\n\n"
        f"[PREVIEW: first {shown} of {total} lines ({current_len} chars). "
        f"Use read_output(ref=..., offset={current_len}) for more.]"
    )


# =============================================================================
# JSON TRUNCATION HELPERS
# =============================================================================

def _truncate_json_value(value, max_items: int, max_depth: int, max_string: int):
    """Recursively truncate a JSON value for preview."""
    if max_depth <= 0:
        return "..."

    if isinstance(value, dict):
        preview = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= max_items:
                preview["..."] = f"({len(value) - max_items} more keys)"
                break
            preview[k] = _truncate_json_value(v, max_items, max_depth - 1, max_string)
        return preview

    if isinstance(value, list):
        preview = []
        for i, item in enumerate(value):
            if i >= max_items:
                preview.append(f"... ({len(value) - max_items} more items)")
                break
            preview.append(_truncate_json_value(item, max_items, max_depth - 1, max_string))
        return preview

    if isinstance(value, str):
        if len(value) > max_string:
            return value[:max_string] + "..."
        return value

    return value


def _count_json_items(value) -> int:
    """Count total leaf-ish items in a JSON structure."""
    if isinstance(value, dict):
        return sum(_count_json_items(v) for v in value.values())
    if isinstance(value, list):
        return sum(_count_json_items(v) for v in value)
    return 1


# =============================================================================
# MODULE-LEVEL SINGLETON (session-scoped)
# =============================================================================

_buffer_instance: Optional[ToolOutputBuffer] = None


def get_output_buffer() -> ToolOutputBuffer:
    """Get the session-scoped output buffer singleton."""
    global _buffer_instance
    if _buffer_instance is None:
        _buffer_instance = ToolOutputBuffer()
    return _buffer_instance


def reset_output_buffer():
    """Reset the buffer (e.g., on session start)."""
    global _buffer_instance
    _buffer_instance = ToolOutputBuffer()
