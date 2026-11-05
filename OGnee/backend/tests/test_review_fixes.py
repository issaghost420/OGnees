#!/usr/bin/env python3
"""
tests/test_review_fixes.py - regressions for fixes from the merge review.

- _trim_messages must never leave a role:"tool" message without the
  assistant tool_calls message that owns it (the MIN_RECENT_MESSAGES floor
  used to cut into the middle of a parallel tool-call group -> API 400).
- the fallback token counter includes native tool_call arguments.
- /buffer search merges overlapping context windows (no repeated lines).

Run from the repo root:  python3 tests/test_review_fixes.py
"""

import sys
import types
import unittest
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]
if "backend" not in sys.modules:
    _pkg = types.ModuleType("backend")
    _pkg.__path__ = [str(PKG_DIR)]
    sys.modules["backend"] = _pkg

from backend.agent import _trim_messages  # noqa: E402
from backend.context import count_message_tokens, create_session  # noqa: E402
from backend.output_buffer import ToolOutputBuffer  # noqa: E402


def _orphans(messages):
    bad = []
    for i, m in enumerate(messages):
        if m.get("role") != "tool":
            continue
        j = i - 1
        while j >= 0 and messages[j].get("role") == "tool":
            j -= 1
        if j < 0 or not messages[j].get("tool_calls"):
            bad.append(i)
    return bad


class TestTrimKeepsToolGroups(unittest.TestCase):
    def test_floor_inside_parallel_group(self):
        state = create_session(model_name="test-model", trust_tier="supervised")
        state.model_config.context_limit = 2000
        big = "x " * 600
        calls = [{"id": f"c{i}", "type": "function",
                  "function": {"name": "f", "arguments": "{}"}} for i in range(6)]
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
                {"role": "assistant", "content": None, "tool_calls": calls}]
        msgs += [{"role": "tool", "tool_call_id": f"c{i}", "content": big} for i in range(6)]
        msgs.append({"role": "assistant", "content": "done"})
        _trim_messages(msgs, state)
        self.assertEqual(_orphans(msgs), [])

    def test_many_groups_still_trimmed_cleanly(self):
        state = create_session(model_name="test-model", trust_tier="supervised")
        state.model_config.context_limit = 3000
        big = "y " * 400
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        for g in range(8):
            calls = [{"id": f"g{g}c{i}", "type": "function",
                      "function": {"name": "f", "arguments": "{}"}} for i in range(3)]
            msgs.append({"role": "assistant", "content": None, "tool_calls": calls})
            msgs += [{"role": "tool", "tool_call_id": f"g{g}c{i}", "content": big} for i in range(3)]
        before = len(msgs)
        _trim_messages(msgs, state)
        self.assertLess(len(msgs), before)
        self.assertEqual(_orphans(msgs), [])


class TestTokenFallback(unittest.TestCase):
    def test_tool_call_arguments_counted(self):
        plain = [{"role": "assistant", "content": None}]
        heavy = [{"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "write_file", "arguments": "{\"content\": \"" + "word " * 500 + "\"}"}}]}]
        self.assertGreater(count_message_tokens(heavy) - count_message_tokens(plain), 300)


class TestBufferSearch(unittest.TestCase):
    def test_overlapping_windows_merged(self):
        buf = ToolOutputBuffer()
        text = "\n".join(["noise"] * 5 + ["hit one", "between", "hit two"] + ["noise"] * 5)
        ref, _ = buf.store("srv", "tool", 0, text)
        rows = [r for r in buf.search(ref, "hit") if r[0] != 0]
        line_nums = [n for n, _ in rows]
        self.assertEqual(len(line_nums), len(set(line_nums)))
        self.assertEqual(sum(1 for _, t in rows if t.startswith(">>>")), 2)


if __name__ == "__main__":
    unittest.main()
