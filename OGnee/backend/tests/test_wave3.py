#!/usr/bin/env python3
"""
tests/test_wave3.py - Wave 3 review-fix tests (stdlib unittest).

Covers:
- B1: _extract_json_objects string tracking — unbalanced braces and escaped
  quotes inside JSON string values (write_file content with } and {).
- M1: gate prompts fail CLOSED (Enter/garbage/EOF/Ctrl-C => Deny).
- M2/M3: local tools and write/move/multi-read/network tool categories go
  through the gate layer (EXCLUDED_PATHS enforced).
- M4: local/speak is opt-in (disabled by default).
- M5: truncated tool batches synthesize rejection results for every
  unexecuted tool_call_id (no dangling ids).
- M6: _trim_messages never splits assistant tool_calls from role:"tool"
  replies and counts tool-call argument tokens.
- M8: per-tool timeout resolution.
- M9: credential stores excluded; tool-call args redacted in session logs.
- Minor: single-line oversized stash preview is not empty.

Run from the repo root:  python3 -m unittest discover -s tests
"""

import asyncio
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

# ---------------------------------------------------------------------------
# Package bootstrap (same pattern as test_wave1/2).
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parents[1]
if "backend" not in sys.modules:
    _pkg = types.ModuleType("backend")
    _pkg.__path__ = [str(PKG_DIR)]
    sys.modules["backend"] = _pkg

from backend import display  # noqa: E402
from backend import inputhub  # noqa: E402
from backend.agent import (  # noqa: E402
    _extract_json_objects,
    _looks_like_tool_json,
    _message_token_cost,
    _trim_messages,
    extract_tool_calls_text,
    build_tool_registry,
)
from backend.config import EXCLUDED_PATHS, get_tool_timeout  # noqa: E402
from backend.context import create_session  # noqa: E402
from backend.gates import gate_tool_call, gate_file_read, GateResult  # noqa: E402
from backend.output_buffer import _build_preview  # noqa: E402
from backend.session_log import redact_args  # noqa: E402
from backend.tools import (  # noqa: E402
    SPEAK_ENABLED,
    ToolResult,
    execute_tool_calls,
)


def _session(tier: str = "supervised"):
    return create_session(model_name="test-model", trust_tier=tier)


# ---------------------------------------------------------------------------
# B1 — string-aware JSON extraction
# ---------------------------------------------------------------------------

class TestExtractJsonStrings(unittest.TestCase):
    def test_unbalanced_close_brace_in_string(self):
        text = '{"tool": "local/read_file", "arguments": {"path": "a}b"}}'
        objs = _extract_json_objects(text)
        self.assertEqual(len(objs), 1)
        parsed = json.loads(objs[0])
        self.assertEqual(parsed["arguments"]["path"], "a}b")

    def test_write_file_content_with_braces(self):
        content = "def f():\n    return {}\n}\n{ unbalanced"
        text = json.dumps({
            "tool": "local/write_file",
            "arguments": {"path": "/tmp/x.py", "content": content},
        })
        objs = _extract_json_objects(text)
        self.assertEqual(len(objs), 1)
        self.assertEqual(json.loads(objs[0])["arguments"]["content"], content)

    def test_escaped_quotes_in_string(self):
        content = 'she said "hello } {" loudly'
        text = json.dumps({"tool": "local/write_file",
                           "arguments": {"path": "/tmp/y", "content": content}})
        objs = _extract_json_objects(text)
        self.assertEqual(len(objs), 1)
        self.assertEqual(json.loads(objs[0])["arguments"]["content"], content)

    def test_extract_tool_calls_text_with_brace_strings(self):
        registry = {"write_file": "local/write_file", "local/write_file": "local/write_file"}
        text = (
            'Some prose before. {"tool": "local/write_file", "arguments": '
            '{"path": "/tmp/z.py", "content": "if x: } else { {"}}} done.'
        )
        calls = extract_tool_calls_text(text, registry)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool"], "write_file")

    def test_multiple_objects_with_strings(self):
        text = (
            '{"tool": "a/b", "arguments": {"x": "}"}} prose '
            '{"tool": "c/d", "arguments": {"y": "{"}}'
        )
        objs = _extract_json_objects(text)
        self.assertEqual(len(objs), 2)


# ---------------------------------------------------------------------------
# M1 — gate prompts fail closed
# ---------------------------------------------------------------------------

class TestPromptChoiceFailClosed(unittest.TestCase):
    OPTIONS = {"1": "Allow", "2": "Deny"}

    def _run(self, side_effect):
        with mock.patch.object(inputhub.InputHub, "read_line", side_effect=side_effect):
            return display.prompt_choice("GATE: test", dict(self.OPTIONS))

    def test_enter_denies(self):
        self.assertEqual(self._run([""]), "2")

    def test_eof_denies(self):
        self.assertEqual(self._run(EOFError()), "2")

    def test_ctrl_c_denies(self):
        self.assertEqual(self._run(KeyboardInterrupt()), "2")

    def test_garbage_reprompts_then_valid(self):
        self.assertEqual(self._run(["garbage", "1"]), "1")

    def test_garbage_then_enter_denies(self):
        self.assertEqual(self._run(["garbage", ""]), "2")

    def test_prompt_approval_enter_denies(self):
        with mock.patch.object(inputhub.InputHub, "read_line", return_value=""):
            self.assertEqual(display.prompt_approval("a", "b"), "n")


# ---------------------------------------------------------------------------
# M2/M3 — gate coverage
# ---------------------------------------------------------------------------

class TestGateCoverage(unittest.TestCase):
    def test_ssh_dir_excluded(self):
        state = _session()
        decision = gate_file_read(os.path.expanduser("~/.ssh/id_ed25519"), state)
        self.assertEqual(decision.result, GateResult.DENY)

    def test_aws_and_gnupg_excluded(self):
        from backend.config import is_path_excluded
        for p in ("~/.aws/credentials", "~/.gnupg/secring.gpg",
                  "~/.config/gh/hosts.yml", "~/.secrets/lms.env",
                  "~/.ssh/id_ed25519"):
            expanded = os.path.expanduser(p)
            self.assertTrue(is_path_excluded(expanded),
                            f"{p} not covered by EXCLUDED_PATHS")

    def test_local_read_file_excluded_path_denied(self):
        state = _session()
        allowed = gate_tool_call(
            "local", "read_file",
            {"path": os.path.expanduser("~/.ssh/id_ed25519")}, state).allowed
        self.assertFalse(allowed)

    def test_read_multiple_files_excluded_path_denied(self):
        state = _session()
        allowed = gate_tool_call(
            "filesystem", "read_multiple_files",
            {"paths": ["/tmp/ok", os.path.expanduser("~/.aws/credentials")]}, state).allowed
        self.assertFalse(allowed)

    def test_move_file_destination_gated(self):
        state = _session()
        allowed = gate_tool_call(
            "filesystem", "move_file",
            {"source": "/tmp/a", "destination": os.path.expanduser("~/.ssh/x")},
            state).allowed
        self.assertFalse(allowed)

    def test_create_directory_gated(self):
        state = _session()
        allowed = gate_tool_call(
            "filesystem", "create_directory",
            {"path": os.path.expanduser("~/.gnupg/evil")}, state).allowed
        self.assertFalse(allowed)

    def test_network_tool_denied_without_tier(self):
        # Tier 1 (supervised) has can_network=False; gate must not allow
        # silently. A bare Enter at the 1-7 menu re-prompts (never allows);
        # closing input then denies.
        state = _session("supervised")
        with mock.patch.object(inputhub.InputHub, "read_line",
                               side_effect=["", EOFError()]) as rl:
            allowed = gate_tool_call(
                "searxng", "searxng_web_search", {"query": "hi"}, state).allowed
        self.assertFalse(allowed)
        self.assertEqual(rl.call_count, 2)

    def test_network_tool_operator_deny(self):
        state = _session("supervised")
        with mock.patch.object(inputhub.InputHub, "read_line", side_effect=["3"]):
            verdict = gate_tool_call(
                "searxng", "searxng_web_search", {"query": "hi"}, state)
        self.assertFalse(verdict.allowed)
        self.assertTrue(verdict.by_operator)

    def test_network_tool_allowed_with_tier(self):
        state = _session("trusted")
        allowed = gate_tool_call(
            "searxng", "searxng_web_search", {"query": "hi"}, state).allowed
        self.assertTrue(allowed)

    def test_speak_gated_as_network(self):
        state = _session("supervised")
        with mock.patch.object(inputhub.InputHub, "read_line",
                               side_effect=["", EOFError()]):
            allowed = gate_tool_call(
                "local", "speak", {"text": "hello"}, state).allowed
        self.assertFalse(allowed)


# ---------------------------------------------------------------------------
# M4 — speak is opt-in
# ---------------------------------------------------------------------------

class TestSpeakOptIn(unittest.TestCase):
    def test_speak_disabled_by_default(self):
        self.assertFalse(SPEAK_ENABLED)


# ---------------------------------------------------------------------------
# M5 — truncated batches synthesize rejection results
# ---------------------------------------------------------------------------

class TestBatchTruncation(unittest.TestCase):
    def test_denied_write_synthesizes_skipped_results(self):
        state = _session()
        excluded = os.path.expanduser("~/Documents/secret.txt")
        calls = [
            {"server": "local", "tool": "write_file",
             "args": {"path": excluded, "content": "x"}, "id": "call_1"},
            {"server": "local", "tool": "write_file",
             "args": {"path": "/tmp/never.txt", "content": "y"}, "id": "call_2"},
            {"server": "local", "tool": "write_file",
             "args": {"path": "/tmp/never2.txt", "content": "z"}, "id": "call_3"},
        ]
        results = asyncio.run(execute_tool_calls({}, calls, state))
        # 1:1 alignment — no dangling tool_call_ids for native mode.
        self.assertEqual(len(results), len(calls))
        self.assertFalse(results[0].success)
        self.assertTrue(results[0].was_denied)
        for r in results[1:]:
            self.assertFalse(r.success)
            self.assertIn("skipped", r.error)
            self.assertTrue(r.was_denied)

    def test_step_mode_synthesizes_skipped_results(self):
        class FakeClient:
            def call_tool(self, name, args):
                return {"result": "ok"}

        state = _session("trusted")  # direct writes to /tmp allowed, no prompt
        state.step_mode = True
        calls = [
            {"server": "fs", "tool": "write_file",
             "args": {"path": "/tmp/step1.txt", "content": "x"}, "id": "c1"},
            {"server": "fs", "tool": "write_file",
             "args": {"path": "/tmp/step2.txt", "content": "y"}, "id": "c2"},
        ]
        results = asyncio.run(
            execute_tool_calls({"fs": FakeClient()}, calls, state))
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
        self.assertIn("skipped", results[1].error)


# ---------------------------------------------------------------------------
# M6 — _trim_messages keeps tool-exchange groups atomic
# ---------------------------------------------------------------------------

def _big(n):
    return "word " * n


class TestTrimToolGroups(unittest.TestCase):
    def _state(self, limit=2000):
        state = _session()
        state.model_config.context_limit = limit
        return state

    def _build(self):
        msgs = [
            {"role": "system", "content": _big(50)},
            {"role": "user", "content": "do the task"},
        ]
        for i in range(5):
            msgs.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "s__t", "arguments": _big(40)},
                }],
            })
            msgs.append({"role": "tool", "tool_call_id": f"call_{i}",
                         "content": _big(40)})
            msgs.append({"role": "assistant", "content": f"turn {i} done"})
        return msgs

    def test_no_split_tool_groups(self):
        msgs = self._build()
        _trim_messages(msgs, self._state())
        ids_with_calls = set()
        ids_with_replies = set()
        for m in msgs:
            for tc in (m.get("tool_calls") or []):
                ids_with_calls.add(tc["id"])
            if m.get("role") == "tool":
                ids_with_replies.add(m["tool_call_id"])
        self.assertEqual(ids_with_calls, ids_with_replies,
                         "assistant tool_calls split from their tool replies")

    def test_tool_call_arguments_counted(self):
        plain = {"role": "assistant", "content": "hi"}
        with_call = {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "x", "type": "function",
                            "function": {"name": "t", "arguments": _big(500)}}],
        }
        self.assertGreater(_message_token_cost(with_call),
                           _message_token_cost(plain) + 100)


# ---------------------------------------------------------------------------
# M8 — per-tool timeouts
# ---------------------------------------------------------------------------

class TestToolTimeouts(unittest.TestCase):
    def test_default_timeout(self):
        self.assertEqual(get_tool_timeout("filesystem", "read_file"), 15)

    def test_deep_research_default(self):
        self.assertEqual(get_tool_timeout("research", "deep_research"), 300)
        self.assertEqual(get_tool_timeout("research", "deep-research"), 300)

    def test_env_override(self):
        with mock.patch.dict(os.environ,
                             {"OGNEE_TOOL_TIMEOUT_MYTOOL": "42",
                              "OGNEE_TOOL_TIMEOUT": "25"}):
            self.assertEqual(get_tool_timeout("srv", "mytool"), 42)
            self.assertEqual(get_tool_timeout("srv", "other"), 25)

    def test_server_scoped_entry(self):
        self.assertEqual(get_tool_timeout("browser", "navigate"), 60)


# ---------------------------------------------------------------------------
# M9 — args redaction
# ---------------------------------------------------------------------------

class TestArgsRedaction(unittest.TestCase):
    def test_sensitive_keys_redacted(self):
        args = {"path": "/tmp/x", "api_key": "supersecretvalue",
                "nested": {"password": "hunter2", "token": "abc"}}
        out = redact_args(args)
        self.assertEqual(out["path"], "/tmp/x")
        self.assertEqual(out["api_key"], "***REDACTED***")
        self.assertEqual(out["nested"]["password"], "***REDACTED***")
        self.assertEqual(out["nested"]["token"], "***REDACTED***")

    def test_bearer_in_content_redacted(self):
        args = {"content": "curl -H 'Authorization: Bearer abcdef123456' x"}
        out = redact_args(args)
        self.assertNotIn("abcdef123456", out["content"])

    def test_list_values_recursed(self):
        args = {"paths": ["/tmp/a"], "headers": [{"secret": "s3cr3t"}]}
        out = redact_args(args)
        self.assertEqual(out["paths"], ["/tmp/a"])
        self.assertEqual(out["headers"][0]["secret"], "***REDACTED***")


# ---------------------------------------------------------------------------
# Minor — single-line oversized stash preview
# ---------------------------------------------------------------------------

class TestStashPreview(unittest.TestCase):
    def test_single_huge_line_preview_not_empty(self):
        text = "x" * 25000  # one line, no newlines
        preview = _build_preview(text, "text", max_chars=1000)
        self.assertIn("x" * 100, preview)
        self.assertIn("PREVIEW", preview)
        self.assertNotIn("first 0 of", preview)


# ---------------------------------------------------------------------------
# Minor — fenced code block replies are not swallowed
# ---------------------------------------------------------------------------

class TestLooksLikeToolJson(unittest.TestCase):
    def test_fenced_code_answer_not_suppressed(self):
        text = "Here's the code:\n```python\ndef f():\n    return 1\n```"
        self.assertFalse(_looks_like_tool_json(text))

    def test_fenced_tool_json_suppressed(self):
        text = '```json\n{"tool": "local/read_file", "arguments": {"path": "/tmp/x"}}\n```'
        self.assertTrue(_looks_like_tool_json(text))

    def test_fenced_plain_example_not_suppressed(self):
        text = '```json\n{"example": true, "note": "not a tool call"}\n```'
        self.assertFalse(_looks_like_tool_json(text))


# ---------------------------------------------------------------------------
# M10 — ssh module header import (NameError regression)
# ---------------------------------------------------------------------------

class TestSshHeaderImport(unittest.TestCase):
    def test_header_importable(self):
        from backend import ssh
        self.assertTrue(hasattr(ssh, "header"))


if __name__ == "__main__":
    unittest.main()
