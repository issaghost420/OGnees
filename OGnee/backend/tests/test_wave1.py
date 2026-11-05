#!/usr/bin/env python3
"""
tests/test_wave1.py - Wave 1 hardening tests (stdlib unittest).

Covers:
- Tool-call parsers per model profile (native / qwen-xml / gemma / malformed),
  including the guarantee that markdown-table prose and JSON examples in text
  never produce phantom tool calls.
- Registry gating: names not in the registry are rejected, never dispatched.
- Loop detector trip on 3 identical (tool, args) calls in a row.
- Tool-result formatting: native role:"tool" messages with tool_call_id,
  shadow/redirect formatting, and the stashed-output reference fix
  (tools.py preview/NameError regression).
- Token accounting: record_tool_call must not double-count on top of
  API-reported usage; session-state injection must strip cleanly even with
  blank lines inside the block.

Run from the repo root:  python3 tests/test_wave1.py
"""

import asyncio
import json
import sys
import types
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Package bootstrap: the repo directory is the `backend` package, but its
# directory name may not be a valid Python identifier (and must NOT be put on
# sys.path itself, or logging.py would shadow the stdlib `logging` module).
# Register it as a namespace-style package alias instead.
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parents[1]
if "backend" not in sys.modules:
    _pkg = types.ModuleType("backend")
    _pkg.__path__ = [str(PKG_DIR)]
    sys.modules["backend"] = _pkg

from backend.agent import (  # noqa: E402
    LoopDetector,
    build_tool_registry,
    detect_tool_profile,
    extract_tool_calls_native,
    extract_tool_calls_text,
    resolve_tool_name,
)
from backend.context import (  # noqa: E402
    create_session,
    inject_state_into_messages,
    record_tool_call,
    update_token_count,
)
from backend.mcp import MCPManager, make_native_name  # noqa: E402
from backend.output_buffer import MAX_TOOL_RESULT_CHARS  # noqa: E402
from backend.tools import (  # noqa: E402
    ToolResult,
    execute_tool_call,
    format_tool_result_for_llm,
    format_tool_results_for_llm,
    get_failure_tracker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeMCP:
    """Minimal stand-in for MCPManager (no subprocesses, no network)."""

    def __init__(self):
        self.all_tools = [
            {"name": "read_file", "full_name": "local-files/read_file",
             "server": "local-files", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "execute_command", "full_name": "shell/execute_command",
             "server": "shell", "inputSchema": {"type": "object", "properties": {}}},
        ]
        self.servers = {}


def make_registry():
    return build_tool_registry(FakeMCP())


def make_state():
    # "supervised" tier: no per-tool approval prompts, tools enabled.
    return create_session(model_name="qwen3-32b", trust_tier="supervised")


class FakeClient:
    """Fake MCP server client returning canned call_tool output."""

    def __init__(self, payload):
        self.payload = payload
        self.tools = []

    def call_tool(self, tool, args):
        return self.payload


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestToolParsers(unittest.TestCase):
    def setUp(self):
        self.registry = make_registry()

    def test_qwen_xml_json(self):
        text = ('Let me check.\n'
                '<tool_call>{"tool": "shell/execute_command", '
                '"arguments": {"command": "ls -la"}}</tool_call>')
        calls = extract_tool_calls_text(text, self.registry, profile="qwen-xml")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["server"], "shell")
        self.assertEqual(calls[0]["tool"], "execute_command")
        self.assertEqual(calls[0]["args"], {"command": "ls -la"})

    def test_gemma_function_call(self):
        text = ('<start_function_call>call:local-files/read_file'
                '{"path": "/tmp/x"}<end_function_call>')
        calls = extract_tool_calls_text(text, self.registry, profile="gemma")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["server"], "local-files")
        self.assertEqual(calls[0]["tool"], "read_file")
        self.assertEqual(calls[0]["args"], {"path": "/tmp/x"})

    def test_gemma_not_parsed_for_other_profiles(self):
        text = ('<start_function_call>call:local-files/read_file'
                '{"path": "/tmp/x"}<end_function_call>')
        calls = extract_tool_calls_text(text, self.registry, profile="json")
        self.assertEqual(calls, [])

    def test_native_tool_calls(self):
        response = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": make_native_name("shell", "execute_command"),
                            "arguments": '{"command": "echo hi"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }]
        }
        calls = extract_tool_calls_native(response, self.registry)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_123")
        self.assertEqual(calls[0]["server"], "shell")
        self.assertEqual(calls[0]["tool"], "execute_command")
        self.assertEqual(calls[0]["args"], {"command": "echo hi"})
        self.assertNotIn("parse_error", calls[0])

    def test_native_malformed_args_flagged_not_munged(self):
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_bad",
                        "function": {
                            "name": make_native_name("shell", "execute_command"),
                            "arguments": '{"command": "echo hi',  # truncated
                        },
                    }],
                }
            }]
        }
        calls = extract_tool_calls_native(response, self.registry)
        self.assertEqual(len(calls), 1)
        # Malformed JSON must surface as parse_error, never silently {}
        self.assertIn("parse_error", calls[0])
        self.assertEqual(calls[0]["args"], {})

    def test_markdown_table_prose_produces_no_phantom_calls(self):
        # Regression for session-log CLASS A: table cells that merely look
        # like server/tool names must never become tool calls.
        text = (
            "Here's the summary:\n\n"
            "| Endpoint | Status |\n"
            "|---|---|\n"
            "| sysmonitor/ws | ok |\n"
            "| api/metrics | 200 |\n"
            "| api/reset | N/A |\n"
            "| /tmp/ram-watchdog-state.json | written |\n\n"
            'Example payload: {"name": "foo", "description": "a widget"}\n'
            "See https://example.com/docs/path for details."
        )
        calls = extract_tool_calls_text(text, self.registry, profile="json")
        self.assertEqual(calls, [])

    def test_unknown_name_in_explicit_format_is_kept_for_validation(self):
        # Explicit formats are parsed, but resolve_tool_name rejects them —
        # the agent loop sends a correction instead of dispatching.
        text = '<tool_call>{"tool": "evil/server_hack", "arguments": {}}</tool_call>'
        calls = extract_tool_calls_text(text, self.registry, profile="qwen-xml")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(
            resolve_tool_name(calls[0]["server"], calls[0]["tool"], self.registry))

    def test_bare_name_resolves_when_unambiguous(self):
        self.assertEqual(
            resolve_tool_name("unknown", "execute_command", self.registry),
            "shell/execute_command",
        )

    def test_native_alias_resolves(self):
        native = make_native_name("local-files", "read_file")
        self.assertNotIn("/", native)
        self.assertEqual(
            resolve_tool_name("unknown", native, self.registry),
            "local-files/read_file",
        )


# ---------------------------------------------------------------------------
# Loop detector
# ---------------------------------------------------------------------------

class TestLoopDetector(unittest.TestCase):
    def test_trips_on_third_identical_call(self):
        det = LoopDetector(max_repeats=3)
        args = {"path": "/tmp/x"}
        self.assertFalse(det.check("local-files/read_file", args))
        self.assertFalse(det.check("local-files/read_file", args))
        self.assertTrue(det.check("local-files/read_file", args))

    def test_different_args_reset_the_streak(self):
        det = LoopDetector(max_repeats=3)
        det.check("local-files/read_file", {"path": "/a"})
        det.check("local-files/read_file", {"path": "/a"})
        det.check("local-files/read_file", {"path": "/b"})  # breaks the streak
        self.assertFalse(det.check("local-files/read_file", {"path": "/a"}))

    def test_arg_order_insensitive(self):
        det = LoopDetector(max_repeats=2)
        det.check("shell/execute_command", {"a": 1, "b": 2})
        self.assertTrue(det.check("shell/execute_command", {"b": 2, "a": 1}))


# ---------------------------------------------------------------------------
# Tool result formatting / shadow of results
# ---------------------------------------------------------------------------

class TestToolResultFormatting(unittest.TestCase):
    def test_native_tool_message_has_tool_call_id(self):
        result = ToolResult(success=True, output="hello")
        msg = format_tool_result_for_llm(result, "shell/execute_command", "call_1")
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "call_1")
        self.assertEqual(msg["name"], "shell/execute_command")
        self.assertEqual(msg["content"], "hello")

    def test_error_and_denied_formatting(self):
        err = format_tool_result_for_llm(
            ToolResult(success=False, output="", error="boom"), "shell/x", "c2")
        self.assertEqual(err["content"], "[ERROR] boom")
        denied = format_tool_result_for_llm(
            ToolResult(success=False, output="", error="no", was_denied=True),
            "shell/x", "c3")
        self.assertTrue(denied["content"].startswith("[DENIED]"))

    def test_redirected_shadow_formatting(self):
        res = ToolResult(success=True, output="data", was_redirected=True,
                         redirect_path="/shadow/home/x")
        msg = format_tool_result_for_llm(res, "local-files/write_file", "c4")
        self.assertIn("[REDIRECTED to shadow: /shadow/home/x]", msg["content"])
        self.assertIn("data", msg["content"])

    def test_batch_formatting_pairs_ids(self):
        results = [ToolResult(success=True, output="a"),
                   ToolResult(success=False, output="", error="e")]
        calls = [{"server": "shell", "tool": "execute_command", "id": "1"},
                 {"server": "shell", "tool": "execute_command", "id": "2"}]
        msgs = format_tool_results_for_llm(results, calls)
        self.assertEqual([m["tool_call_id"] for m in msgs], ["1", "2"])


class TestStashedOutputFix(unittest.TestCase):
    """Regression for the tools.py preview/NameError bug: every successful
    result path must return valid output, and stashed results must carry the
    reference key inline."""

    def test_small_output_passes_through(self):
        state = make_state()
        client = FakeClient({"result": "short output"})
        result = asyncio.run(execute_tool_call(
            {"shell": client}, "shell", "execute_command",
            {"command": "echo hi"}, state))
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output, "short output")

    def test_large_output_stashed_with_reference_inline(self):
        state = make_state()
        big = "x" * (MAX_TOOL_RESULT_CHARS + 500)
        client = FakeClient({"result": big})
        result = asyncio.run(execute_tool_call(
            {"shell": client}, "shell", "execute_command",
            {"command": "cat big"}, state))
        self.assertTrue(result.success, result.error)
        self.assertIn("Reference: out_", result.output)
        self.assertIn("local/read_output", result.output)
        self.assertLess(len(result.output), len(big))

    def test_is_error_flag_propagates_to_failure(self):
        state = make_state()
        client = FakeClient({"result": "something failed", "isError": True})
        result = asyncio.run(execute_tool_call(
            {"shell": client}, "shell", "execute_command",
            {"command": "bad"}, state))
        self.assertFalse(result.success)
        self.assertEqual(result.error, "something failed")

    def test_unknown_server_does_not_touch_failure_tracker(self):
        state = make_state()
        tracker = get_failure_tracker()
        tracker.clear()
        result = asyncio.run(execute_tool_call(
            {}, "nosuchserver", "whatever", {}, state))
        self.assertFalse(result.success)
        self.assertIn("Unknown MCP server", result.error)
        self.assertEqual(tracker.consecutive, {})


# ---------------------------------------------------------------------------
# Token accounting / state injection
# ---------------------------------------------------------------------------

class TestTokenAccounting(unittest.TestCase):
    def test_record_tool_call_does_not_inflate_tokens_used(self):
        state = make_state()
        before = state.tokens_used
        record_tool_call(state, "shell", "execute_command",
                         {"command": "ls"}, "file1\nfile2", True)
        # Diagnostic per-tool estimate only — total untouched.
        self.assertEqual(state.tokens_used, before)
        self.assertGreater(state.tokens_tool_calls, 0)

    def test_api_usage_is_ground_truth_after_tool_calls(self):
        state = make_state()
        record_tool_call(state, "shell", "execute_command",
                         {"command": "ls"}, "output", True)
        update_token_count(state, api_usage={
            "prompt_tokens": 900, "completion_tokens": 100,
            "total_tokens": 1000,
        })
        # Tool tokens are inside prompt_tokens already; no double count.
        self.assertEqual(state.tokens_used, 1000)

    def test_state_block_strips_cleanly_with_blank_lines(self):
        state = make_state()
        # Force a failure summary (adds a blank line inside the block)
        record_tool_call(state, "shell", "execute_command",
                         {"command": "x"}, "err", False)
        messages = [{"role": "system", "content": "BASE PROMPT. Keep me."}]
        inject_state_into_messages(messages, state)
        inject_state_into_messages(messages, state)
        content = messages[0]["content"]
        self.assertEqual(content.count("[SESSION STATE]"), 1)
        self.assertEqual(content.count("[/SESSION STATE]"), 1)
        self.assertIn("BASE PROMPT. Keep me.", content)


# ---------------------------------------------------------------------------
# Profiles / MCP manager helpers (no subprocesses)
# ---------------------------------------------------------------------------

class TestProfiles(unittest.TestCase):
    def test_profile_detection(self):
        state = make_state()
        state.model_name = "gemma-4-it"
        state.model_supports_tools = False
        self.assertEqual(detect_tool_profile(state), "gemma")

        state.model_name = "qwen3-coder-next"
        self.assertEqual(detect_tool_profile(state), "native")

        state.model_name = "qwen2.5-7b"
        self.assertEqual(detect_tool_profile(state), "qwen-xml")

        state.model_name = "mistral-nemo"
        self.assertEqual(detect_tool_profile(state), "json")

        state.model_supports_tools = True  # lms ps says native-capable
        self.assertEqual(detect_tool_profile(state), "native")

    def test_tools_schema_uses_openai_legal_names(self):
        mgr = MCPManager.__new__(MCPManager)
        mgr.all_tools = FakeMCP().all_tools
        import re as _re
        for schema in mgr.get_tools_schema():
            self.assertRegex(schema["function"]["name"],
                             _re.compile(r"^[a-zA-Z0-9_-]{1,64}$"))
        name_map = mgr.native_name_map()
        self.assertEqual(
            name_map[make_native_name("shell", "execute_command")],
            "shell/execute_command",
        )

    def test_single_format_prompt_per_profile(self):
        mgr = MCPManager.__new__(MCPManager)
        mgr.all_tools = FakeMCP().all_tools
        gemma_prompt = mgr.get_tools_prompt(profile="gemma")
        self.assertIn("<start_function_call>", gemma_prompt)
        self.assertNotIn("<tool_call>", gemma_prompt)
        qwen_prompt = mgr.get_tools_prompt(profile="qwen-xml")
        self.assertIn("<tool_call>", qwen_prompt)
        self.assertNotIn("<start_function_call>", qwen_prompt)
        native_prompt = mgr.get_tools_prompt(profile="native")
        self.assertNotIn("<tool_call>", native_prompt)
        self.assertNotIn("```json", native_prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
