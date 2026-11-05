#!/usr/bin/env python3
"""
tests/test_regenerate.py - /regenerate (stdlib unittest).

Covers:
- Every reply shown in the REPL gets a number bound to its assistant message,
  in chat-only mode and in agentic mode; one-shot mode numbers nothing.
- plan_regenerate: latest by default, any earlier reply by number, unknown
  and trimmed-out replies refused, confirmation only when later turns or tool
  calls would be dropped.
- apply_regenerate rewinds to just before the reply and forgets the numbers
  it dropped; agent_loop(task=None) then re-asks without adding a user turn.
- /regenerate argument parsing and the busy guard.

Run from the repo root:  python3 tests/test_regenerate.py
"""

import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

PKG_DIR = Path(__file__).resolve().parents[1]
if "backend" not in sys.modules:
    _pkg = types.ModuleType("backend")
    _pkg.__path__ = [str(PKG_DIR)]
    sys.modules["backend"] = _pkg

from backend import agent  # noqa: E402
from backend.agent import (  # noqa: E402
    agent_loop,
    apply_regenerate,
    plan_regenerate,
)
from backend.commands import BUSY_REFUSED, execute_command  # noqa: E402
from backend.context import (  # noqa: E402
    SessionState,
    bind_response,
    create_session,
    latest_response_id,
    reserve_response_id,
    transition_state,
)


class _Log:
    """SessionLogger stand-in that records events."""

    def __init__(self):
        self.events = []
        self.user = []

    def log_user_message(self, text):
        self.user.append(text)

    def log_event(self, kind, details=None):
        self.events.append((kind, details or {}))

    def __getattr__(self, name):          # log_assistant_message, log_reasoning, ...
        return lambda *a, **k: None


class _MCP:
    all_tools = []
    servers = {}

    def get_tools_prompt(self, **kwargs):
        return ""


def _reply(text, **extra):
    msg = {"role": "assistant", "content": text}
    msg.update(extra)
    return {"choices": [{"message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def _fake_chat(replies):
    queue = list(replies)
    seen = []

    async def chat(messages, tools=None):
        seen.append([dict(m) for m in messages])
        return queue.pop(0)
    chat.seen = seen
    return chat


def _session(numbered=True, tools=False):
    state = create_session(model_name="test-model", trust_tier="supervised")
    state.number_responses = numbered
    state.tools_enabled = tools
    return state


def _run(coro):
    return asyncio.run(coro)


def _conversation(state):
    """system, then three exchanges; reply #2 ran a tool (native)."""
    msgs = [{"role": "system", "content": "sys"}]
    for i, (user, reply) in enumerate([("hi", "hello"), ("list /tmp", None), ("thanks", "np")], 1):
        msgs.append({"role": "user", "content": user})
        if reply is None:
            call = {"role": "assistant", "content": "looking",
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "local__list_dir", "arguments": "{}"}}]}
            msgs.append(call)
            bind_response(state, reserve_response_id(state), call)
            msgs.append({"role": "tool", "tool_call_id": "c1", "content": "a b c"})
            final = {"role": "assistant", "content": "found a b c"}
            msgs.append(final)
            bind_response(state, reserve_response_id(state), final)
        else:
            m = {"role": "assistant", "content": reply}
            msgs.append(m)
            bind_response(state, reserve_response_id(state), m)
    return msgs


class TestNumbering(unittest.TestCase):
    def test_chat_only_reply_is_numbered(self):
        state, log = _session(tools=False), _Log()
        msgs = [{"role": "system", "content": "sys"}]
        with mock.patch.object(agent, "chat", _fake_chat([_reply("one"), _reply("two")])):
            _run(agent_loop("a", _MCP(), state, log, "sys", msgs))
            _run(agent_loop("b", _MCP(), state, log, "sys", msgs))
        self.assertEqual(sorted(state.response_marks), [1, 2])
        self.assertIs(state.response_marks[2], msgs[-1])
        self.assertEqual(msgs[-1]["content"], "two")

    def test_agentic_reply_is_numbered(self):
        state, log = _session(tools=True), _Log()
        msgs = [{"role": "system", "content": "sys"}]
        with mock.patch.object(agent, "chat", _fake_chat([_reply("done")])):
            _run(agent_loop("go", _MCP(), state, log, "sys", msgs))
        self.assertEqual(list(state.response_marks), [1])
        self.assertIs(state.response_marks[1], msgs[-1])

    def test_one_shot_numbers_nothing(self):
        state = _session(numbered=False)
        self.assertIsNone(reserve_response_id(state))
        bind_response(state, None, {"role": "assistant", "content": "x"})
        self.assertEqual(state.response_marks, {})

    def test_footer_printed(self):
        import contextlib
        import io
        from backend import display
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            display.print_ognee_response("hey", 7)
            display.print_ognee_response("no number")
        self.assertIn("/regen 7", out.getvalue())
        self.assertEqual(out.getvalue().count("/regen"), 1)


class TestPlan(unittest.TestCase):
    def test_latest_needs_no_confirm(self):
        state = _session()
        msgs = _conversation(state)
        plan, why = plan_regenerate(msgs, state)
        self.assertEqual(why, "")
        self.assertEqual(plan.response_id, 4)
        self.assertEqual(plan.index, len(msgs) - 1)
        self.assertTrue(plan.is_latest)
        self.assertFalse(plan.needs_confirm)

    def test_earlier_reply_counts_what_is_dropped(self):
        state = _session()
        msgs = _conversation(state)
        plan, _ = plan_regenerate(msgs, state, 1)
        self.assertEqual(plan.later_replies, [2, 3, 4])
        self.assertEqual(plan.operator_turns, 2)
        self.assertEqual(plan.tool_calls_ran, 1)
        self.assertTrue(plan.needs_confirm)

    def test_reply_that_called_tools_needs_confirm(self):
        state = _session()
        msgs = _conversation(state)
        plan, _ = plan_regenerate(msgs, state, 3)     # final answer after the tool
        self.assertEqual(plan.tool_calls_ran, 0)
        self.assertEqual(plan.later_replies, [4])
        plan2, _ = plan_regenerate(msgs, state, 2)    # the tool-calling reply itself
        self.assertEqual(plan2.tool_calls_ran, 1)

    def test_text_mode_tool_results_counted(self):
        state = _session()
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
        a = {"role": "assistant", "content": "Using tools..."}
        msgs.append(a)
        bind_response(state, reserve_response_id(state), a)
        msgs.append({"role": "user", "content": "[TOOL RESULTS]\n\n=== x/y (success) ===\nok\n\n"
                                                "=== x/z (success) ===\nok\n[/TOOL RESULTS]"})
        plan, _ = plan_regenerate(msgs, state, 1)
        self.assertEqual(plan.tool_calls_ran, 2)
        self.assertEqual(plan.operator_turns, 0)

    def test_unknown_and_empty(self):
        state = _session()
        plan, why = plan_regenerate([{"role": "system", "content": "s"}], state)
        self.assertIsNone(plan)
        self.assertIn("No OGneey replies", why)
        msgs = _conversation(state)
        plan, why = plan_regenerate(msgs, state, 99)
        self.assertIsNone(plan)
        self.assertIn("#99", why)

    def test_trimmed_reply_refused(self):
        state = _session()
        msgs = _conversation(state)
        target = state.response_marks[1]
        msgs[:] = [m for m in msgs if m is not target]
        plan, why = plan_regenerate(msgs, state, 1)
        self.assertIsNone(plan)
        self.assertIn("trimmed", why)


class TestApplyAndRerun(unittest.TestCase):
    def test_rewind_then_rerun_replaces_reply(self):
        state, log = _session(tools=False), _Log()
        msgs = _conversation(state)
        plan, _ = plan_regenerate(msgs, state, 3)
        apply_regenerate(msgs, state, plan, log)
        self.assertEqual(msgs[-1]["role"], "tool")          # just before reply #3
        self.assertEqual(sorted(state.response_marks), [1, 2])
        self.assertEqual(log.events[-1][0], "regenerate")
        self.assertEqual(log.events[-1][1]["replaced_text"], "found a b c")

        fake = _fake_chat([_reply("found a, b and c")])
        with mock.patch.object(agent, "chat", fake):
            _run(agent_loop(None, _MCP(), state, log, "sys", msgs, new_task=False))
        # the model saw the history ending at the tool result — no new user turn
        self.assertEqual(fake.seen[0][-1]["role"], "tool")
        self.assertEqual(log.user, [])
        self.assertEqual(msgs[-1]["content"], "found a, b and c")
        self.assertEqual(latest_response_id(state), 5)       # numbers never repeat
        self.assertIs(state.response_marks[5], msgs[-1])

    def test_regenerate_latest_twice(self):
        state, log = _session(tools=False), _Log()
        msgs = [{"role": "system", "content": "sys"}]
        with mock.patch.object(agent, "chat", _fake_chat([_reply("v1"), _reply("v2"), _reply("v3")])):
            _run(agent_loop("q", _MCP(), state, log, "sys", msgs))
            for _ in range(2):
                plan, _ = plan_regenerate(msgs, state)
                self.assertFalse(plan.needs_confirm)
                apply_regenerate(msgs, state, plan, log)
                _run(agent_loop(None, _MCP(), state, log, "sys", msgs, new_task=False))
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])
        self.assertEqual(msgs[-1]["content"], "v3")

    def test_task_none_without_history_rejected(self):
        state = _session()
        with self.assertRaises(ValueError):
            _run(agent_loop(None, _MCP(), state, _Log(), "sys", None))


class _PlanMCP:
    """One write-capable MCP tool. In plan mode it is only ever queued."""
    all_tools = [{"server": "files", "name": "write_file", "full_name": "files/write_file",
                  "description": "write a file",
                  "inputSchema": {"type": "object", "properties": {"path": {"type": "string"},
                                                                    "content": {"type": "string"}}}}]
    servers = {}

    def get_tools_prompt(self, **kwargs):
        return ""


def _write_call_text(path):
    return _reply('{"tool": "files/write_file", "arguments": {"path": "%s", "content": "x"}}' % path)


def _write_call_native(path):
    call = {"id": "call_" + path.strip("/").replace("/", "_"), "type": "function",
            "function": {"name": "files__write_file",
                         "arguments": '{"path": "%s", "content": "x"}' % path}}
    return _reply("", tool_calls=[call])


class TestPlanMode(unittest.TestCase):
    """The case regenerate is mostly for: a dumb plan."""

    def _plan_session(self, native=False):
        state = _session(tools=True)
        state.model_supports_tools = native
        transition_state(state, SessionState.PLAN_REVIEW)
        return state

    def _run_plan_turn(self, native):
        state, log = self._plan_session(native), _Log()
        msgs = [{"role": "system", "content": "sys"}]
        call = _write_call_native if native else _write_call_text
        fake = _fake_chat([call("/tmp/dumb"), _reply("Plan: write /tmp/dumb"),
                           call("/tmp/good"), _reply("Plan: write /tmp/good")])
        with mock.patch.object(agent, "chat", fake):
            _run(agent_loop("make a plan", _PlanMCP(), state, log, "sys", msgs))
            self.assertEqual([s["args"]["path"] for s in state.pending_plan], ["/tmp/dumb"])
            plan, why = plan_regenerate(msgs, state)
            self.assertEqual(why, "")
            self.assertTrue(plan.whole_turn)
            self.assertEqual(plan.plan_steps, 1)
            self.assertEqual(plan.tool_calls_ran, 0)      # queued, never ran
            self.assertFalse(plan.needs_confirm)          # instant, like any AI tool
            apply_regenerate(msgs, state, plan, log)
            self.assertEqual(state.pending_plan, [])
            self.assertEqual(msgs[-1]["content"], "make a plan")   # back to your message
            _run(agent_loop(None, _PlanMCP(), state, log, "sys", msgs, new_task=False))
        # exactly one plan queued — the new one
        self.assertEqual([s["args"]["path"] for s in state.pending_plan], ["/tmp/good"])
        self.assertTrue(state.plan_mode)
        self.assertEqual(msgs[-1]["content"], "Plan: write /tmp/good")

    def test_text_profile_plan_redo_replaces_queue(self):
        self._run_plan_turn(native=False)

    def test_native_profile_plan_redo_replaces_queue(self):
        self._run_plan_turn(native=True)

    def test_explicit_number_redoes_only_that_reply(self):
        state, log = self._plan_session(), _Log()
        msgs = [{"role": "system", "content": "sys"}]
        with mock.patch.object(agent, "chat", _fake_chat([_write_call_text("/tmp/a"),
                                                          _reply("Plan: write /tmp/a")])):
            _run(agent_loop("plan it", _PlanMCP(), state, log, "sys", msgs))
        rid = latest_response_id(state)
        plan, _ = plan_regenerate(msgs, state, rid)
        self.assertFalse(plan.whole_turn)
        self.assertEqual(plan.plan_steps, 0)              # the step belongs to the earlier reply
        apply_regenerate(msgs, state, plan, log)
        self.assertEqual(len(state.pending_plan), 1)

    def test_bare_regen_outside_plan_mode_is_latest_reply_only(self):
        state = _session()
        msgs = _conversation(state)
        plan, _ = plan_regenerate(msgs, state)
        self.assertFalse(plan.whole_turn)
        self.assertEqual(plan.index, len(msgs) - 1)

    def test_paused_out_of_plan_mode_counts_as_plan(self):
        state, log = self._plan_session(), _Log()
        msgs = [{"role": "system", "content": "sys"}]
        with mock.patch.object(agent, "chat", _fake_chat([_write_call_text("/tmp/a"),
                                                          _reply("Plan: write /tmp/a")])):
            _run(agent_loop("plan it", _PlanMCP(), state, log, "sys", msgs))
        transition_state(state, SessionState.PAUSED)
        plan, _ = plan_regenerate(msgs, state)
        self.assertTrue(plan.whole_turn)


class TestSideEffectCounting(unittest.TestCase):
    def test_reads_and_denials_dont_count(self):
        from backend.agent import _count_side_effect_calls
        span = [
            {"role": "tool", "name": "files/read_file", "content": "data"},
            {"role": "tool", "name": "shell/execute_command", "content": "[DENIED] no"},
            {"role": "tool", "name": "files/write_file", "content": "[PLAN MODE] Queued for review — NOT executed yet."},
            {"role": "tool", "name": "shell/execute_command", "content": "ok"},
            {"role": "user", "content": "[TOOL RESULTS]\n\n=== files/read_file (success) ===\nx\n\n"
                                        "=== shell/execute_command (FAILED) ===\nNot executed — paused\n\n"
                                        "=== shell/execute_command (success) ===\ndone\n[/TOOL RESULTS]"},
        ]
        self.assertEqual(_count_side_effect_calls(span), 2)


class TestCommand(unittest.TestCase):
    def test_parsing(self):
        state = _session()
        with mock.patch("builtins.print"):
            self.assertEqual(execute_command("/regenerate", state).data, {"regenerate": None})
            self.assertEqual(execute_command("/regen 3", state).data, {"regenerate": 3})
            self.assertEqual(execute_command("/retry #2", state).data, {"regenerate": 2})
            self.assertIsNone(execute_command("/regen three", state).data)
            self.assertIsNone(execute_command("/regen 1 2", state).data)

    def test_refused_while_busy(self):
        self.assertIn("regenerate", BUSY_REFUSED)


if __name__ == "__main__":
    unittest.main()
