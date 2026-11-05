#!/usr/bin/env python3
"""
tests/test_wave2.py - Wave 2 hardening tests (stdlib unittest).

Covers:
- Shadow path mapping: idempotence (writing to an already-shadow path must
  NOT double the prefix), under-root vs outside-root mapping, and the
  inverse mapping used by /diff and /apply.
- Path ACLs: Path.is_relative_to semantics (/tmp must not match /tmpevil),
  tilde expansion.
- Dotfile backup naming (.bashrc -> .bashrc.bak, not a with_suffix crash).
- clear_shadow honoring its paths argument.
- Gates: writes already under SHADOW_ROOT are early-ALLOWed (no re-redirect).
- Session state machine: single SessionState enum, transition sync of the
  legacy flags, /pause /resume /step /execute never fall through to the LLM,
  messages typed while paused are queued and surfaced (never swallowed).
- Session logging: collision-proof session ids, atomic rotation, fd close.

Run from the repo root:  python3 tests/test_wave2.py
"""

import sys
import types
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Package bootstrap (same trick as test_wave1): register this directory as
# the `backend` package without putting it on sys.path.
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parents[1]
if "backend" not in sys.modules:
    _pkg = types.ModuleType("backend")
    _pkg.__path__ = [str(PKG_DIR)]
    sys.modules["backend"] = _pkg

from backend.config import (  # noqa: E402
    HOME,
    OGNEE_ROOT,
    SHADOW_ROOT,
    get_shadow_path,
    get_real_path,
    is_path_writable,
    is_path_read_only,
)
from backend.context import (  # noqa: E402
    SessionState,
    create_session,
    transition_state,
    resume_target,
    queue_user_message,
    drain_queued_messages,
)
from backend.commands import execute_command, CommandResult  # noqa: E402
from backend.gates import gate_file_write, GateResult  # noqa: E402
from backend import shadow  # noqa: E402
from backend.session_log import _rotate_if_needed, new_session_id  # noqa: E402
from backend.tools import queue_tool_call  # noqa: E402


def _session(tier: str = "supervised"):
    return create_session(model_name="test-model", trust_tier=tier)


class TestShadowPathMapping(unittest.TestCase):
    def test_under_root_mapping(self):
        p = OGNEE_ROOT / "foo" / "bar.py"
        self.assertEqual(get_shadow_path(p), SHADOW_ROOT / "foo" / "bar.py")

    def test_outside_root_mapping(self):
        self.assertEqual(
            get_shadow_path("/etc/foo"),
            SHADOW_ROOT / "outside" / "etc" / "foo",
        )

    def test_idempotent_on_shadow_paths(self):
        sp = get_shadow_path("/etc/foo")
        self.assertEqual(get_shadow_path(sp), sp)
        self.assertEqual(get_shadow_path(str(sp)), sp)

    def test_no_doubling_regression(self):
        # The reported bug: re-gating a staged shadow path produced
        # SHADOW_ROOT/<full shadow path> nested inside itself.
        sp = get_shadow_path("/etc/foo")
        doubled = get_shadow_path(sp)
        self.assertNotIn(str(SHADOW_ROOT), str(doubled.relative_to(SHADOW_ROOT)))

    def test_tilde_expansion(self):
        sp = get_shadow_path("~/x")
        self.assertNotIn("~", str(sp))
        self.assertTrue(sp.is_relative_to(SHADOW_ROOT))

    def test_inverse_mapping_roundtrip(self):
        for real in ("/etc/foo", str(OGNEE_ROOT / "a" / "b.py"), "/tmp/scratch.txt"):
            self.assertEqual(get_real_path(get_shadow_path(real)), Path(real).resolve())

    def test_inverse_of_non_shadow_is_identity(self):
        self.assertEqual(get_real_path("/etc/foo"), Path("/etc/foo"))


class TestPathACLs(unittest.TestCase):
    def test_tmp_does_not_match_tmpevil(self):
        self.assertTrue(is_path_writable("/tmp/somefile"))
        self.assertFalse(is_path_writable("/tmpevil/somefile"))

    def test_home_prefix_does_not_match_sibling(self):
        # str(HOME) matched "/home/tower2" under startswith; is_relative_to
        # must not.
        sibling = str(HOME) + "evil" + "/file"
        self.assertFalse(is_path_read_only(sibling))
        self.assertTrue(is_path_read_only(str(HOME / "file")))

    def test_tilde_matches_home_read_only(self):
        self.assertTrue(is_path_read_only("~/somefile"))


class TestShadowFileOps(unittest.TestCase):
    """Real filesystem round-trips inside the shadow tree; cleaned up."""

    def setUp(self):
        self.real = OGNEE_ROOT / "OGneeys" / "scratch" / "wave2-test" / ".bashrc"
        self.real.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for p in (self.real, self.real.with_name(self.real.name + ".bak")):
            p.unlink(missing_ok=True)
        shadow.clear_shadow(paths=[str(self.real)])
        try:
            self.real.parent.rmdir()
        except OSError:
            pass

    def test_dotfile_backup_naming(self):
        self.real.write_text("original")
        ok, _ = shadow.shadow_write(str(self.real), "modified")
        self.assertTrue(ok)
        ok, msg = shadow.apply_shadow_change(str(self.real))
        self.assertTrue(ok, msg)
        backup = self.real.with_name(self.real.name + ".bak")
        self.assertTrue(backup.exists(), "backup must be .bashrc.bak")
        self.assertEqual(backup.read_text(), "original")
        self.assertEqual(self.real.read_text(), "modified")

    def test_apply_removes_shadow_and_lists_inverse(self):
        shadow.shadow_write(str(self.real), "staged")
        changes = shadow.list_shadow_changes()
        entry = [c for c in changes if c["real_path"] == str(self.real)]
        self.assertEqual(len(entry), 1, "list_shadow_changes must invert the mapping")
        results = shadow.apply_shadow_changes(paths=[str(self.real)])
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0][1], results[0][2])
        self.assertFalse(shadow.has_shadow_version(str(self.real)))

    def test_clear_shadow_honors_paths(self):
        other = "/etc/wave2-other"
        shadow.shadow_write(str(self.real), "a")
        shadow.shadow_write(other, "b")
        removed = shadow.clear_shadow(paths=[str(self.real)])
        self.assertEqual(removed, 1)
        self.assertFalse(shadow.has_shadow_version(str(self.real)))
        self.assertTrue(shadow.has_shadow_version(other))
        self.assertEqual(shadow.clear_shadow(paths=[other]), 1)


class TestGateShadowEarlyAllow(unittest.TestCase):
    def test_shadow_path_write_is_allowed_not_redirected(self):
        state = _session("supervised")  # can_write_shadow, not direct
        shadow_target = get_shadow_path("/etc/foo")
        decision = gate_file_write(str(shadow_target), state)
        self.assertIs(decision.result, GateResult.ALLOW)
        self.assertIsNone(decision.redirect_path)


class TestSessionStateMachine(unittest.TestCase):
    def test_transition_syncs_legacy_flags(self):
        state = _session()
        self.assertIs(state.mode, SessionState.RUNNING)
        transition_state(state, SessionState.PAUSED)
        self.assertTrue(state.paused)
        self.assertFalse(state.plan_mode)
        transition_state(state, SessionState.PLAN_REVIEW)
        self.assertFalse(state.paused)
        self.assertTrue(state.plan_mode)
        transition_state(state, SessionState.RUNNING)
        self.assertFalse(state.paused)
        self.assertFalse(state.plan_mode)

    def test_pause_queues_messages_and_resume_surfaces(self):
        state = _session()
        resp = execute_command("/pause", state)
        self.assertIsNot(resp.result, CommandResult.CONTINUE)  # no fallthrough
        self.assertIs(state.mode, SessionState.PAUSED)

        # User types a message while paused — queued, never swallowed.
        depth = queue_user_message(state, "pivot: it's better if we refactor")
        self.assertEqual(depth, 1)

        resp = execute_command("/resume", state)
        self.assertIs(resp.result, CommandResult.HANDLED)  # never sent to LLM
        self.assertEqual(resp.data, {"resume": True})
        self.assertIs(state.mode, SessionState.RUNNING)

        queued = drain_queued_messages(state)
        self.assertEqual(queued, ["pivot: it's better if we refactor"])
        self.assertEqual(state.queued_messages, [])

    def test_resume_when_not_paused(self):
        state = _session()
        resp = execute_command("/resume", state)
        self.assertIs(resp.result, CommandResult.HANDLED)
        self.assertIsNone(resp.data)

    def test_step_never_falls_through(self):
        state = _session()
        resp = execute_command("/step", state)
        self.assertIs(resp.result, CommandResult.HANDLED)
        self.assertTrue(state.step_mode)
        self.assertIsNone(resp.data)

    def test_plan_execute_no_fallthrough(self):
        state = _session()
        resp = execute_command("/plan", state)
        self.assertIs(resp.result, CommandResult.HANDLED)
        self.assertIs(state.mode, SessionState.PLAN_REVIEW)

        # Empty queue: /execute exits plan mode, no task sent to the LLM
        resp = execute_command("/execute", state)
        self.assertIs(resp.result, CommandResult.HANDLED)
        self.assertIsNone(resp.data)
        self.assertIs(state.mode, SessionState.RUNNING)

        # With a queued call: /execute hands the plan to the loop as data
        execute_command("/plan", state)
        queue_tool_call(state, "local", "write_file", {"path": "/tmp/x"})
        resp = execute_command("/execute", state)
        self.assertIs(resp.result, CommandResult.HANDLED)
        self.assertIn("execute_plan", resp.data)
        self.assertEqual(resp.data["execute_plan"][0]["tool"], "write_file")
        self.assertIs(state.mode, SessionState.RUNNING)

    def test_execute_outside_plan_mode(self):
        state = _session()
        resp = execute_command("/execute", state)
        self.assertIs(resp.result, CommandResult.HANDLED)
        self.assertIsNone(resp.data)

    def test_resume_returns_to_plan_review_with_live_queue(self):
        state = _session()
        execute_command("/plan", state)
        queue_tool_call(state, "local", "write_file", {"path": "/tmp/y"})
        execute_command("/pause", state)
        self.assertIs(state.mode, SessionState.PAUSED)
        self.assertIs(resume_target(state), SessionState.PLAN_REVIEW)
        execute_command("/resume", state)
        self.assertIs(state.mode, SessionState.PLAN_REVIEW)
        self.assertTrue(state.plan_mode)

    def test_clear_exits_plan_review(self):
        state = _session()
        execute_command("/plan", state)
        queue_tool_call(state, "local", "write_file", {"path": "/tmp/z"})
        resp = execute_command("/clear", state)
        self.assertIs(resp.result, CommandResult.HANDLED)
        self.assertIs(state.mode, SessionState.RUNNING)
        self.assertEqual(state.pending_plan, [])


class TestSessionLogging(unittest.TestCase):
    def test_session_ids_unique(self):
        ids = {new_session_id() for _ in range(1000)}
        self.assertEqual(len(ids), 1000)

    def test_atomic_rotation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "test.log"
            log.write_text("x" * 2048)
            _rotate_if_needed(log, max_bytes=1024)
            backup = Path(str(log) + ".1")
            self.assertTrue(backup.exists())
            self.assertEqual(backup.read_text(), "x" * 2048)
            self.assertEqual(log.read_text(), "")

    def test_logger_close_is_idempotent_and_writes_survive(self):
        from backend.session_log import SessionLogger
        logger = SessionLogger(new_session_id(), "test-model")
        log_path, transcript_path = logger.log_path, logger.transcript_path
        try:
            logger.log_event("test", {"k": "v"})
            logger.close()
            logger.close()  # idempotent — must not raise
            self.assertTrue(logger._log_fh.closed)
            # Post-close writes fall back to one-off append (no crash).
            logger.log_event("post_close", {})
            self.assertIn("post_close", log_path.read_text())
        finally:
            log_path.unlink(missing_ok=True)
            transcript_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
