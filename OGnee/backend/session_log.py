#!/usr/bin/env python3
"""
session_log.py - Session transcripts, continuation scrolls, tool call logs

The paper trail. Everything gets logged so you can audit what your
OGneeys have been up to, and so they can pick up where they left off.

(Named session_log.py — not logging.py — so the module can never shadow
the stdlib `logging` package for absolute imports.)
"""

import atexit
import json
import os
import re
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import (
    SESSION_LOGS,
    CONTINUATION_DIR,
    TOOL_LOG,
    STATE_FILE,
    LOG_TOOL_CALLS,
    LOG_FULL_RESPONSES,
    LOG_TOKEN_USAGE,
)
from .display import print_system, print_success, print_warning, COLORS

# =============================================================================
# SECRET REDACTION
# =============================================================================

# FIX 1: Pre-compiled regex patterns for secret detection and redaction.
# These patterns catch common credential leaks in tool outputs before they
# are persisted to disk.
_OPENAI_KEY_RE = re.compile(r"sk-[a-zA-Z0-9]{48}")
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN\s+(?:OPENSSH\s+|RSA\s+|EC\s+|DSA\s+|PGP\s+)?PRIVATE\s+KEY-----"
    r".*?"
    r"-----END\s+(?:OPENSSH\s+|RSA\s+|EC\s+|DSA\s+|PGP\s+)?PRIVATE\s+KEY-----",
    re.DOTALL,
)
_PASSWORD_RE = re.compile(r"(password\s*[=:]\s*)\S+", re.IGNORECASE)
_API_KEY_RE = re.compile(r"(api_key\s*[=:]\s*)\S+", re.IGNORECASE)
_BEARER_RE = re.compile(r"(Authorization:\s*Bearer\s+)\S+", re.IGNORECASE)
_TOKEN_RE = re.compile(
    r"((?:token|secret|apikey|access_key)\s*[=:]\s*)\S+", re.IGNORECASE
)
_AWS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")


def new_session_id() -> str:
    """Collision-proof session id: timestamp + nanoseconds + short uuid.

    Second-resolution ids collided when two sessions started within the
    same second (overlapping logs, clobbered continuation scrolls).
    """
    return (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{time.time_ns() % 1_000_000_000:09d}"
        f"_{uuid.uuid4().hex[:6]}"
    )


def redact_secrets(text: str) -> str:
    """Scans *text* for common secret patterns and replaces them.

    Covers:
      - OpenAI-style API keys (sk-...)
      - AWS access key ids (AKIA...)
      - PEM / OpenSSH private key blocks
      - password=... / password: ...
      - api_key / token / secret / access_key assignments
      - Authorization: Bearer headers

    IMPORTANT: always redact BEFORE truncating — a secret spanning the
    truncation cut would otherwise leak its prefix.

    Returns a new string with all matches redacted.
    """
    if not text:
        return text

    # 1) Private keys (multi-line blocks)
    text = _PRIVATE_KEY_RE.sub("***PRIVATE_KEY_REDACTED***", text)

    # 2) API keys (OpenAI-style, AWS)
    text = _OPENAI_KEY_RE.sub("sk-***REDACTED***", text)
    text = _AWS_KEY_RE.sub("AKIA***REDACTED***", text)

    # 3) Bearer headers, password / api_key / token / secret assignments
    text = _BEARER_RE.sub(r"\1***REDACTED***", text)
    text = _PASSWORD_RE.sub(r"\1***REDACTED***", text)
    text = _API_KEY_RE.sub(r"\1***REDACTED***", text)
    text = _TOKEN_RE.sub(r"\1***REDACTED***", text)

    return text


# Keys whose VALUES are redacted wholesale when they appear in tool-call args
# (a write_file "content" arg can itself contain a secret, so string values
# also get pattern-redacted via redact_secrets).
_SENSITIVE_KEY_RE = re.compile(
    r"(token|secret|api[_-]?key|access[_-]?key|password|passwd|credential|"
    r"auth|private[_-]?key)", re.IGNORECASE
)


def redact_args(args):
    """Redact secrets from tool-call ARGS before persisting them.

    - Values of keys matching token/secret/key/password/auth patterns are
      replaced wholesale.
    - Other string values are pattern-scanned with redact_secrets (catches
      Bearer strings and key material embedded in e.g. write_file content).
    Recurses into nested dicts/lists.
    """
    if isinstance(args, dict):
        out = {}
        for key, value in args.items():
            if isinstance(key, str) and _SENSITIVE_KEY_RE.search(key):
                out[key] = "***REDACTED***"
            else:
                out[key] = redact_args(value)
        return out
    if isinstance(args, list):
        return [redact_args(v) for v in args]
    if isinstance(args, str):
        return redact_secrets(args)
    return args


# =============================================================================
# LOG ROTATION HELPERS
# =============================================================================

# FIX 4: Size thresholds for automatic log rotation.
_SESSION_LOG_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
_TOOL_LOG_MAX_BYTES = 50 * 1024 * 1024       # 50 MB


def _rotate_if_needed(log_path: Path, max_bytes: int) -> None:
    """Rotate *log_path* to *log_path*.1 when it exceeds *max_bytes*.

    Atomic: ``os.replace`` renames the log onto the backup in one
    operation (the old copy-then-truncate lost the whole log if the
    process crashed between the two steps). Keeps one backup generation.
    """
    if not log_path.exists():
        return

    try:
        if log_path.stat().st_size > max_bytes:
            backup = Path(str(log_path) + ".1")
            os.replace(log_path, backup)  # atomic rename, replaces old .1
            log_path.touch()
    except OSError:
        # Rotation is best-effort.  Do not let a rotation failure break
        # the caller's logging flow.
        pass


# =============================================================================
# SESSION TRANSCRIPT
# =============================================================================

class SessionLogger:
    """Logs everything that happens in a session."""

    def __init__(self, session_id: str, model_name: str):
        self.session_id = session_id
        self.model_name = model_name
        self.start_time = datetime.now()
        # FIX 2 & 3: Removed unbounded in-memory lists (self.entries,
        # self.tool_calls).  The JSONL file is the authoritative log.
        # If session summaries are needed, they are computed from file.
        self.token_usage: List[Dict[str, int]] = []

        # Ensure directories exist
        SESSION_LOGS.mkdir(parents=True, exist_ok=True)
        CONTINUATION_DIR.mkdir(parents=True, exist_ok=True)

        # Log file path
        self.log_path = SESSION_LOGS / f"session_{session_id}.jsonl"
        self.transcript_path = SESSION_LOGS / f"session_{session_id}.md"

        # FIX 4: Rotate logs if they have grown too large from a previous run
        _rotate_if_needed(self.log_path, _SESSION_LOG_MAX_BYTES)
        _rotate_if_needed(self.transcript_path, _SESSION_LOG_MAX_BYTES)

        # Persistent file handles — avoids open/close on every log line
        self._log_fh = open(self.log_path, "a")
        self._transcript_fh = open(self.transcript_path, "a")
        self._closed = False

        # Close handles even if the process dies without a clean shutdown
        # (finalize() also closes; close() is idempotent).
        atexit.register(self.close)

        # Write session header
        self._write_header()

    def _write_header(self):
        """Write session header to transcript."""
        header = f"""# Session {self.session_id}

**Started:** {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}
**Model:** {self.model_name}

---

"""
        self._transcript_fh.write(header)
        self._transcript_fh.flush()

    def _append_jsonl(self, entry: Dict[str, Any]):
        """Append entry to JSONL log."""
        # FIX 1: Redact any text fields that may contain secrets before
        # persisting the entry.
        entry = self._redact_entry(entry)
        line = json.dumps(entry) + "\n"
        if self._closed:
            # Post-finalize stragglers (e.g. atexit-ordered logging) must
            # not crash on a closed handle — fall back to a one-off append.
            with open(self.log_path, "a") as f:
                f.write(line)
            return
        self._log_fh.write(line)
        self._log_fh.flush()

    def _append_transcript(self, text: str):
        """Append text to markdown transcript."""
        # FIX 1: Redact before writing to the persistent transcript.
        safe_text = redact_secrets(text)
        if self._closed:
            with open(self.transcript_path, "a") as f:
                f.write(safe_text)
            return
        self._transcript_fh.write(safe_text)
        self._transcript_fh.flush()

    def close(self):
        """Flush and close persistent file handles. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for fh in (self._log_fh, self._transcript_fh):
            try:
                if fh and not fh.closed:
                    fh.flush()
                    fh.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @staticmethod
    def _redact_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy of *entry* with secret-bearing fields redacted."""
        entry = dict(entry)  # shallow copy
        for key in ("content", "result", "message"):
            if key in entry and isinstance(entry[key], str):
                entry[key] = redact_secrets(entry[key])
        return entry

    def log_user_message(self, message: str):
        """Log a user message."""
        entry = {
            "type": "user",
            "timestamp": datetime.now().isoformat(),
            "content": message
        }
        # FIX 2: Removed ``self.entries.append(entry)`` to prevent
        # unbounded memory growth.  The JSONL file is the source of truth.
        self._append_jsonl(entry)
        self._append_transcript(f"\n## User\n\n{message}\n")

    def log_assistant_message(self, message: str, tokens: int = 0):
        """Log an assistant message."""
        entry = {
            "type": "assistant",
            "timestamp": datetime.now().isoformat(),
            "content": message,
            "tokens": tokens
        }
        # FIX 2: Removed ``self.entries.append(entry)``.
        self._append_jsonl(entry)

        # Truncate for transcript if too long
        display_msg = message if len(message) < 5000 else message[:5000] + "\n\n...(truncated)..."
        self._append_transcript(f"\n## Assistant\n\n{display_msg}\n")

    def log_tool_call(
        self,
        server: str,
        tool: str,
        args: Dict[str, Any],
        result: str,
        success: bool,
        duration_ms: int
    ):
        """Log a tool call."""
        entry = {
            "type": "tool_call",
            "timestamp": datetime.now().isoformat(),
            "server": server,
            "tool": tool,
            "args": redact_args(args),
            # FIX 1: Redact BEFORE truncating — a secret spanning the
            # truncation cut would otherwise leak its prefix. Raw tool output
            # often contains credentials (e.g. ~/.ssh/id_ed25519, /etc/shadow,
            # env files).
            "result": redact_secrets(result)[:1000],
            "success": success,
            "duration_ms": duration_ms
        }
        # FIX 3: Removed ``self.tool_calls.append(entry)`` to prevent
        # unbounded memory growth.
        self._append_jsonl(entry)

        if LOG_TOOL_CALLS:
            status = "✓" if success else "✗"
            # FIX 1: Redact the preview shown in the transcript as well.
            result_preview = redact_secrets(result[:100] + "..." if len(result) > 100 else result)
            self._append_transcript(
                f"\n### Tool: {server}/{tool} {status}\n\n"
                f"```\n{result_preview}\n```\n"
            )

    def log_token_usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        """Log token usage for a turn."""
        entry = {
            "type": "tokens",
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens
        }
        self.token_usage.append(entry)
        if LOG_TOKEN_USAGE:
            self._append_jsonl(entry)

    def log_event(self, event_type: str, details: Dict[str, Any] = None):
        """Log a system event (pause, resume, command, etc)."""
        entry = {
            "type": "event",
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        # FIX 2: Removed ``self.entries.append(entry)``.
        self._append_jsonl(entry)

    def log_gate_decision(self, action: str, decision: str, reason: str):
        """Log a gate/permission decision."""
        entry = {
            "type": "gate",
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "decision": decision,
            "reason": reason
        }
        # FIX 2: Removed ``self.entries.append(entry)``.
        self._append_jsonl(entry)

    # -----------------------------------------------------------------
    # FIX 2 & 3: ``get_summary`` no longer relies on in-memory lists.
    # It reads the JSONL file (the authoritative log) and computes counts
    # from the persisted entries.
    # -----------------------------------------------------------------
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary stats from the JSONL file."""
        end_time = datetime.now()
        duration = end_time - self.start_time

        total_tokens = sum(t.get("total", 0) for t in self.token_usage)

        tool_count = 0
        tool_failures = 0
        message_count = 0

        if self.log_path.exists():
            try:
                with open(self.log_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        rtype = rec.get("type")
                        if rtype == "tool_call":
                            tool_count += 1
                            if not rec.get("success", True):
                                tool_failures += 1
                        elif rtype in ("user", "assistant"):
                            message_count += 1
            except OSError:
                pass  # Best-effort read from file.

        return {
            "session_id": self.session_id,
            "model": self.model_name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": int(duration.total_seconds()),
            "total_tokens": total_tokens,
            "tool_calls": tool_count,
            "tool_failures": tool_failures,
            "message_count": message_count,
        }

    def finalize(self, reason: str = "normal", summary_text: str = ""):
        """Finalize the session log with summary."""
        stats = self.get_summary()

        footer = f"""
---

## Session Summary

**Ended:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Reason:** {reason}
**Duration:** {stats['duration_seconds'] // 60}m {stats['duration_seconds'] % 60}s
**Tokens Used:** {stats['total_tokens']:,}
**Tool Calls:** {stats['tool_calls']} ({stats['tool_failures']} failures)
**Messages:** {stats['message_count']}

{summary_text}
"""
        self._append_transcript(footer)

        # Write final summary to JSONL
        self._append_jsonl({
            "type": "session_end",
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "stats": stats
        })

        print_success(f"  📝 Session logged: {self.transcript_path.name}")
        self.close()
        return self.transcript_path


# =============================================================================
# TOOL CALL LOG (APPEND-ONLY GLOBAL LOG)
# =============================================================================

def log_tool_to_global(
    session_id: str,
    server: str,
    tool: str,
    args: Dict[str, Any],
    result: str,
    success: bool,
    duration_ms: int
):
    """Append tool call to the global tool log."""
    TOOL_LOG.parent.mkdir(parents=True, exist_ok=True)

    # FIX 4: Rotate the global tool log if it has grown beyond the limit.
    _rotate_if_needed(TOOL_LOG, _TOOL_LOG_MAX_BYTES)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "session": session_id,
        "server": server,
        "tool": tool,
        "args": redact_args(args),
        # FIX 1: Redact (before truncating) secrets from the global tool log.
        "result": redact_secrets(result)[:500],
        "success": success,
        "duration_ms": duration_ms
    }

    with open(TOOL_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

def get_recent_tool_calls(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent tool calls from global log.

    Reads only the tail (deque with maxlen) — the log can be tens of MB.
    """
    if not TOOL_LOG.exists():
        return []

    calls = []
    with open(TOOL_LOG, "r") as f:
        for line in deque(f, maxlen=limit):
            try:
                calls.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    return calls

# =============================================================================
# CONTINUATION SCROLL MANAGEMENT
# =============================================================================

def save_continuation(
    session_id: str,
    model_name: str,
    elapsed_seconds: int,
    tokens_used: int,
    context_limit: int,
    conversation_summary: str,
    recent_tools: List[Dict[str, Any]],
    failed_tools: Dict[str, int],
    active_tasks: List[str] = None,
    notes: str = ""
) -> Path:
    """Save a continuation scroll for session handoff."""

    CONTINUATION_DIR.mkdir(parents=True, exist_ok=True)

    # Format tool summaries
    tool_lines = []
    for tc in recent_tools[-10:]:
        status = "✓" if tc.get("success", True) else "✗"
        tool_lines.append(f"  {status} {tc.get('server', '?')}/{tc.get('tool', '?')}")
    tool_summary = "\n".join(tool_lines) if tool_lines else "  (none)"

    failed_lines = [f"  - {k}: {v} failures" for k, v in failed_tools.items() if v >= 2]
    failed_summary = "\n".join(failed_lines) if failed_lines else "  (none)"

    tasks_summary = "\n".join(f"  - {t}" for t in (active_tasks or [])) or "  (none)"

    scroll = f"""# Continuation Scroll
## Session {session_id}

**Handoff Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Previous Model:** {model_name}
**Session Duration:** {elapsed_seconds // 3600:02d}:{(elapsed_seconds % 3600) // 60:02d}:{elapsed_seconds % 60:02d}
**Tokens Used:** {tokens_used:,} / {context_limit:,}
**Reason for Handoff:** Context limit approaching

---

## Conversation Summary

{conversation_summary if conversation_summary else "(No summary provided)"}

---

## Active Tasks

{tasks_summary}

---

## Recent Tool Activity

{tool_summary}

## Tools to Avoid (Multiple Failures)

{failed_summary}

---

## Notes

{notes if notes else "(none)"}

---

## Instructions for Next OGneey

Pick up where the previous session left off. Review the conversation summary
above and continue assisting the user. The tool failure information should
help you avoid repeating unsuccessful approaches.

If the user's task was incomplete, summarize what remains and propose next steps.
"""

    filepath = CONTINUATION_DIR / f"continuation_{session_id}.md"
    filepath.write_text(scroll)

    print_success(f"  📜 Continuation scroll saved: {filepath.name}")
    return filepath

def get_latest_continuation() -> Optional[Path]:
    """Get the most recent continuation scroll."""
    if not CONTINUATION_DIR.exists():
        return None

    scrolls = sorted(CONTINUATION_DIR.glob("continuation_*.md"), reverse=True)
    return scrolls[0] if scrolls else None

def load_continuation(filepath: Path) -> Optional[str]:
    """Load a continuation scroll."""
    if filepath and filepath.exists():
        return filepath.read_text()
    return None

def list_continuations(limit: int = 10) -> List[Dict[str, Any]]:
    """List recent continuation scrolls."""
    if not CONTINUATION_DIR.exists():
        return []

    scrolls = sorted(CONTINUATION_DIR.glob("continuation_*.md"), reverse=True)[:limit]

    results = []
    for scroll in scrolls:
        # Extract session ID from filename
        session_id = scroll.stem.replace("continuation_", "")
        results.append({
            "path": scroll,
            "session_id": session_id,
            "modified": datetime.fromtimestamp(scroll.stat().st_mtime),
            "size": scroll.stat().st_size
        })

    return results

# =============================================================================
# STATE FILE MANAGEMENT (THE PERSISTENT BRAIN)
# =============================================================================

def load_state() -> Optional[str]:
    """Load the persistent state file."""
    if STATE_FILE.exists():
        return STATE_FILE.read_text()
    return None

def save_state(
    current_context: str = "",
    active_tasks: List[str] = None,
    notes: List[str] = None,
    recent_sessions: List[Dict[str, Any]] = None
):
    """Save/update the persistent state file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    tasks = "\n".join(f"  - {t}" for t in (active_tasks or [])) or "  (none)"
    notes_str = "\n".join(f"  - {n}" for n in (notes or [])) or "  (none)"

    sessions_str = ""
    for s in (recent_sessions or [])[-5:]:
        sessions_str += f"- **{s.get('session_id', '?')}**: {s.get('model', '?')}, {s.get('duration', '?')}, {s.get('tokens', 0):,} tokens\n"
    sessions_str = sessions_str or "  (none)\n"

    content = f"""# OGnee State
*Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Current Context

{current_context if current_context else "(No active context)"}

## Active Tasks

{tasks}

## Notes

{notes_str}

## Recent Sessions

{sessions_str}
"""

    STATE_FILE.write_text(content)
    print_system(f"  💾 State saved: {STATE_FILE.name}")

def append_to_state(section: str, content: str):
    """Append content to a section of the state file."""
    if not STATE_FILE.exists():
        save_state()

    current = STATE_FILE.read_text()

    # Use an anchored regex for exact section-name matching.
    section_pattern = re.compile(rf"^(##\s+{re.escape(section)})$", re.MULTILINE)

    if section_pattern.search(current):
        # Insert after the section header
        parts = current.split(f"## {section}")
        if len(parts) == 2:
            # Find next section or end
            next_section = parts[1].find("\n## ")
            if next_section == -1:
                # Append at end of section
                new_content = parts[0] + f"## {section}" + parts[1].rstrip() + f"\n  - {content}\n"
            else:
                # Insert before next section
                new_content = (
                    parts[0] + f"## {section}" +
                    parts[1][:next_section].rstrip() +
                    f"\n  - {content}\n" +
                    parts[1][next_section:]
                )
            STATE_FILE.write_text(new_content)


# =============================================================================
# QUICK HELPERS
# =============================================================================

def get_session_log_path(session_id: str) -> Path:
    """Get path to a session's transcript."""
    return SESSION_LOGS / f"session_{session_id}.md"

def list_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    """List recent sessions."""
    if not SESSION_LOGS.exists():
        return []

    logs = sorted(SESSION_LOGS.glob("session_*.md"), reverse=True)[:limit]

    results = []
    for log in logs:
        session_id = log.stem.replace("session_", "")
        results.append({
            "path": log,
            "session_id": session_id,
            "modified": datetime.fromtimestamp(log.stat().st_mtime),
            "size": log.stat().st_size
        })

    return results

def cleanup_old_logs(days: int = 30):
    """Remove logs older than N days."""
    from datetime import timedelta

    cutoff = datetime.now() - timedelta(days=days)
    removed = 0

    for log_dir in [SESSION_LOGS, CONTINUATION_DIR]:
        if not log_dir.exists():
            continue

        for f in log_dir.iterdir():
            if f.is_file():
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    f.unlink()
                    removed += 1

    if removed:
        print_system(f"  🧹 Cleaned up {removed} old log files")

    return removed
