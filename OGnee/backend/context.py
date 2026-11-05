#!/usr/bin/env python3
"""
context.py - Token tracking, session state, hard stops

The brain's awareness of its own limits. Tracks token usage,
injects state into prompts, triggers hard stops when context runs low.
"""

import os
import re
import time
from datetime import datetime
from enum import Enum, auto

# tiktoken is optional: when it is unavailable (minimal/Nix environments)
# count_tokens() falls back to a chars-per-token estimate.
try:
    import tiktoken
except ImportError:  # pragma: no cover - depends on environment
    tiktoken = None
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from pathlib import Path

from .config import (
    get_model_config, 
    ModelConfig,
    STATE_INJECTION_INTERVAL,
    CONTINUATION_SUMMARY_TOKENS,
    STATE_FILE,
    CONTINUATION_DIR,
    is_path_excluded,
    is_path_read_only,
    is_path_writable,
    can_access_path,
    get_shadow_path,
)
from .display import (
    print_hard_stop,
    print_context_warning,
    print_system,
    print_warning,
    SessionStatus,
)

# =============================================================================
# TOKEN COUNTING
# =============================================================================

# Cache the tokenizer - cl100k_base works for most modern models
_tokenizer = None

# FIX #8: Per-model message overhead constants. Different model families use
# different chat template formats, so the fixed 4-token overhead was inaccurate.
# Keys are model name substrings (matched against state.model_name.lower()).
MESSAGE_OVERHEAD_TOKENS = {
    "gpt-4": 3,
    "gpt-3.5": 3,
    "claude": 5,
    "qwen": 4,
    "llama": 3,
    "mistral": 3,
    "nemotron": 3,
}


def get_tokenizer():
    """Get or create the tokenizer (None if tiktoken is unavailable)."""
    global _tokenizer
    if _tokenizer is None and tiktoken is not None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback: rough estimate if tiktoken fails
            _tokenizer = None
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    tokenizer = get_tokenizer()
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    # Fallback: rough estimate (~4 chars per token)
    return len(text) // 4


def get_message_overhead(model_name: str = "") -> int:
    """Get the per-message token overhead for a given model."""
    name_lower = model_name.lower()
    for key, overhead in MESSAGE_OVERHEAD_TOKENS.items():
        if key in name_lower:
            return overhead
    return 4  # Default overhead


def count_message_tokens(
    messages: List[Dict[str, Any]],
    model_name: str = "",
) -> int:
    """Count tokens in a message list.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model_name: Model name for selecting the correct per-message overhead.
                   Empty string uses the default 4-token overhead.
    """
    total = 0
    overhead = get_message_overhead(model_name)
    for msg in messages:
        # FIX #8: Use configurable overhead instead of hardcoded 4.
        total += overhead  # Role + format tokens (model-dependent)
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            # Multi-part content (text + tool calls etc)
            for part in content:
                if isinstance(part, dict):
                    total += count_tokens(str(part))
                else:
                    total += count_tokens(str(part))
    return total


# =============================================================================
# SESSION STATE
# =============================================================================

@dataclass
class ToolCall:
    """Record of a single tool call."""
    timestamp: datetime
    server: str
    tool: str
    args: Dict[str, Any]
    result: Optional[str] = None
    success: bool = True
    tokens_used: int = 0
    duration_ms: int = 0


class SessionState(Enum):
    """The session's control state — the single source of truth for
    pause/plan behavior.

    States
    ------
    RUNNING
        Normal operation. User input is sent to the agent loop; tool
        calls execute (subject to gates).
    PAUSED
        Agent halted. The interactive loop accepts slash commands only;
        any other input is appended to ``Session.queued_messages`` and
        surfaced when the session resumes — it is never sent to the LLM
        and never silently dropped. ``agent_loop`` is never entered
        while PAUSED.
    AWAITING_APPROVAL
        A gate/approval prompt is open. Reserved for gate code that
        prompts asynchronously; current gates prompt synchronously, so
        the REPL is in RUNNING/PAUSED/PLAN_REVIEW at every input read.
    PLAN_REVIEW
        Plan mode. User input still flows to the agent loop (the model
        needs it to plan), but write tools are queued into
        ``Session.pending_plan`` instead of executing; ``/execute`` runs
        the queue through the normal gated path, ``/clear`` discards it.

    All transitions go through :func:`transition_state` — the single
    transition point — which keeps the legacy boolean flags
    (``paused``, ``plan_mode``) in sync for tools/gates that read them.
    """
    RUNNING = auto()
    PAUSED = auto()
    AWAITING_APPROVAL = auto()
    PLAN_REVIEW = auto()


def transition_state(state: "Session", target: SessionState) -> SessionState:
    """Move *state* to *target* — the ONLY place the mode changes.

    Records the previous mode (so ``/resume`` can return to PLAN_REVIEW)
    and synchronizes the legacy ``paused``/``plan_mode`` booleans read by
    tools.py, gates.py and the state block generator.
    """
    previous = getattr(state, "mode", SessionState.RUNNING)
    if previous is not target:
        state.previous_mode = previous
    state.mode = target
    state.paused = target is SessionState.PAUSED
    state.plan_mode = target is SessionState.PLAN_REVIEW
    return previous


def resume_target(state: "Session") -> SessionState:
    """State that ``/resume`` should return to (PLAN_REVIEW if we paused
    out of plan mode with a live queue, otherwise RUNNING)."""
    previous = getattr(state, "previous_mode", SessionState.RUNNING)
    if previous is SessionState.PLAN_REVIEW and state.pending_plan:
        return SessionState.PLAN_REVIEW
    return SessionState.RUNNING


def queue_user_message(state: "Session", text: str) -> int:
    """Queue a user message typed while PAUSED. Returns queue depth."""
    state.queued_messages.append(text)
    return len(state.queued_messages)


def drain_queued_messages(state: "Session") -> List[str]:
    """Pop all queued user messages (delivered on resume)."""
    queued = list(state.queued_messages)
    state.queued_messages.clear()
    return queued



@dataclass
class Session:
    """Complete state of the current session."""
    # Identity
    session_id: str
    model_name: str
    model_config: ModelConfig
    trust_tier: str
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Token tracking
    tokens_used: int = 0
    tokens_system_prompt: int = 0
    tokens_conversation: int = 0
    tokens_tool_calls: int = 0
    
    # Turn tracking
    turn_count: int = 0
    last_state_injection: int = 0
    
    # Tool tracking
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_failure_counts: Dict[str, int] = field(default_factory=dict)
    
    # Control state (kept in sync with `mode` by transition_state())
    mode: SessionState = SessionState.RUNNING
    previous_mode: SessionState = SessionState.RUNNING
    paused: bool = False
    tools_enabled: bool = True
    plan_mode: bool = False
    pending_plan: List[Dict] = field(default_factory=list)
    # User messages typed while PAUSED — surfaced on resume, never dropped
    queued_messages: List[str] = field(default_factory=list)
    
    # Warnings issued
    warning_issued: bool = False
    
    # FIX #6: Dynamically-added fields now formally declared on the dataclass.
    # These were being attached at runtime (e.g., startup.py lines 308-312) but
    # were absent from the declaration, which breaks __repr__, type-checking,
    # and dataclass ergonomics.
    max_context_tokens: int = 0
    model_supports_vision: bool = False
    model_supports_tools: bool = False
    model_architecture: str = "unknown"
    model_quantization: str = "unknown"
    custom_prompt: str = ""
    disabled_tools: Set[str] = field(default_factory=set)
    step_mode: bool = False
    available_tools: List[str] = field(default_factory=list)
    
    @property
    def tokens_remaining(self) -> int:
        return self.model_config.context_limit - self.tokens_used
    
    @property
    def at_warning_threshold(self) -> bool:
        return self.tokens_remaining <= self.model_config.warning_threshold
    
    @property
    def at_hard_stop(self) -> bool:
        return self.tokens_remaining <= self.model_config.hard_stop_threshold
    
    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)
    
    @property
    def tool_failure_count(self) -> int:
        return sum(1 for tc in self.tool_calls if not tc.success)
    
    @property
    def runway_turns(self) -> int:
        """Estimate how many turns left at current burn rate."""
        if self.turn_count == 0:
            return self.tokens_remaining // self.model_config.tokens_per_turn_estimate
        avg_per_turn = self.tokens_used / self.turn_count
        if avg_per_turn == 0:
            return 999
        return int(self.tokens_remaining / avg_per_turn)
    
    @property
    def elapsed_seconds(self) -> int:
        return int((datetime.now() - self.start_time).total_seconds())
    
    @property
    def elapsed_formatted(self) -> str:
        total = self.elapsed_seconds
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def to_status(self) -> SessionStatus:
        """Convert to display SessionStatus."""
        return SessionStatus(
            start_time=self.start_time,
            tokens_used=self.tokens_used,
            context_limit=self.model_config.context_limit,
            tool_calls=self.tool_call_count,
            tool_failures=self.tool_failure_count,
            model_name=self.model_name,
            trust_tier=self.trust_tier,
            paused=self.paused,
            tools_enabled=self.tools_enabled,
        )


# =============================================================================
# STATE INJECTION
# =============================================================================

# Delimiters for the injected state block. The explicit end marker lets the
# strip-regex survive blank lines *inside* the block (the old regex stopped at
# the first blank line, leaving stale state fragments in the system prompt).
STATE_BLOCK_START = "[SESSION STATE]"
STATE_BLOCK_END = "[/SESSION STATE]"


def generate_state_block(state: Session) -> str:
    """Generate the state block to inject into system prompt."""
    
    # Recent tool failures
    recent_failures = [
        tc for tc in state.tool_calls[-10:] 
        if not tc.success
    ]
    failure_summary = ""
    if recent_failures:
        failure_lines = []
        for tc in recent_failures[-3:]:
            ago = (datetime.now() - tc.timestamp).total_seconds()
            if ago < 60:
                ago_str = f"{int(ago)}s ago"
            else:
                ago_str = f"{int(ago/60)}m ago"
            failure_lines.append(f"  - {tc.server}/{tc.tool}: {ago_str}")
        failure_summary = "\n".join(failure_lines)
    
    block = f"""{STATE_BLOCK_START}
Elapsed: {state.elapsed_formatted}
Tokens used: {state.tokens_used:,} / {state.model_config.context_limit:,} ({(state.tokens_used/state.model_config.context_limit)*100:.1f}%)
Tokens remaining: {state.tokens_remaining:,}
Estimated runway: ~{state.runway_turns} turns at current burn rate
Tool calls this session: {state.tool_call_count} ({state.tool_failure_count} failed)
Turn: {state.turn_count}
"""
    
    if failure_summary:
        block += f"\nRecent tool failures:\n{failure_summary}\n"
    
    if state.paused:
        block += "\nSTATUS: ⏸ PAUSED - waiting for user\n"
    if not state.tools_enabled:
        block += "\nSTATUS: 🔒 TOOLS DISABLED - chat only\n"
    if state.plan_mode:
        block += "\nSTATUS: 📋 PLAN MODE - propose actions, do not execute\n"
    
    if state.at_warning_threshold and not state.at_hard_stop:
        block += f"\n⚠️ WARNING: Context running low. {state.tokens_remaining:,} tokens remaining.\n"
        block += "Consider wrapping up current task and summarizing.\n"

    block += STATE_BLOCK_END + "\n"
    return block

def should_inject_state(state: Session) -> bool:
    """Check if we should inject state this turn."""
    turns_since = state.turn_count - state.last_state_injection
    return turns_since >= STATE_INJECTION_INTERVAL

# Matches a fully-delimited state block (blank lines inside are fine).
_STATE_BLOCK_RE = re.compile(
    re.escape(STATE_BLOCK_START) + r"[\s\S]*?" + re.escape(STATE_BLOCK_END) + r"\s*"
)
# Legacy fallback for blocks written before the end marker existed: strip from
# the start marker up to the first blank line. Only used if no end marker is
# found anywhere in the text.
_STATE_BLOCK_LEGACY_RE = re.compile(
    re.escape(STATE_BLOCK_START) + r"[^\n]*(?:\n[^\n][^\n]*)*"
)


def _strip_state_blocks(text: str) -> str:
    """Remove previously-injected session-state blocks from a prompt."""
    if STATE_BLOCK_END in text:
        return _STATE_BLOCK_RE.sub("", text)
    return _STATE_BLOCK_LEGACY_RE.sub("", text)


def inject_state_into_messages(
    messages: List[Dict[str, Any]],
    state: Session
) -> List[Dict[str, Any]]:
    """Inject state block into the system prompt."""
    if not messages:
        return messages

    state_block = generate_state_block(state)

    # Find or create system message
    if messages[0].get("role") == "system":
        # Append to existing system prompt
        original = messages[0].get("content", "")
        # Strip any previously-injected state blocks to prevent accumulation
        # when messages are reused across turns.
        original = _strip_state_blocks(original)
        original = re.sub(r"\n{3,}", "\n\n", original).strip()
        messages[0]["content"] = f"{state_block}\n\n{original}"
    else:
        # Insert new system message
        messages.insert(0, {
            "role": "system",
            "content": state_block
        })
    
    state.last_state_injection = state.turn_count
    return messages

# =============================================================================
# CONTEXT CHECKS
# =============================================================================

def check_context_limits(state: Session) -> tuple[bool, Optional[str]]:
    """
    Check if we've hit context limits.
    Returns (should_continue, reason_if_stopping)
    """
    if state.at_hard_stop:
        print_hard_stop(state.tokens_remaining, state.model_config.hard_stop_threshold)
        return False, "hard_stop_context_limit"
    
    if state.at_warning_threshold and not state.warning_issued:
        print_context_warning(state.tokens_remaining, state.model_config.warning_threshold)
        state.warning_issued = True
    
    return True, None


def update_token_count(
    state: Session,
    messages: List[Dict[str, Any]] = None,
    response_text: str = None,
    tool_tokens: int = 0,
    api_usage: Optional[Dict[str, int]] = None,
):
    """Update token counts after a turn.
    
    Args:
        state: The current session state.
        messages: Full conversation messages list (used for fallback counting).
        response_text: Response text from the model (used for fallback counting).
        tool_tokens: Tokens consumed by tool calls (used for fallback counting).
        api_usage: Optional API-reported usage dict with keys like
                   'prompt_tokens', 'completion_tokens', 'total_tokens'.
                   When provided, this is treated as the ground truth and
                   all other counting is skipped.
    
    FIX #4: Previously tokens_used only tracked response tokens, never counted
    prompt tokens or the full conversation. Hard-stop protection was unreliable.
    Now prioritizes API-reported usage.total_tokens as ground truth, and falls
    back to counting the full messages list.
    
    FIX #5: Previously tool tokens were double/triple counted because they were
    added incrementally on top of response text AND messages. Now when API usage
    is available, it is trusted exclusively without adding tool tokens on top.
    """
    if api_usage is not None:
        # FIX #4 & #5: API-reported usage is ground truth. Trust it exclusively
        # and do not add tool_tokens or response_text on top.
        total = api_usage.get("total_tokens")
        if total is not None:
            state.tokens_used = total
            # Track breakdowns when available
            state.tokens_conversation = api_usage.get("prompt_tokens", state.tokens_conversation)
            state.tokens_tool_calls = api_usage.get("completion_tokens", state.tokens_tool_calls)
            state.last_activity = datetime.now()
            return
        # If total_tokens is missing, fall through to heuristic counting.
    
    # Fallback: count tokens heuristically from messages and response.
    if messages:
        # FIX #4: Count full messages list, not just the latest response.
        state.tokens_conversation = count_message_tokens(messages, state.model_name)
        # Set total used to the conversation count (system + conversation)
        state.tokens_used = state.tokens_system_prompt + state.tokens_conversation
    
    if response_text:
        state.tokens_used += count_tokens(response_text)
    
    if tool_tokens:
        # FIX #5: Only add tool tokens in fallback mode (no API usage).
        # When API usage is available, tool tokens are already included.
        state.tokens_tool_calls += tool_tokens
        state.tokens_used += tool_tokens
    
    state.last_activity = datetime.now()


def record_tool_call(
    state: Session,
    server: str,
    tool: str,
    args: Dict[str, Any],
    result: str,
    success: bool,
    duration_ms: int = 0
):
    """Record a tool call in session state."""
    tokens = count_tokens(str(args)) + count_tokens(result)
    
    tc = ToolCall(
        timestamp=datetime.now(),
        server=server,
        tool=tool,
        args=args,
        result=result[:500] if result else None,  # Truncate for storage
        success=success,
        tokens_used=tokens,
        duration_ms=duration_ms,
    )
    
    state.tool_calls.append(tc)
    # Token accounting: the authoritative total comes from the API usage dict
    # passed to update_token_count(), which already includes tool I/O in
    # prompt_tokens. Adding `tokens` to state.tokens_used here would count
    # every tool call twice, so we only track the per-tool estimate for
    # diagnostics.
    state.tokens_tool_calls += tokens
    
    # Track failures per tool
    tool_key = f"{server}/{tool}"
    if not success:
        state.tool_failure_counts[tool_key] = state.tool_failure_counts.get(tool_key, 0) + 1

def get_tool_failure_count(state: Session, server: str, tool: str) -> int:
    """Get failure count for a specific tool."""
    tool_key = f"{server}/{tool}"
    return state.tool_failure_counts.get(tool_key, 0)

# =============================================================================
# SESSION CREATION
# =============================================================================

def create_session(
    model_name: str,
    trust_tier: str = "supervised",
    session_id: str = None
) -> Session:
    """Create a new session state."""
    if session_id is None:
        # Local import avoids a module cycle; ids are collision-proof
        # (timestamp + nanoseconds + short uuid).
        from .session_log import new_session_id
        session_id = new_session_id()
    
    model_config = get_model_config(model_name)
    
    return Session(
        session_id=session_id,
        model_name=model_name,
        model_config=model_config,
        trust_tier=trust_tier,
    )

# =============================================================================
# CONTINUATION SCROLL GENERATION
# =============================================================================

def generate_continuation_scroll(state: Session, conversation_summary: str = "") -> str:
    """
    Generate a continuation scroll for session handoff.
    This is what the next OGneey reads to pick up where we left off.
    """
    
    # Recent tool calls summary
    recent_tools = state.tool_calls[-10:]
    tool_summary = []
    for tc in recent_tools:
        status = "✓" if tc.success else "✗"
        tool_summary.append(f"  {status} {tc.server}/{tc.tool}")
    tool_summary_str = "\n".join(tool_summary) if tool_summary else "  (none)"
    
    # Failed tools to avoid
    failed_tools = [
        f"  - {k}: {v} failures" 
        for k, v in state.tool_failure_counts.items() 
        if v >= 2
    ]
    failed_tools_str = "\n".join(failed_tools) if failed_tools else "  (none)"
    
    scroll = f"""# Continuation Scroll
## Session {state.session_id}

**Handoff Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Previous Model:** {state.model_name}
**Session Duration:** {state.elapsed_formatted}
**Tokens Used:** {state.tokens_used:,} / {state.model_config.context_limit:,}
**Reason for Handoff:** Context limit reached

---

## Conversation Summary

{conversation_summary if conversation_summary else "(No summary provided)"}

---

## Recent Tool Activity

{tool_summary_str}

## Tools to Avoid (Multiple Failures)

{failed_tools_str}

---

## Session State at Handoff

- Trust Tier: {state.trust_tier}
- Tools Enabled: {state.tools_enabled}
- Plan Mode: {state.plan_mode}
- Total Tool Calls: {state.tool_call_count}
- Total Failures: {state.tool_failure_count}

---

## Notes for Next OGneey

Pick up where the previous session left off. Review the conversation summary
above and continue assisting the user. The tool failure information should
help you avoid repeating unsuccessful approaches.

If the user's task was incomplete, summarize what remains and propose next steps.
"""
    
    return scroll

def save_continuation_scroll(state: Session, summary: str = "") -> Path:
    """Save continuation scroll to file."""
    scroll = generate_continuation_scroll(state, summary)
    
    CONTINUATION_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = f"continuation_{state.session_id}.md"
    filepath = CONTINUATION_DIR / filename
    
    filepath.write_text(scroll)
    
    return filepath

# =============================================================================
# STATE FILE (PERSISTENT BRAIN)
# =============================================================================

def load_state_file() -> Optional[str]:
    """Load the persistent state file if it exists."""
    if STATE_FILE.exists():
        try:
            return STATE_FILE.read_text()
        except Exception:
            return None
    return None

def update_state_file(
    state: Session,
    user_context: str = "",
    active_tasks: List[str] = None,
    notes: List[str] = None
):
    """Update the persistent state file."""
    
    active_tasks = active_tasks or []
    notes = notes or []
    
    tasks_str = "\n".join(f"  - {t}" for t in active_tasks) if active_tasks else "  (none)"
    notes_str = "\n".join(f"  - {n}" for n in notes) if notes else "  (none)"
    
    content = f"""# OGnee State
*Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Current Context

{user_context if user_context else "(No context set)"}

## Active Tasks

{tasks_str}

## Notes

{notes_str}

## Recent Sessions

- **{state.session_id}**: {state.model_name}, {state.elapsed_formatted}, {state.tokens_used:,} tokens, {state.tool_call_count} tool calls

## System Info

- Default Trust Tier: {state.trust_tier}
- Last Model: {state.model_name}
"""
    
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(content)

def get_latest_continuation() -> Optional[Path]:
    """Get the most recent continuation scroll, if any."""
    if not CONTINUATION_DIR.exists():
        return None
    
    scrolls = sorted(CONTINUATION_DIR.glob("continuation_*.md"), reverse=True)
    return scrolls[0] if scrolls else None


# =============================================================================
# PATH HELPERS (with ~ expansion)
# =============================================================================

# FIX #7: The config.py path validation functions don't expand ~ (tilde) before
# calling Path.resolve(), so paths like "~/Documents" are never matched against
# EXCLUDED_PATHS or READ_ONLY_PATHS correctly. These wrappers add the missing
# os.path.expanduser() step and should be used in place of the raw config
# functions whenever user-provided paths may contain a tilde.


def _expand_path(path: str) -> str:
    """Expand ~ and resolve a path string."""
    return str(Path(os.path.expanduser(path)).resolve())


def is_path_excluded_expanded(path: str) -> bool:
    """Check if path is excluded, with ~ expansion."""
    return is_path_excluded(_expand_path(path))


def is_path_read_only_expanded(path: str) -> bool:
    """Check if path is read-only, with ~ expansion."""
    return is_path_read_only(_expand_path(path))


def is_path_writable_expanded(path: str) -> bool:
    """Check if path is writable, with ~ expansion."""
    return is_path_writable(_expand_path(path))


def can_access_path_expanded(path: str) -> bool:
    """Check if path is accessible, with ~ expansion."""
    return can_access_path(_expand_path(path))


def get_shadow_path_expanded(real_path: str) -> Path:
    """Get shadow path for a real path, with ~ expansion."""
    return get_shadow_path(_expand_path(real_path))
