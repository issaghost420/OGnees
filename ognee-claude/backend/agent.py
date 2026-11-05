#!/usr/bin/env python3
"""
agent.py - Main agent loop

The core. Ties together MCP servers, tool execution, context tracking,
gates, logging, and the LLM. This is where OGneeys come to life.
"""

import asyncio
import concurrent.futures
import http.client
import json
import os
import re
import signal
import socket
import threading
import time
import urllib.parse
import requests
import wcwidth
from collections import Counter
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from wcwidth import wcswidth
from .output_buffer import reset_output_buffer, MAX_TOOL_RESULT_CHARS

from .config import (
    LM_STUDIO_URL,
    LM_STUDIO_TOKEN,
    MAX_ITERATIONS,
    REQUEST_TIMEOUT,
    DEFAULT_TRUST_TIER,
    INPUT_HISTORY_FILE,
)
from .context import (
    Session,
    SessionState,
    transition_state,
    queue_user_message,
    drain_queued_messages,
    post_inbox,
    drain_inbox,
    request_force_pause,
    check_context_limits,
    update_token_count,
    inject_state_into_messages,
    should_inject_state,
    save_continuation_scroll,
    count_tokens,
    count_message_tokens,
)
from .commands import (
    is_command,
    execute_command,
    CommandResult,
)
from .gates import get_tier_name, get_tier_level, check_and_enforce_ssh
from .tools import (
    execute_tool_call,
    execute_tool_calls,
    format_tool_results_for_llm,
    format_tools_for_prompt,
    get_available_tools,
    get_failure_summary,
    get_local_tools,
    ToolResult,
    READ_ONLY_TOOLS,
)
from .mcp import MCPManager, load_mcp, make_native_name, NATIVE_NAME_SEP
from .session_log import SessionLogger, log_tool_to_global, redact_secrets
from .inputhub import get_hub, strip_ansi
from .shell_policy import get_policy
from .startup import startup, shutdown, build_system_prompt
from .display import (
    print_system,
    print_success,
    print_warning,
    print_error,
    print_ognee_response,
    print_ognee_thinking,
    print_status_line,
    print_tool_result,
    print_dim,
    read_input,
    COLORS,
)

# =============================================================================
# CONSTANTS
# =============================================================================

INNER_WIDTH = 78

# Token-budget-based message retention (replaces hard message count cap)
MAX_CONTEXT_FOR_MESSAGES = 0.75  # use up to 75% of context for conversation history
MIN_RECENT_MESSAGES = 6          # always keep at least this many recent messages

# =============================================================================
# MODEL TOOL-CALLING PROFILES
# =============================================================================

# Each profile gates BOTH the tool-format section of the system prompt and
# the parser used on model output:
#   native   - model is called with the OpenAI `tools` parameter; responses
#              carry structured tool_calls; results go back as role:"tool".
#   qwen-xml - prompt + parser expect <tool_call>{json}</tool_call>.
#   json     - prompt + parser expect a plain {"tool": ..., "arguments": ...}
#              JSON object.
#   gemma    - prompt + parser expect Gemma's
#              <start_function_call>call:name{json}<end_function_call>.
TOOL_PROFILES = ("native", "qwen-xml", "json", "gemma")

# Name substrings of model families known to support native tool calling in
# LM Studio / llama.cpp. `lms ps` trainedForToolUse (state.model_supports_tools)
# is checked first; this list is the fallback when detection is unavailable.
_NATIVE_MODEL_HINTS = (
    "qwen3-coder", "qwen3_coder", "glm", "nemotron", "seed-oss", "seed_oss",
    "gpt-oss", "gpt_oss", "functionary",
)

# Default completion budget. 4096 truncated large JSON argument payloads
# mid-object, producing unparseable calls; raise it and make it configurable.
DEFAULT_MAX_TOKENS = int(os.environ.get("OGNEE_MAX_TOKENS", "8192"))
DEFAULT_TEMPERATURE = float(os.environ.get("OGNEE_TEMPERATURE", "0.7"))

# How many identical (tool, args) calls in a row trigger the loop guard.
LOOP_GUARD_MAX_REPEATS = 3


def detect_tool_profile(state: 'Session') -> str:
    """Choose the tool-calling profile for the loaded model.

    Precedence: OGNEE_TOOL_PROFILE env override > gemma name match >
    lms-detected native support / native model hints > qwen name match >
    plain JSON fallback.
    """
    override = os.environ.get("OGNEE_TOOL_PROFILE", "").strip().lower()
    if override in TOOL_PROFILES:
        return override

    name = (getattr(state, "model_name", "") or "").lower()
    if "gemma" in name:
        return "gemma"
    if getattr(state, "model_supports_tools", False):
        return "native"
    if any(hint in name for hint in _NATIVE_MODEL_HINTS):
        return "native"
    if "qwen" in name:
        return "qwen-xml"
    return "json"


def build_tool_registry(mcp: MCPManager) -> Dict[str, str]:
    """Build the tool-name registry: alias -> canonical ``server/tool``.

    Aliases include the full name, the OpenAI-legal native name
    (``server__tool``), and the bare tool name when it is unambiguous.
    Every dispatch must resolve through this registry — a tool name that
    isn't registered is never executed.
    """
    full_names: List[str] = []
    for tool in mcp.all_tools:
        full = tool.get("full_name")
        if full and "/" in full:
            full_names.append(full)
    for local_tool in get_local_tools():
        full_names.append(local_tool["full_name"])

    aliases: Dict[str, str] = {}
    for full in full_names:
        aliases[full] = full
        server, tool = full.split("/", 1)
        aliases[make_native_name(server, tool)] = full

    bare_counts = Counter(full.split("/", 1)[1] for full in full_names)
    for full in full_names:
        bare = full.split("/", 1)[1]
        if bare_counts[bare] == 1:
            aliases.setdefault(bare, full)
    return aliases


def resolve_tool_name(server: str, tool: str, registry: Dict[str, str]) -> Optional[str]:
    """Resolve a parsed (server, tool) pair to a canonical registry name.

    Returns the canonical ``server/tool`` name, or None if the name is not in
    the registry (never dispatch such a call).
    """
    candidates = []
    if server and server != "unknown":
        candidates.append(f"{server}/{tool}")
    if tool:
        candidates.append(tool)
    for candidate in candidates:
        if candidate in registry:
            return registry[candidate]
    return None


class LoopDetector:
    """Detects a model repeating the identical tool call over and over.

    Tracks consecutive calls keyed by (full tool name, canonical args). When
    the same key appears LOOP_GUARD_MAX_REPEATS times in a row, the guard
    trips and the agent loop stops instead of burning iterations (and
    spawning subprocess work) on a stuck model.
    """

    def __init__(self, max_repeats: int = LOOP_GUARD_MAX_REPEATS):
        self.max_repeats = max_repeats
        self._last_key: Optional[tuple] = None
        self._count = 0

    def check(self, full_name: str, args: dict) -> bool:
        """Record a call; returns True if the loop guard has tripped."""
        try:
            args_key = json.dumps(args or {}, sort_keys=True, default=str)
        except (TypeError, ValueError):
            args_key = str(args)
        key = (full_name, args_key)
        if key == self._last_key:
            self._count += 1
        else:
            self._last_key, self._count = key, 1
        return self._count >= self.max_repeats

    def reset(self):
        self._last_key = None
        self._count = 0


def build_native_tools(mcp: MCPManager, state: Optional[Session] = None) -> list:
    """OpenAI ``tools`` schemas for MCP tools plus built-in local tools.

    Tools/servers the operator disabled (/disable) are left out entirely, so
    the model never sees them and no tokens are spent describing them.
    """
    disabled = set(getattr(state, "disabled_tools", set()) or ()) if state is not None else set()

    def _off(server: str, name: str) -> bool:
        return server in disabled or f"{server}/{name}" in disabled

    schemas = []
    for tool in mcp.all_tools:
        server, name = tool.get("server", ""), tool.get("name", "")
        if _off(server, name):
            continue
        schemas.append({
            "type": "function",
            "function": {
                "name": make_native_name(server, name),
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
            },
        })
    for local_tool in get_local_tools():
        if _off("local", local_tool["name"]):
            continue
        schemas.append({
            "type": "function",
            "function": {
                "name": make_native_name("local", local_tool["name"]),
                "description": local_tool["description"],
                "parameters": local_tool.get(
                    "inputSchema", {"type": "object", "properties": {}}),
            },
        })
    return schemas

# recompiling on every call to extract_tool_calls_text().
# (removed brittle regex constants -- replaced by robust multi-format parser below)



def _message_token_cost(msg: dict) -> int:
    """Token cost of a message, INCLUDING native tool-call arguments.

    Tool-call argument blobs (e.g. a 13k-char write_file call) are often the
    bulk of a turn; counting only ``content`` made them invisible to the
    context budget.
    """
    cost = count_tokens(msg.get("content") or "")
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        try:
            cost += count_tokens(json.dumps(tool_calls))
        except (TypeError, ValueError):
            cost += count_tokens(str(tool_calls))
    return cost


def _trim_messages(messages: list, state: Session):
    """Trim conversation history to fit within token budget.

    Keeps: system prompt (idx 0) + first user message (the task) + as many
    recent TURN GROUPS as fit. A turn group is an assistant message carrying
    ``tool_calls`` together with ALL of its ``role: "tool"`` replies — the
    group is trimmed or kept atomically so a cut never lands between a
    tool_calls message and its replies (which would produce dangling
    tool_call_ids and API 400s).
    """
    if len(messages) <= MIN_RECENT_MESSAGES + 1:
        return  # nothing to trim

    budget = int(state.model_config.context_limit * MAX_CONTEXT_FOR_MESSAGES)

    # Cost of system prompt + first user message (the original task)
    fixed_cost = _message_token_cost(messages[0])
    if len(messages) > 1:
        fixed_cost += _message_token_cost(messages[1])

    # Group the tail (index >= 2) into atomic units: an assistant message
    # with tool_calls owns the role:"tool" messages that follow it; a bare
    # role:"tool" message with no preceding owner attaches to whatever group
    # precedes it so we never emit an orphan tool message either.
    groups = []  # list of lists of message indices
    for i in range(2, len(messages)):
        msg = messages[i]
        if msg.get("role") == "tool" and groups:
            groups[-1].append(i)
        else:
            groups.append([i])
            # assistant + tool_calls: absorb following tool messages via the
            # rule above on subsequent iterations.

    remaining = budget - fixed_cost
    if remaining <= 0:
        # System prompt alone exceeds budget — keep minimum recent messages
        messages[:] = [messages[0]] + messages[-MIN_RECENT_MESSAGES:]
        return

    # Walk groups backwards, accumulating until budget is exhausted
    keep_from_group = len(groups)  # oldest group to keep (exclusive of fixed)
    for g in range(len(groups) - 1, -1, -1):
        group_cost = sum(_message_token_cost(messages[i]) for i in groups[g])
        if group_cost > remaining:
            break
        remaining -= group_cost
        keep_from_group = g

    # Always keep at least MIN_RECENT_MESSAGES from the tail
    max_keep_from = max(2, len(messages) - MIN_RECENT_MESSAGES)
    keep_from = min(groups[keep_from_group][0] if keep_from_group < len(groups) else len(messages),
                    max_keep_from)

    if keep_from <= 2:
        return  # keeping everything already

    # Trim: system + first user msg + kept tail
    dropped = keep_from - 2
    messages[:] = [messages[0], messages[1]] + messages[keep_from:]
    if dropped > 0:
        # Insert a breadcrumb so the model knows history was trimmed
        messages.insert(2, {
            "role": "user",
            "content": f"[{dropped} earlier messages trimmed for context budget]"
        })

# =============================================================================
# DISPLAY WIDTH HELPERS
# =============================================================================

_ANSI_SPLIT_RE = re.compile(r"(\x1b\[[0-9;?]*[ -/]*[@-~])")


def visual_width(s: str) -> int:
    return max(0, wcswidth(strip_ansi(s)))

# width tracking using wcwidth() per character; ANSI escapes are zero-width
# (they used to count as -1 each, which is why the banner box was ragged).
def pad_to_width(s: str, width: int) -> str:
    cumulative = 0
    out = []
    for part in _ANSI_SPLIT_RE.split(s):
        if not part:
            continue
        if _ANSI_SPLIT_RE.fullmatch(part):
            out.append(part)
            continue
        for ch in part:
            ch_width = wcwidth.wcwidth(ch)
            if ch_width is None or ch_width < 0:
                ch_width = 0
            if cumulative + ch_width > width:
                break
            cumulative += ch_width
            out.append(ch)
    result = "".join(out)
    if cumulative < width:
        result += " " * (width - cumulative)
    return result + COLORS.RESET

def row(content: str) -> str:
    return f"│ {pad_to_width(content, INNER_WIDTH - 2)} │"

# =============================================================================
# MID-TURN CONTROL
# =============================================================================
#
# The keyboard is read on the main thread while the agent runs on a worker
# thread (see interactive_session). /pf calls request_force_pause(), which
# cancels the in-flight LLM request below, cancels MCP tool calls, and
# resolves any open approval prompt; race_abort() is the safety net.

# =============================================================================
# LLM COMMUNICATION
# =============================================================================

# Reuse a single TCP connection to LM Studio (localhost) across all calls.
_llm_session = requests.Session()
_llm_session.headers.update({
    "Content-Type": "application/json",
    "Authorization": f"Bearer {LM_STUDIO_TOKEN}",
})

def get_current_model() -> str:
    """Query what model is currently loaded in LM Studio."""
    try:
        response = _llm_session.post(
            LM_STUDIO_URL,
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=30
        )
        data = response.json()
        return data.get("model", "unknown")
    except Exception:
        return "unknown"


class OperatorCancelled(Exception):
    """The operator force-paused while an LLM request was in flight."""


class _InflightRequest:
    """Handle on one LLM HTTP request so /pf can kill its socket.

    Shutting the socket down unblocks the waiting thread immediately and
    tells LM Studio the client went away, so it can stop generating.
    """

    def __init__(self):
        self.conn = None
        self.cancelled = False
        self._lock = threading.Lock()

    def attach(self, conn):
        with self._lock:
            self.conn = conn
            cancelled = self.cancelled
        if cancelled:
            self._shutdown()

    def cancel(self):
        with self._lock:
            self.cancelled = True
        self._shutdown()

    def _shutdown(self):
        conn = self.conn
        sock = getattr(conn, "sock", None) if conn is not None else None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


_inflight_lock = threading.Lock()
_inflight_requests: set = set()

# OGNEE_HTTP_CLIENT=requests restores the old (uncancellable) transport.
_HTTP_CLIENT = os.environ.get("OGNEE_HTTP_CLIENT", "cancellable").strip().lower()


def cancel_inflight_llm_requests() -> int:
    """Abort every LLM request in flight (called by /pf). Returns the count."""
    with _inflight_lock:
        reqs = list(_inflight_requests)
    for r in reqs:
        r.cancel()
    return len(reqs)


def _post_json_requests(url: str, payload: dict, timeout: float) -> dict:
    response = _llm_session.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _post_json_cancellable(url: str, payload: dict, timeout: float, req: _InflightRequest) -> dict:
    """POST JSON with http.client so the socket can be shut down from /pf.

    Errors are raised as requests exceptions so existing handlers still
    catch them as "API Error".
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return _post_json_requests(url, payload, timeout)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parts.hostname, port, timeout=timeout)
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {LM_STUDIO_TOKEN}",
    }
    try:
        conn.connect()
        req.attach(conn)
        if req.cancelled:
            raise OperatorCancelled()
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
    except OperatorCancelled:
        raise
    except (OSError, http.client.HTTPException) as e:
        if req.cancelled:
            raise OperatorCancelled() from None
        if isinstance(e, TimeoutError):
            raise requests.exceptions.Timeout(
                f"LM Studio didn't answer within {timeout}s ({url})") from None
        raise requests.exceptions.ConnectionError(f"{type(e).__name__}: {e} ({url})") from None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if req.cancelled:
        raise OperatorCancelled()
    if resp.status >= 400:
        snippet = data[:500].decode("utf-8", "replace")
        raise requests.exceptions.HTTPError(f"{resp.status} {resp.reason} from {url}: {snippet}")
    try:
        return json.loads(data.decode("utf-8"))
    except ValueError as e:
        raise ValueError(f"LM Studio returned invalid JSON: {e}") from None


async def chat(messages: list, tools: list = None) -> dict:
    """Send chat request to LM Studio (async — does not block the event loop).

    When ``tools`` is provided (native tool-calling profile), the schemas are
    sent via the OpenAI ``tools`` parameter with ``tool_choice="auto"``.
    The request is registered so /pf can cancel it mid-generation.
    """
    payload = {
        "messages": messages,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }

    # Add tools if provided (for native function calling)
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    loop = asyncio.get_event_loop()
    if _HTTP_CLIENT == "requests":
        return await loop.run_in_executor(
            None, lambda: _post_json_requests(LM_STUDIO_URL, payload, REQUEST_TIMEOUT))
    req = _InflightRequest()
    with _inflight_lock:
        _inflight_requests.add(req)
    try:
        return await loop.run_in_executor(
            None, lambda: _post_json_cancellable(LM_STUDIO_URL, payload, REQUEST_TIMEOUT, req))
    finally:
        with _inflight_lock:
            _inflight_requests.discard(req)

# empty turns. Now checks for API errors first, only catches structural errors.
def extract_response_text(response: dict) -> str:
    """Extract text from OpenAI-format response."""
    # Check for API-level errors before trying to extract content
    if "error" in response:
        error_msg = response["error"]
        if isinstance(error_msg, dict):
            error_msg = error_msg.get("message", str(error_msg))
        print_error(f"  LLM API error: {error_msg}")
        return f"[Error: {error_msg}]"
    
    try:
        return response["choices"][0]["message"].get("content", "") or ""
    except (KeyError, IndexError):
        # Only structural errors (malformed response shape) land here
        return ""

def extract_reasoning_content(response: dict) -> str:
    """Extract reasoning/thinking content from response if present."""
    # Check for API-level errors before trying to extract content
    if "error" in response:
        return ""
    
    try:
        msg = response["choices"][0]["message"]
        # Check various fields where reasoning might live
        reasoning = msg.get("reasoning_content", "")
        if not reasoning:
            reasoning = msg.get("thinking", "")
        if not reasoning:
            reasoning = msg.get("reasoning", "")
        return reasoning
    except (KeyError, IndexError):
        # Only structural errors (malformed response shape) land here
        return ""

def extract_tool_calls_native(response: dict, registry: Dict[str, str] = None) -> list:
    """Extract tool calls from native OpenAI tool_calls format.

    Names are resolved through the tool registry (which knows the
    ``server__tool`` native aliases). Malformed JSON arguments are NOT
    silently dropped — the call is returned with a ``parse_error`` key so the
    agent loop can send a correction back to the model instead of executing
    a tool with missing arguments.
    """
    try:
        msg = response["choices"][0]["message"]
        tool_calls = msg.get("tool_calls", [])

        extracted = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_str = fn.get("arguments", "{}")

            call = {
                "id": tc.get("id", ""),
                "server": "unknown",
                "tool": name,
                "args": {},
            }

            if isinstance(args_str, str):
                try:
                    call["args"] = json.loads(args_str) if args_str.strip() else {}
                except json.JSONDecodeError as e:
                    call["parse_error"] = str(e)
            elif isinstance(args_str, dict):
                call["args"] = args_str
            else:
                call["parse_error"] = f"unexpected arguments type: {type(args_str).__name__}"

            # Resolve the name via the registry (handles server__tool aliases)
            if registry:
                full = registry.get(name)
                if full:
                    call["server"], call["tool"] = full.split("/", 1)

            extracted.append(call)

        return extracted
    except (KeyError, IndexError):
        return []

def _extract_json_objects(text: str) -> list:
    """Extract complete JSON objects from text using brace-depth counting.

    This is the key to handling nested objects/arrays: it walks character by
    character, tracking brace depth while respecting string boundaries, so
    braces inside strings or nested objects don't prematurely terminate.
    """
    objects = []
    i = 0
    while i < len(text):
        # Find the next unescaped opening brace
        if text[i] == '{':
            depth = 0
            in_string = False
            escape_next = False
            start = i

            for j in range(i, len(text)):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"' and not in_string:
                    in_string = True
                elif ch == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            # Found a complete JSON object
                            candidate = text[start:j+1]
                            objects.append(candidate)
                            i = j + 1
                            break
            else:
                # No matching close brace -- incomplete object, skip it
                i += 1
        else:
            i += 1

    return objects

def _parse_tool_json(obj: dict) -> Optional[dict]:
    """Parse a JSON dict into a normalized tool call record.

    Handles multiple key naming conventions:
    - {"tool": "server/name", "arguments": {...}}
    - {"tool": "server/name", "args": {...}}
    - {"name": "server/name", "parameters": {...}}
    - {"name": "server/name", "arguments": {...}}
    """
    # Determine tool name
    tool_name = obj.get("tool", obj.get("name", ""))
    if not tool_name:
        return None

    # Determine arguments
    args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
    if not isinstance(args, dict):
        args = {}

    # Parse server/tool from name
    if "/" in tool_name:
        server, tool = tool_name.split("/", 1)
    elif "__" in tool_name:
        server, tool = tool_name.split("__", 1)
    else:
        server = "unknown"
        tool = tool_name

    return {
        "id": obj.get("id", ""),
        "server": server,
        "tool": tool,
        "args": args
    }

def _parse_python_style_call(text: str) -> Optional[dict]:
    """Parse a Python-style bare call like: server/tool(arg1=val1, arg2=val2)

    Returns None if the text doesn't match the expected pattern.
    Handles string values (both single and double quotes), numbers, booleans,
    and bare identifiers.
    """
    text = text.strip()

    # Match pattern: namespace/tool_name(...)
    match = re.match(r'^([\w\-./]+)\s*\((.*)\)\s*$', text, re.DOTALL)
    if not match:
        return None

    full_name = match.group(1).strip()
    args_str = match.group(2).strip()

    # Parse server/tool
    if "/" in full_name:
        server, tool = full_name.split("/", 1)
    elif "__" in full_name:
        server, tool = full_name.split("__", 1)
    else:
        server = "unknown"
        tool = full_name

    # Parse keyword arguments
    args = {}
    if args_str:
        args = _parse_kwargs(args_str)

    return {
        "id": "",
        "server": server,
        "tool": tool,
        "args": args
    }

def _parse_kwargs(args_str: str) -> dict:
    """Parse keyword arguments from a string like: arg1=val1, arg2="string val", arg3=123

    Handles quoted strings (both single and double quotes), numbers, booleans,
    and None. Returns a dict of parsed arguments.
    """
    args = {}
    if not args_str or args_str.strip() == '':
        return args

    # Split on commas, but respect nested structures and strings
    parts = []
    current = []
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_string = False
    string_char = None
    escape_next = False

    for ch in args_str:
        if escape_next:
            current.append(ch)
            escape_next = False
            continue
        if ch == '\\':
            current.append(ch)
            escape_next = True
            continue
        if ch in ('"', "'") and not in_string:
            in_string = True
            string_char = ch
            current.append(ch)
        elif ch == string_char and in_string:
            in_string = False
            string_char = None
            current.append(ch)
        elif in_string:
            current.append(ch)
        else:
            if ch == '(':
                depth_paren += 1
            elif ch == ')':
                depth_paren -= 1
            elif ch == '[':
                depth_bracket += 1
            elif ch == ']':
                depth_bracket -= 1
            elif ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace -= 1

            if ch == ',' and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)

    if current:
        parts.append(''.join(current).strip())

    # Parse each key=value pair
    for part in parts:
        if '=' not in part:
            continue
        # Split on first = only
        eq_idx = part.index('=')
        key = part[:eq_idx].strip()
        val_str = part[eq_idx + 1:].strip()

        if not key:
            continue

        # Parse the value
        value = _parse_value(val_str)
        args[key] = value

    return args

def _parse_value(val_str: str) -> Any:
    """Parse a single value from a kwargs string."""
    val_str = val_str.strip()

    if not val_str:
        return ""

    # String literals (quoted)
    if (val_str.startswith('"') and val_str.endswith('"')) or \
       (val_str.startswith("'") and val_str.endswith("'")):
        return val_str[1:-1]

    # Try JSON parse first (handles numbers, booleans, null, arrays, objects)
    try:
        return json.loads(val_str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Boolean literals
    if val_str.lower() == 'true':
        return True
    if val_str.lower() == 'false':
        return False
    if val_str.lower() == 'none' or val_str.lower() == 'null':
        return None

    # Integer
    try:
        return int(val_str)
    except ValueError:
        pass

    # Float
    try:
        return float(val_str)
    except ValueError:
        pass

    # Bare string (unquoted identifier)
    return val_str

def extract_tool_calls_text(text: str, registry: Dict[str, str] = None,
                            profile: str = "json") -> list:
    """Extract tool calls from model text.

    Strategies, most specific first:
    1. Fenced JSON blocks: ```json {"tool": "...", "arguments": {...}} ```
    2. Inline JSON objects: {"tool": "...", "arguments": {...}}
       (AMBIGUOUS — only accepted if the name resolves in the registry, so
       JSON examples in prose never become phantom tool calls)
    3. XML tool calls: <tool_call>server/tool(arg1=val1)</tool_call>
    4. XML-wrapped JSON: <tool_call>{"tool":"...",...}</tool_call>
    5. Gemma-style: <start_function_call>call:server/name{json}<end_function_call>
       (only tried for the "gemma" profile)
    6. Array wrappers: [{"tool":...}] (AMBIGUOUS — registry match required)
    7. Python-style bare: server/tool(arg1=val1) (AMBIGUOUS — registry match
       required, so markdown tables and paths in prose are never executed)

    Nested JSON is handled via brace-depth counting (not brittle regex), so
    braces inside strings or nested objects don't cause premature truncation.
    """
    if not text or not text.strip():
        return []

    registry = registry or {}

    def _known(parsed: Optional[dict]) -> Optional[dict]:
        """Return the parsed call only if its name resolves in the registry.

        Used for ambiguous strategies: anything name-like in prose (markdown
        table cells, URLs, file paths) is dropped here instead of executed.
        """
        if not parsed:
            return None
        if not registry:
            return parsed  # no registry available — accept (legacy behavior)
        if resolve_tool_name(parsed.get("server", ""), parsed.get("tool", ""), registry):
            return parsed
        return None

    extracted = []

    # ==========================================================================
    # Strategy 1: Fenced JSON blocks (```json ... ```) — explicit format
    # ==========================================================================
    fenced_pattern = re.compile(r'```(?:json)?\s*\n?(.*?)\n?\s*```', re.DOTALL)
    for match in fenced_pattern.finditer(text):
        block_content = match.group(1).strip()
        if not block_content:
            continue
        # Extract JSON objects from inside the fence
        json_objs = _extract_json_objects(block_content)
        for obj_str in json_objs:
            try:
                obj = json.loads(obj_str)
                parsed = _parse_tool_json(obj)
                if parsed:
                    extracted.append(parsed)
                elif isinstance(obj, list):
                    # Bare array inside a fence: [{"tool":...}]
                    for item in obj:
                        if isinstance(item, dict):
                            parsed = _parse_tool_json(item)
                            if parsed:
                                extracted.append(parsed)
            except json.JSONDecodeError:
                continue

    # ==========================================================================
    # Strategy 2: Inline JSON objects (bare JSON in the text) — AMBIGUOUS
    # ==========================================================================
    json_objs = _extract_json_objects(text)
    for obj_str in json_objs:
        try:
            obj = json.loads(obj_str)

            # Check for {"tool_calls": [...]} wrapper
            if isinstance(obj, dict) and 'tool_calls' in obj:
                tc_list = obj['tool_calls']
                if isinstance(tc_list, list):
                    for tc in tc_list:
                        if isinstance(tc, dict):
                            parsed = _known(_parse_tool_json(tc))
                            if parsed:
                                extracted.append(parsed)
                continue

            # Standard single tool call object
            parsed = _known(_parse_tool_json(obj))
            if parsed:
                extracted.append(parsed)
        except json.JSONDecodeError:
            continue

    # ==========================================================================
    # Strategy 3: XML-style tool calls with Python syntax — explicit format
    #    <tool_call>server/tool(arg1=val1, arg2=val2)</tool_call>
    # ==========================================================================
    xml_python_pattern = re.compile(
        r'<tool_call>\s*([\w\-./]+\s*\([^)]*\))\s*</tool_call>',
        re.DOTALL
    )
    for match in xml_python_pattern.finditer(text):
        call_text = match.group(1).strip()
        parsed = _parse_python_style_call(call_text)
        if parsed:
            extracted.append(parsed)

    # ==========================================================================
    # Strategy 4: XML-wrapped JSON — explicit format
    #    <tool_call>{"tool": "server/tool", "arguments": {...}}</tool_call>
    # ==========================================================================
    xml_json_pattern = re.compile(
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
        re.DOTALL
    )
    for match in xml_json_pattern.finditer(text):
        json_str = match.group(1).strip()
        # Find the complete JSON object using brace counting
        json_objs = _extract_json_objects(json_str)
        for obj_str in json_objs:
            try:
                obj = json.loads(obj_str)
                parsed = _parse_tool_json(obj)
                if parsed:
                    extracted.append(parsed)
            except json.JSONDecodeError:
                continue

    # ==========================================================================
    # Strategy 5: Gemma-style function calls — explicit format
    #    <start_function_call>call:server/name{"arg": "val"}<end_function_call>
    # ==========================================================================
    if profile == "gemma":
        gemma_pattern = re.compile(
            r'<start_function_call>\s*call:([\w\-./]+)\s*(\{.*?\})?\s*<end_function_call>',
            re.DOTALL
        )
        for match in gemma_pattern.finditer(text):
            full_name = match.group(1).strip()
            args_raw = match.group(2) or ""
            call = {"id": "", "args": {}}
            if "/" in full_name:
                call["server"], call["tool"] = full_name.split("/", 1)
            elif NATIVE_NAME_SEP in full_name:
                call["server"], call["tool"] = full_name.split(NATIVE_NAME_SEP, 1)
            else:
                call["server"], call["tool"] = "unknown", full_name
            if args_raw.strip():
                try:
                    call["args"] = json.loads(args_raw)
                except json.JSONDecodeError as e:
                    call["parse_error"] = str(e)
            extracted.append(call)

    # ==========================================================================
    # Strategy 6: Bare arrays (not inside fences) — AMBIGUOUS
    #    [{"tool": "server/name", "arguments": {...}}, ...]
    # ==========================================================================
    array_pattern = re.compile(r'\[(\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*)\]', re.DOTALL)
    for match in array_pattern.finditer(text):
        array_content = match.group(0)  # Full array including brackets
        try:
            arr = json.loads(array_content)
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict):
                        parsed = _known(_parse_tool_json(item))
                        if parsed:
                            extracted.append(parsed)
        except json.JSONDecodeError:
            # Fall back to extracting individual objects from inside the array
            inner = match.group(1)
            json_objs = _extract_json_objects(inner)
            for obj_str in json_objs:
                try:
                    obj = json.loads(obj_str)
                    if isinstance(obj, dict):
                        parsed = _known(_parse_tool_json(obj))
                        if parsed:
                            extracted.append(parsed)
                except json.JSONDecodeError:
                    continue

    # ==========================================================================
    # Strategy 7: Python-style bare calls (not in XML) — AMBIGUOUS
    #    server/tool(arg1=val1, arg2=val2)
    # ==========================================================================
    bare_call_pattern = re.compile(
        r'(?<![\w])([a-zA-Z_][\w\-]*(?:[./][a-zA-Z_][\w\-]*)+)\s*\(([^)]*)\)',
        re.DOTALL
    )
    for match in bare_call_pattern.finditer(text):
        full_name = match.group(1).strip()
        args_str = match.group(2).strip()

        # Heuristic: only treat as tool call if the name contains / or __
        # (which is our tool naming convention)
        if "/" not in full_name and NATIVE_NAME_SEP not in full_name:
            continue

        parsed = _known(_parse_python_style_call(f"{full_name}({args_str})"))
        if parsed:
            extracted.append(parsed)

    # Deduplicate: same tool call may be found by multiple strategies
    seen = set()
    unique = []
    for tc in extracted:
        key = (tc["server"], tc["tool"], json.dumps(tc.get("args", {}), sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(tc)
    return unique

def detect_tool_call_intent(text: str) -> bool:
    """Detect if the model's text LOOKS like it was trying to make a tool call.

    This heuristic catches malformed or unclosed tool call patterns that
    our parsers failed to extract. Critical for turn continuity -- if the
    model tried to call a tool but we couldn't parse it, we need to tell
    the model and give it another chance instead of ending the turn.

    Returns True if the text contains patterns that suggest an attempted
    tool call that didn't parse correctly.
    """
    if not text:
        return False

    text = text.strip()
    indicators = 0

    # Pattern 1: Unclosed or malformed XML tool call tags
    if '<tool_call' in text:
        # Has opening tag but no closing tag (or vice versa)
        open_count = text.count('<tool_call>') + text.count('<tool_call ')
        close_count = text.count('</tool_call>')
        if open_count != close_count:
            return True
        # Tags are balanced but content between them might not have parsed
        if open_count > 0 and close_count > 0:
            indicators += 1

    # Pattern 2: Contains {"tool" or {"tool_calls"
    if '{"tool"' in text or '{"tool_calls"' in text or "{'tool'" in text:
        indicators += 1

    # Pattern 3: Bare function call with tool-like naming
    if re.search(r'[\w\-]+/[\w\-]+\s*\(', text):
        indicators += 1

    # Pattern 4: Contains both a tool name pattern AND argument-like content
    has_tool_pattern = bool(re.search(r'[\w\-]+/[\w\-]+', text))
    has_arg_content = '(' in text and ('=' in text or '""' in text)
    if has_tool_pattern and has_arg_content:
        indicators += 1

    # Pattern 5: Contains tool-related keywords in a structured context
    structured_indicators = [
        r'"tool"\s*:',
        r'"arguments"\s*:',
        r'"parameters"\s*:',
        r'"name"\s*:',
    ]
    for pattern in structured_indicators:
        if re.search(pattern, text):
            indicators += 1

    # If we see 2+ indicators, it's likely an attempted tool call
    return indicators >= 2

def build_format_correction_message(failed_text: str, registry: Dict[str, str] = None,
                                    reason: str = "") -> str:
    """Build a user-message telling the model its tool call wasn't accepted.

    Includes the specific reason (unparseable format, unknown tool name, or
    malformed JSON arguments) and the valid tool names so the model can
    correct itself instead of guessing.
    """
    parts = []
    if reason:
        parts.append(f"Problem: {reason}\n")
    parts.append(
        "I couldn't use your tool call. Please re-issue it in this exact format:\n"
        '```json\n'
        '{\n'
        '  "tool": "server/tool_name",\n'
        '  "arguments": {\n'
        '    "arg1": "value1",\n'
        '    "arg2": "value2"\n'
        '  }\n'
        '}\n'
        '```\n\n'
        "**Important:**\n"
        "- Use ONLY tool names from the Tools section, exactly as listed\n"
        "- The JSON must be valid: double-quoted keys/strings, balanced braces, no trailing commas\n"
        "- Escape quotes inside string values: \\\"\n"
    )
    if registry:
        canonical = sorted(set(registry.values()))
        shown = ", ".join(canonical[:20])
        parts.append(f"\nValid tools: {shown}")
        if len(canonical) > 20:
            parts.append(f" ... and {len(canonical) - 20} more")
        parts.append("\n")
    parts.append(
        "\nYour previous attempt was:\n"
        f"```\n{failed_text[:500]}{'...' if len(failed_text) > 500 else ''}\n```"
    )
    return "".join(parts)

# happens to start with '{' and mention "tool" in a prose explanation.
def _looks_like_tool_json(text: str) -> bool:
    """Return True if text is a raw tool call JSON block (not prose)."""
    stripped = text.strip()
    # Suppress raw JSON object starting with {"tool"...} or {"tool_calls"...}
    if stripped.startswith('{"tool"') or stripped.startswith('{"tool_calls"'):
        return True
    # Fenced code block: suppress only if it actually parses as a tool call.
    # A legitimate reply that merely STARTS with a fenced block (e.g. "here's
    # the code: ```...") must still be displayed and logged.
    if stripped.startswith("```"):
        for candidate in _extract_json_objects(stripped):
            try:
                obj = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict) and _parse_tool_json(obj) is not None:
                return True
        return False
    # Suppress XML-wrapped tool calls
    if stripped.startswith("<tool_call"):
        return True
    # Suppress Gemma-style function calls
    if stripped.startswith("<start_function_call>"):
        return True
    # Suppress bare arrays of tool calls
    if stripped.startswith('[{"tool"') or stripped.startswith('[{"name"'):
        return True
    return False

# Plan-mode marker appended to/removed from the system prompt. Defined once
# so toggling plan mode mid-session can reliably remove it again.
PLAN_MODE_MARKER = (
    "\n\n[PLAN MODE] You are in planning mode. Propose tool calls as you "
    "normally would, but they will be queued for review instead of executing "
    "immediately. Read-only tools (read_file, list_directory, search_files) "
    "will execute immediately to help you gather information for planning.\n"
)

# =============================================================================
# AGENT LOOP
# =============================================================================

class ForcePaused(Exception):
    """Raised inside the agent loop when the operator force-pauses (/pf)."""


async def race_abort(coro, state: Session):
    """Await *coro*, but give up the moment the operator force-pauses.

    The /pf handler already cancels the HTTP request and MCP calls so they
    return quickly; this is the safety net for anything that doesn't.
    """
    task = asyncio.ensure_future(coro)
    ev = getattr(state, "abort_event", None)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.1)
            if done:
                return task.result()
            if ev is not None and ev.is_set():
                task.cancel()
                await asyncio.wait({task}, timeout=2.0)
                raise ForcePaused()
    except OperatorCancelled:
        raise ForcePaused()
    except asyncio.CancelledError:
        task.cancel()
        raise


def _append_assistant(messages: list, text: str):
    """Append assistant text without creating two assistant turns in a row."""
    if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
        prev = messages[-1].get("content") or ""
        messages[-1]["content"] = (prev + "\n\n" + text).strip()
    else:
        messages.append({"role": "assistant", "content": text})


def _deliver_inbox(messages: list, state: Session, logger: SessionLogger):
    """Hand anything the operator typed mid-turn to the model."""
    items = drain_inbox(state)
    if not items:
        return
    text = "\n\n".join(items)
    content = f"[Mid-turn message from the operator]\n{text}"
    if messages and messages[-1].get("role") == "user" and isinstance(messages[-1].get("content"), str):
        messages[-1]["content"] += "\n\n" + content
    else:
        messages.append({"role": "user", "content": content})
    logger.log_user_message(text)
    print_system(f"  📨 delivered {len(items)} message(s) from you to OGneey")


def _park_inbox(state: Session):
    """Move undelivered mid-turn messages to the paused queue (/resume delivers)."""
    leftover = drain_inbox(state)
    for item in leftover:
        queue_user_message(state, item)
    return len(leftover)


def _gentle_pause_exit(state: Session, logger: SessionLogger):
    state.resume_pending = True
    parked = _park_inbox(state)
    logger.log_event("pause", {"kind": "gentle", "turn": state.turn_count})
    extra = f" · {parked} message(s) queued" if parked else ""
    print_system(f"\n⏸  Paused at a turn boundary{extra}. /r resumes · anything you type now is queued for /r")


def _force_pause_exit(messages: list, state: Session, logger: SessionLogger,
                      inflight: Optional[list] = None, text: str = ""):
    if inflight:
        names = ", ".join(f"{c.get('server')}/{c.get('tool')}" for c in inflight)
        note = (f"[Interrupted by the operator (force pause) while running: {names}. "
                f"Results unknown — some of these may have partly run.]")
        _append_assistant(messages, ((text or "") + "\n\n" + note).strip())
    else:
        _append_assistant(messages, "(Stopped by the operator before I finished responding.)")
    state.abort_event.clear()
    state.resume_pending = True
    parked = _park_inbox(state)
    if state.mode is not SessionState.PAUSED:
        transition_state(state, SessionState.PAUSED)
    logger.log_event("pause", {"kind": "force", "turn": state.turn_count,
                               "inflight": [f"{c.get('server')}/{c.get('tool')}" for c in (inflight or [])]})
    extra = f" · {parked} message(s) queued" if parked else ""
    print_warning(f"\n⏹  Stopped{extra}. Session is paused — /r to resume, or type a message first (delivered on /r).")
    return True


def _show_reasoning(state: Session, logger: SessionLogger, reasoning: str):
    if not reasoning:
        return
    state.last_reasoning = reasoning
    logger.log_reasoning(state.turn_count, reasoning)
    mode = getattr(state, "think_mode", "short")
    if mode == "off":
        return
    if mode == "full":
        body = reasoning.strip().replace("\n", "\n     ")
        print_dim(f"  💭 Thinking:\n     {body}")
        return
    short = reasoning[:300]
    more = len(reasoning) - len(short)
    tail = f"… (+{more:,} chars · /think last to expand · /think full to always expand)" if more > 0 else ""
    print_dim(f"  💭 Thinking: {short}{tail}")


async def agent_loop(
    task: str,
    mcp: MCPManager,
    state: Session,
    logger: SessionLogger,
    system_prompt: str,
    messages: list = None,
    new_task: bool = True,
) -> bool:
    """
    Main agent loop with MCP tool execution.

    Returns True if task completed (or was paused) cleanly, False on error.

    Operator control points (input arrives on another thread while this runs):
      * top of every iteration: /pf (abort), /p (pause), mid-turn messages
      * the LLM request and the tool batch are raced against /pf
      * after tool results are safely in history: /pf, /p
    """
    if new_task and hasattr(state, "gate_exceptions"):
        state.gate_exceptions.setdefault("task", set()).clear()

    # Initialize messages if not provided
    if messages is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]
        # One-shot path: the system prompt has no tool list yet — inject the
        # profile-appropriate tools section (interactive mode injects it once
        # at session start instead).
        if state.tools_enabled and "## Tools" not in system_prompt:
            profile = detect_tool_profile(state)
            messages[0]["content"] += mcp.get_tools_prompt(
                profile=profile, extra_tools=get_local_tools()
            )
    else:
        # Append new task (merge if the history already ends on a user turn,
        # e.g. after a force pause)
        if messages and messages[-1].get("role") == "user" and isinstance(messages[-1].get("content"), str):
            messages[-1]["content"] += "\n\n" + task
        else:
            messages.append({"role": "user", "content": task})

    # Inject custom prompt if any
    if hasattr(state, 'custom_prompt') and state.custom_prompt:
        if "## Additional Instructions" not in messages[0]["content"]:
            messages[0]["content"] += f"\n\n## Additional Instructions\n{state.custom_prompt}"

    # Log user message
    logger.log_user_message(task)

    # Handle plan mode toggle (tools already in system prompt from session
    # init). The marker is added when plan mode is on AND removed when it is
    # off, so mid-session /plan and /plan-off leave the prompt consistent.
    if state.tools_enabled:
        if state.plan_mode:
            if "[PLAN MODE]" not in messages[0]["content"]:
                messages[0]["content"] += PLAN_MODE_MARKER
        else:
            messages[0]["content"] = messages[0]["content"].replace(PLAN_MODE_MARKER, "")
    else:
        if "[TOOLS DISABLED]" not in messages[0]["content"]:
            messages[0]["content"] += "\n\n[TOOLS DISABLED] You are in chat-only mode. Do NOT output any tool calls, JSON blocks, or code blocks with tool invocations. Just respond conversationally.\n"

    # ==========================================================================
    # CHAT-ONLY MODE: Single turn when tools fully disabled
    # ==========================================================================
    if not state.tools_enabled:
        try:
            print_dim(f"\n--- Chat Mode (1 turn) ---")
            print_ognee_thinking()
            response = await race_abort(chat(messages), state)

            # Extract text and reasoning
            text = extract_response_text(response)
            reasoning = extract_reasoning_content(response)
            _show_reasoning(state, logger, reasoning)

            # Update tokens
            usage = response.get("usage", {})
            if usage:
                update_token_count(
                    state,
                    messages=messages,
                    api_usage=usage
                )
                print_dim(f"  📊 Tokens: {usage.get('total_tokens', '?')}")

            state.turn_count += 1

            if text:
                messages.append({"role": "assistant", "content": text})
                if not _looks_like_tool_json(text):
                    print_ognee_response(text)
                    logger.log_assistant_message(text, usage.get("total_tokens", 0))
            else:
                print_dim("  (no text response)")

            return True
        except ForcePaused:
            return _force_pause_exit(messages, state, logger)
        except KeyboardInterrupt:
            print_warning("\n⏸ Interrupted")
            return True
        except Exception as e:
            print_error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ==========================================================================
    # AGENTIC MODE: Multi-turn with tool execution
    # ==========================================================================
    MAX_FORMAT_CORRECTIONS = 3  # Don't burn iterations on a model that can't learn the format
    format_correction_count = 0  # resets per user turn (agent_loop is per-task)

    # Per-model tool profile gates BOTH the prompt format and the parser.
    profile = detect_tool_profile(state)
    native_mode = profile == "native"
    registry = build_tool_registry(mcp)
    loop_detector = LoopDetector()
    loop_notice_sent = False  # deliver the "you're stuck" notice once
    last_tool_count = None

    if not native_mode:
        print_dim(f"  🔧 Tool profile: {profile} (text-parsed)")

    for iteration in range(MAX_ITERATIONS):
        # ---- operator control points --------------------------------------
        if state.abort_event.is_set():
            return _force_pause_exit(messages, state, logger)
        if state.paused:
            _gentle_pause_exit(state, logger)
            return True
        _deliver_inbox(messages, state, logger)

        # Check context limits
        can_continue, stop_reason = check_context_limits(state)
        if not can_continue:
            # Hard stop - save continuation and exit
            summary = "Session ended due to context limit."
            save_continuation_scroll(state, summary)
            return False

        # Inject state if needed (mutates messages[0] in-place — no copy needed)
        if should_inject_state(state):
            inject_state_into_messages(messages, state)
        messages_with_state = messages

        # Native tool schemas are rebuilt every turn so /disable and /enable
        # take effect immediately (disabled tools are not sent at all).
        native_tools = build_native_tools(mcp, state) if native_mode else None
        if native_mode:
            total = len(mcp.all_tools) + len(get_local_tools())
            if native_tools is not None and len(native_tools) != last_tool_count:
                hidden = total - len(native_tools)
                extra = f" — {hidden} disabled" if hidden > 0 else ""
                print_dim(f"  🔧 Tool profile: native ({len(native_tools)} of {total} tools via API{extra})"
                          if hidden > 0 else f"  🔧 Tool profile: native ({len(native_tools)} tools via API)")
                last_tool_count = len(native_tools)

        print_dim(f"\n--- Turn {iteration + 1}/{MAX_ITERATIONS} ---")

        inflight = None
        text = ""
        try:
            # Call LLM — with native tool schemas when the model profile
            # supports them, otherwise rely on the profiled text format.
            print_ognee_thinking()
            response = await race_abort(chat(messages_with_state, tools=native_tools), state)

            # Extract response text and reasoning
            text = extract_response_text(response)
            reasoning = extract_reasoning_content(response)
            _show_reasoning(state, logger, reasoning)

            # Update token count
            usage = response.get("usage", {})
            if usage:
                update_token_count(
                    state,
                    messages=messages_with_state,
                    api_usage=usage
                )
                logger.log_token_usage(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0)
                )
                print_dim(f"  📊 Tokens: {usage.get('total_tokens', '?')}")

            state.turn_count += 1

            if state.abort_event.is_set():
                # The reply landed just as the operator hit /pf: honor the stop.
                return _force_pause_exit(messages, state, logger)

            # Surface completion-cap truncation to the model instead of
            # silently failing to parse a response that was cut mid-JSON.
            finish_reason = ""
            try:
                finish_reason = response["choices"][0].get("finish_reason", "") or ""
            except (KeyError, IndexError, AttributeError):
                pass
            if finish_reason == "length":
                print_warning("  ⚠️ Model output hit the max_tokens cap (truncated)")
                messages.append({"role": "assistant", "content": text or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "SYSTEM NOTICE: your previous response was truncated because it hit "
                        "the maximum output length. Whatever you were producing did NOT complete "
                        "and any tool call in it was NOT executed. Retry with smaller chunks "
                        "(e.g. write files in parts, narrow the request, or split the work "
                        "across multiple tool calls)."
                    ),
                })
                _trim_messages(messages, state)
                continue

            # Try native tool calls first; text extraction is the fallback for
            # non-native model profiles only.
            tool_calls = extract_tool_calls_native(response, registry)
            native_response = bool(tool_calls)
            if not tool_calls and text and not native_mode:
                tool_calls = extract_tool_calls_text(text, registry, profile=profile)

            # --------------------------------------------------------------
            # Validation gate: registry names only, no malformed-arg munging.
            # Rejected calls are never executed and never reach the failure
            # tracker — the model gets a correction message instead.
            # --------------------------------------------------------------
            valid_calls = []
            rejected_reasons = []
            for tc in tool_calls:
                if tc.get("parse_error"):
                    rejected_reasons.append(
                        f"malformed JSON arguments ({tc['parse_error']})"
                    )
                    continue
                full = resolve_tool_name(tc.get("server", ""), tc.get("tool", ""), registry)
                if not full:
                    rejected_reasons.append(
                        f"unknown tool '{tc.get('server', '')}/{tc.get('tool', '')}'"
                    )
                    continue
                tc["server"], tc["tool"] = full.split("/", 1)
                valid_calls.append(tc)

            if rejected_reasons:
                format_correction_count += 1
                reason = "; ".join(rejected_reasons[:3])
                if format_correction_count <= MAX_FORMAT_CORRECTIONS:
                    correction = build_format_correction_message(
                        text or "(native tool call)", registry, reason=reason
                    )
                    messages.append({"role": "assistant", "content": text or ""})
                    messages.append({"role": "user", "content": correction})
                    _trim_messages(messages, state)
                    print_warning(f"  ⚠️ Rejected tool call ({reason}) — correction {format_correction_count}/{MAX_FORMAT_CORRECTIONS}")
                    continue
                else:
                    print_warning(f"  ❌ {MAX_FORMAT_CORRECTIONS} corrections failed. Asking model to respond in plain text.")
                    messages.append({"role": "assistant", "content": text or ""})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your tool calls still aren't valid after multiple attempts. "
                            "Please respond in plain text instead — describe what you want "
                            "to do and I'll help you figure out the right approach."
                        ),
                    })
                    _trim_messages(messages, state)
                    continue
            tool_calls = valid_calls

            # Check for tool call intent even if parsing failed (text profiles
            # only — native responses don't need heuristic rescue).
            if not tool_calls and text and not native_mode and detect_tool_call_intent(text):
                format_correction_count += 1
                if format_correction_count <= MAX_FORMAT_CORRECTIONS:
                    # Model tried to call a tool but we couldn't parse it — help it
                    correction = build_format_correction_message(text, registry)
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": correction})
                    _trim_messages(messages, state)
                    print_warning(f"  ⚠️ Format correction {format_correction_count}/{MAX_FORMAT_CORRECTIONS}")
                    continue  # Loop again, giving model a chance to retry
                else:
                    # Model can't learn the format — stop wasting iterations
                    print_warning(f"  ❌ {MAX_FORMAT_CORRECTIONS} format corrections failed. Model can't produce valid tool calls.")
                    messages.append({"role": "assistant", "content": text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your tool calls still aren't parsing after multiple attempts. "
                            "Please respond in plain text instead — describe what you want to do "
                            "and I'll help you figure out the right approach."
                        )
                    })
                    _trim_messages(messages, state)
                    continue  # Give model one last chance to respond conversationally

            if tool_calls:
                # A turn with valid parsed calls resets the format-correction
                # counter — the model demonstrably can produce the format.
                format_correction_count = 0

                # Loop guard: identical (tool, args) repeated N times in a row
                # means the model is stuck. Stop and tell it, instead of
                # burning iterations (and spawning subprocess work).
                looped = None
                for tc in tool_calls:
                    if loop_detector.check(f"{tc['server']}/{tc['tool']}", tc.get("args", {})):
                        looped = tc
                        break
                if looped is not None:
                    loop_key = f"{looped['server']}/{looped['tool']}"
                    print_warning(f"  🔁 Loop detected: {loop_key} called {LOOP_GUARD_MAX_REPEATS}x with identical args — stopping")
                    if loop_notice_sent:
                        # Already delivered the notice once and the model kept
                        # looping anyway — end the turn.
                        logger.log_event("loop_detected_stop", {"tool": loop_key})
                        return True
                    loop_notice_sent = True
                    messages.append({"role": "assistant", "content": text or ""})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"SYSTEM NOTICE: you called {loop_key} with identical arguments "
                            f"{LOOP_GUARD_MAX_REPEATS} times in a row. Repeating it will not "
                            "produce a different result. Do NOT call it again — instead, "
                            "summarize what you have so far and explain what is blocking you, "
                            "or try a fundamentally different approach."
                        ),
                    })
                    _trim_messages(messages, state)
                    logger.log_event("loop_detected", {"tool": loop_key})
                    # Continue the loop so the model actually RECEIVES the
                    # notice and gets one chance to recover.
                    loop_detector.reset()
                    continue
            else:
                loop_detector.reset()

            # Execute tool calls if any
            # FIX: Don't reset _last_turn_had_tools here. The flag is set when
            # tools execute and cleared only after the nudge check (in the else
            # branch) or after a non-empty response. Resetting it here meant the
            # nudge never fired because the flag was always False by check time.
            tool_results = []

            queued_write_calls = []

            if tool_calls and state.tools_enabled:
                inflight = tool_calls
                if state.plan_mode:
                    # PLAN MODE: Execute read-only tools, queue write tools.
                    # tool_results stays aligned 1:1 with tool_calls so
                    # zip() never mispairs results (and every native
                    # tool_call_id gets a matching role:"tool" message).
                    read_calls = [tc for tc in tool_calls if tc["tool"] in READ_ONLY_TOOLS]
                    write_calls = [tc for tc in tool_calls if tc["tool"] not in READ_ONLY_TOOLS]

                    read_results = []
                    if read_calls:
                        print_dim(f"  📖 Executing {len(read_calls)} read-only tool(s)...")
                        read_results = await race_abort(execute_tool_calls(
                            mcp_clients=mcp.servers,
                            tool_calls=read_calls,
                            state=state,
                            iteration=iteration
                        ), state)

                        # Log read tool calls
                        for tc, result in zip(read_calls, read_results):
                            logger.log_tool_call(
                                server=tc["server"],
                                tool=tc["tool"],
                                args=tc["args"],
                                result=result.output or result.error or "",
                                success=result.success,
                                duration_ms=result.duration_ms
                            )

                    # Queue write tools for later execution
                    if write_calls:
                        from .tools import queue_tool_call
                        for tc in write_calls:
                            queue_tool_call(state, tc["server"], tc["tool"], tc["args"])
                            queued_write_calls.append(tc)
                        print_system(f"  📋 Queued {len(write_calls)} tool(s) for review. Use /execute to run, /clear to discard.")

                    result_by_call = {id(tc): res for tc, res in zip(read_calls, read_results)}
                    tool_results = []
                    for tc in tool_calls:
                        res = result_by_call.get(id(tc))
                        if res is None:
                            res = ToolResult(
                                success=True,
                                output="[PLAN MODE] Queued for review — NOT executed yet.",
                                duration_ms=0,
                            )
                        tool_results.append(res)
                else:
                    # NORMAL MODE: Execute all tools
                    state._last_turn_had_tools = True
                    tool_results = await race_abort(execute_tool_calls(
                        mcp_clients=mcp.servers,
                        tool_calls=tool_calls,
                        state=state,
                        iteration=iteration
                    ), state)

                    # Log tool calls
                    for tc, result in zip(tool_calls, tool_results):
                        logger.log_tool_call(
                            server=tc["server"],
                            tool=tc["tool"],
                            args=tc["args"],
                            result=result.output or result.error or "",
                            success=result.success,
                            duration_ms=result.duration_ms
                        )

                        # Global tool log
                        log_tool_to_global(
                            session_id=state.session_id,
                            server=tc["server"],
                            tool=tc["tool"],
                            args=tc["args"],
                            result=result.output or result.error or "",
                            success=result.success,
                            duration_ms=result.duration_ms
                        )
                inflight = None

            # NOTE: there used to be an `if state.paused: return` right here —
            # it threw away the results of tools that had just run, so the
            # model never saw them. Pause is now honored AFTER the results are
            # appended to history (below).

            if text:
                if not _looks_like_tool_json(text):
                    print_ognee_response(text)
                    logger.log_assistant_message(text, usage.get("total_tokens", 0))
            else:
                print_dim("  (no text response)")

            # Build next message based on what happened
            if tool_calls and tool_results:
                # Tools were used — feed results back and loop.
                # Zip over executed calls only: plan mode truncates
                # tool_calls to read_calls before we get here, and results
                # are returned in call order.
                if native_response:
                    # Native mode: proper assistant.tool_calls + role:"tool"
                    # messages with tool_call_id, so the model's fine-tuned
                    # tool-loop behavior actually engages.
                    raw_msg = response.get("choices", [{}])[0].get("message", {})
                    assistant_msg = {
                        "role": "assistant",
                        "content": text or None,
                        "tool_calls": raw_msg.get("tool_calls", []),
                    }
                    messages.append(assistant_msg)
                    for tr_msg in format_tool_results_for_llm(tool_results, tool_calls):
                        if len(tr_msg.get("content", "")) > MAX_TOOL_RESULT_CHARS:
                            tr_msg["content"] = (
                                tr_msg["content"][:MAX_TOOL_RESULT_CHARS]
                                + f"\n\n[TRUNCATED: exceeded {MAX_TOOL_RESULT_CHARS} characters.]"
                            )
                        messages.append(tr_msg)
                else:
                    # Text-fallback profiles: results go back in one clearly
                    # delimited user-role block.
                    results_text = []
                    for tc, tr in zip(tool_calls, tool_results):
                        tool_key = f"{tc['server']}/{tc['tool']}"
                        if tr.success:
                            body = tr.output
                            if len(body) > MAX_TOOL_RESULT_CHARS:
                                body = body[:MAX_TOOL_RESULT_CHARS] + f"\n\n[TRUNCATED: Result exceeded {MAX_TOOL_RESULT_CHARS} characters. Full output stored internally.]"
                            results_text.append(f"=== {tool_key} (success) ===\n{body}")
                        else:
                            body = tr.error or ""
                            if len(body) > MAX_TOOL_RESULT_CHARS:
                                body = body[:MAX_TOOL_RESULT_CHARS] + f"\n\n[TRUNCATED: Error exceeded {MAX_TOOL_RESULT_CHARS} characters.]"
                            results_text.append(f"=== {tool_key} (FAILED) ===\n{body}")

                    # If write tools were queued in plan mode, inform the LLM
                    # so it knows what is pending review.
                    queued_note = ""
                    if queued_write_calls:
                        queued_list = ", ".join(f"{q['server']}/{q['tool']}" for q in queued_write_calls)
                        queued_note = (
                            f"\n\n[PLAN MODE] The following write/modify tools are queued for review "
                            f"and have NOT been executed yet: {queued_list}. "
                            f"Use /execute to run them or /clear to discard."
                        )

                    messages.append({"role": "assistant", "content": text or "Using tools..."})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[TOOL RESULTS]\n\n" + "\n\n".join(results_text)
                            + "\n[/TOOL RESULTS]" + queued_note
                            + "\n\nSummarize what you found and your next steps in plain text "
                              "before making additional tool calls. If you need more data, "
                              "explain what you're looking for and why."
                        )
                    })

                # Trim conversation to fit token budget
                _trim_messages(messages, state)

                # Results are safely in history — now honor stop/pause.
                if state.abort_event.is_set() or any(getattr(r, "aborted", False) for r in tool_results):
                    state.abort_event.set()
                    return _force_pause_exit(messages, state, logger)
                if state.paused:
                    _gentle_pause_exit(state, logger)
                    return True

                # Loop continues — model gets tool results and decides next step
            else:
                # No tool calls — model chose to respond conversationally.
                # Turn is complete. Record and exit.
                messages.append({"role": "assistant", "content": text})

                _trim_messages(messages, state)


                # FIX: If the model produced empty text after receiving tool results,
                # it may have stalled instead of summarizing or continuing. Nudge once.
                if not text and getattr(state, "_last_turn_had_tools", False):
                    state._last_turn_had_tools = False  # Reset to prevent infinite looping
                    messages.append({
                        "role": "user",
                        "content": "You received tool results but produced no response. Please summarize what you found and state your next steps, or make additional tool calls if needed."
                    })
                    _trim_messages(messages, state)
                    continue  # Loop one more time instead of returning
                return True

        except ForcePaused:
            return _force_pause_exit(messages, state, logger, inflight=inflight, text=text)
        except requests.exceptions.RequestException as e:
            print_error(f"\n❌ API Error: {e}")
            logger.log_event("error", {"type": "api_error", "message": str(e)})
            return False
        except KeyboardInterrupt:
            print_warning("\n\n⏹️ Interrupted by user")
            logger.log_event("interrupted", {})
            return False
        except Exception as e:
            print_error(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            logger.log_event("error", {"type": "exception", "message": str(e)})
            return False

    print_warning(f"\n⚠️ Max iterations ({MAX_ITERATIONS}) reached")
    logger.log_event("max_iterations", {"iterations": MAX_ITERATIONS})
    return False

# =============================================================================
# INTERACTIVE MODE
# =============================================================================

async def execute_plan_queue(
    plan: List[Dict[str, Any]],
    mcp: MCPManager,
    state: Session,
    logger: SessionLogger,
) -> List[ToolResult]:
    """Run a queued plan through the normal gated execution path.

    Unlike the old interactive-loop shortcut, results here are printed,
    recorded in session state, AND written to the session/global logs.
    ``/execute`` narration (feeding results back to the model) is the
    caller's choice.
    """
    # Strip bookkeeping keys (e.g. queued_at) so tools receive clean args.
    tool_calls = [
        {
            "id": step.get("id", ""),
            "server": step["server"],
            "tool": step["tool"],
            "args": step.get("args", {}),
        }
        for step in plan
    ]
    tool_results = await execute_tool_calls(
        mcp_clients=mcp.servers,
        tool_calls=tool_calls,
        state=state,
        iteration=0
    )
    for tc, result in zip(tool_calls, tool_results):
        body = result.output or result.error or ""
        print_tool_result(body[:500], success=result.success)
        logger.log_tool_call(
            server=tc["server"],
            tool=tc["tool"],
            args=tc["args"],
            result=body,
            success=result.success,
            duration_ms=result.duration_ms
        )
        log_tool_to_global(
            session_id=state.session_id,
            server=tc["server"],
            tool=tc["tool"],
            args=tc["args"],
            result=body,
            success=result.success,
            duration_ms=result.duration_ms
        )
    state.pending_plan = []
    print_success(f"✓ Executed {len(tool_results)} planned step(s)")
    return tool_results


RESUME_NUDGE = (
    "[The operator resumed the session. Continue the task from where you left off. "
    "If you're not sure what's next, or you were waiting on something, say so.]"
)

from .commands import BUSY_REFUSED  # noqa: E402  (single source of truth)


class AgentRunner:
    """Runs agent work on its own thread with a persistent event loop, so the
    main thread can keep reading the keyboard (that's what makes /p, /pf,
    /tier and mid-turn messages work while OGneey is busy)."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=32, thread_name_prefix="ognee-io")
        self.loop.set_default_executor(self.executor)
        self._thread = threading.Thread(target=self._run, name="ognee-agent", daemon=True)
        self._future: Optional[concurrent.futures.Future] = None
        self._lock = threading.Lock()
        self.on_idle = None

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self):
        self._thread.start()

    @property
    def busy(self) -> bool:
        f = self._future
        return f is not None and not f.done()

    def submit(self, coro) -> concurrent.futures.Future:
        with self._lock:
            if self.busy:
                coro.close()
                raise RuntimeError("agent is busy")
            fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
            self._future = fut
        fut.add_done_callback(self._finished)
        return fut

    def _finished(self, fut):
        cb = self.on_idle
        if cb:
            try:
                cb(fut)
            except Exception:
                pass

    def wait(self, timeout: float = None):
        f = self._future
        if f is None:
            return
        try:
            f.result(timeout=timeout)
        except Exception:
            pass

    def stop(self):
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError:
            pass
        self._thread.join(timeout=3)
        self.executor.shutdown(wait=False, cancel_futures=True)


async def _run_turns(text: str, mcp: MCPManager, state: Session, logger: SessionLogger,
                     system_prompt: str, messages: list, new_task: bool = True):
    """One operator request, plus anything typed while it was finishing."""
    try:
        task = text
        fresh = new_task
        while True:
            await agent_loop(task=task, mcp=mcp, state=state, logger=logger,
                             system_prompt=system_prompt, messages=messages, new_task=fresh)
            fresh = False
            if state.abort_event.is_set() or state.paused or state.quit_requested:
                break
            follow = drain_inbox(state)
            if not follow:
                break
            print_system(f"  📨 delivering {len(follow)} message(s) you typed while OGneey was finishing")
            task = "[Message from the operator, typed while you were finishing]\n" + "\n\n".join(follow)
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def _run_plan(plan, mcp, state, logger, system_prompt, messages):
    try:
        results = await race_abort(execute_plan_queue(plan, mcp, state, logger), state)
    except ForcePaused:
        state.abort_event.clear()
        print_warning("⏹  Plan execution stopped.")
        return
    if not results:
        return
    summary_lines = [
        f"{'✓' if r.success else '✗'} {tc['server']}/{tc['tool']}: {(r.output or r.error or '')[:200]}"
        for tc, r in zip(plan, results)
    ]
    narration = ("[PLAN EXECUTED] The queued plan was executed. Results:\n"
                 + "\n".join(summary_lines) + "\nPlease summarize the outcome for the user.")
    await _run_turns(narration, mcp, state, logger, system_prompt, messages, new_task=False)


def _tool_names(mcp: MCPManager) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"local": [t["name"] for t in get_local_tools()]}
    for t in mcp.all_tools:
        out.setdefault(t.get("server", "?"), []).append(t.get("name", ""))
    return out


def _tool_catalog(mcp: MCPManager) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {
        "local": [{"name": t["name"], "description": t.get("description", "")} for t in get_local_tools()]
    }
    for t in mcp.all_tools:
        out.setdefault(t.get("server", "?"), []).append(
            {"name": t.get("name", ""), "description": t.get("description", "") or ""})
    for name in mcp.servers:
        out.setdefault(name, [])
    return out


def interactive_session(mcp: MCPManager, state: Session, logger: SessionLogger, system_prompt: str):
    """Interactive REPL. The keyboard stays live while OGneey works.

    Main thread: reads lines, runs slash commands, answers approval prompts.
    Agent thread: runs the turns. Plain text typed mid-turn is delivered at
    the next turn boundary; /p pauses at the boundary, /pf (or Ctrl-C) stops
    immediately.
    """
    from .commands import (execute_command, CommandResult, canonical_command,
                           command_names, command_arg_choices)

    hub = get_hub()
    runner = AgentRunner()

    # Reset the tool output buffer for this session
    reset_output_buffer()

    # Track conversation messages
    messages = [{"role": "system", "content": system_prompt}]

    # Inject tools into system prompt once at session start. ONE format
    # section, generated from the model's tool profile (native models get
    # their schemas via the API instead of a prompt block).
    if state.tools_enabled:
        profile = detect_tool_profile(state)
        tools_prompt = mcp.get_tools_prompt(profile=profile, extra_tools=get_local_tools())
        # Inject read_output usage reminder for large outputs
        tools_prompt += (
            "\n**local/read_output**: If a previous tool result says "
            "[OUTPUT STORED — Reference: out_XXXX ...], call "
            "local/read_output with ref=\"out_XXXX\" to paginate through it.\n"
        )
        if "## Tools" not in messages[0]["content"]:
            messages[0]["content"] += tools_prompt

    # Inject custom prompt if any
    if hasattr(state, 'custom_prompt') and state.custom_prompt:
        messages[0]["content"] += f"\n\n## Additional Instructions\n{state.custom_prompt}"

    # ---- wiring --------------------------------------------------------------
    state.available_tools = _tool_names(mcp)
    state.tool_catalog_provider = lambda: _tool_catalog(mcp)

    def _gate_event(ev: Dict[str, Any]):
        clean = {k: (redact_secrets(v) if isinstance(v, str) else v) for k, v in ev.items()}
        logger.log_event("gate", clean)
    state.gate_event_hook = _gate_event

    def _on_force_pause():
        cancel_inflight_llm_requests()
        try:
            mcp.cancel_all_inflight()
        except Exception:
            pass
        hub.cancel_pending("abort")
    state.force_pause_hooks.append(_on_force_pause)

    def prompt_provider() -> str:
        if runner.busy:
            if state.mode is SessionState.PAUSED:
                return f"{COLORS.WARNING}(pausing at the next boundary · /pf stops now){COLORS.RESET} › "
            return f"{COLORS.DIM}(working · /p pause · /pf stop){COLORS.RESET} › "
        if state.mode is SessionState.PAUSED:
            return f"{COLORS.WARNING}(paused · /r resumes){COLORS.RESET} › "
        if state.mode is SessionState.PLAN_REVIEW:
            return f"{COLORS.SYSTEM}(plan){COLORS.RESET} › "
        return f"{COLORS.COMMAND}›{COLORS.RESET} "

    def _idle(_fut):
        state.agent_busy = False
        hub.refresh_prompt()
    runner.on_idle = _idle

    def start(coro):
        state.abort_event.clear()
        state.agent_busy = True
        runner.submit(coro)
        hub.refresh_prompt()

    def start_task(text: str, new_task: bool = True):
        start(_run_turns(text, mcp, state, logger, system_prompt, messages, new_task))

    def run_command_line(line: str) -> str:
        """Execute a slash command on the main thread; returns its canonical name."""
        name = canonical_command(line)
        busy = runner.busy
        if busy and name in BUSY_REFUSED:
            print_warning(f"  ⏳ /{name} can't run while OGneey is working — "
                          f"/p (pause at the boundary) or /pf (stop now) first.")
            return name
        result = execute_command(line, state)
        if result.result == CommandResult.QUIT:
            state.quit_requested = True
            if busy or hub.pending() is not None:
                request_force_pause(state)
            return "quit"
        data = result.data or {}
        if data.get("force_pause"):
            if busy or hub.pending() is not None:
                request_force_pause(state)
            else:
                print_system("  Nothing is running. (/p pauses an idle session; it's already idle.)")
        elif "inject_user_message" in data:
            msg = data["inject_user_message"]
            if busy:
                post_inbox(state, msg)
                print_dim("  📥 queued for the next turn boundary")
            elif state.mode is SessionState.PAUSED:
                queue_user_message(state, msg)
                print_dim("  📥 queued — /r delivers it")
            else:
                start_task(msg)
        elif "execute_plan" in data:
            start(_run_plan(data["execute_plan"], mcp, state, logger, system_prompt, messages))
        elif "ssh_command" in data:
            # /ssh goes through the same gate/approval path as other shell
            # tools — no bypass.
            from .ssh import run_ssh_command
            ssh_data = data["ssh_command"]
            allowed, _ = check_and_enforce_ssh(ssh_data["host"], ssh_data["command"], state)
            if allowed:
                ssh_result = run_ssh_command(ssh_data["host"], ssh_data["command"])
                if ssh_result.success:
                    print(ssh_result.stdout)
                else:
                    print_error(ssh_result.stderr)
        elif data.get("resume"):
            queued = drain_queued_messages(state)
            continuing = state.resume_pending
            state.resume_pending = False
            if busy:
                pass        # un-paused before the boundary: the turn just keeps going
            elif queued:
                for msg in queued:
                    print_system(f"  📤 Delivering queued message: {msg[:80]}")
                text = "\n\n".join(queued)
                if continuing:
                    text = RESUME_NUDGE + "\n\n" + text
                start_task(text, new_task=not continuing)
            elif continuing:
                start_task(RESUME_NUDGE, new_task=False)
        hub.refresh_prompt()
        return name

    hub.commands_provider = command_names
    hub.args_provider = lambda cmd: command_arg_choices(cmd, state)
    hub.command_executor = run_command_line
    hub.start(history_path=INPUT_HISTORY_FILE, prompt_provider=prompt_provider, threaded=True)
    runner.start()

    # Get model info
    model_name = state.model_name
    tier_display = f"{COLORS.SYSTEM}{get_tier_name(state)}{COLORS.RESET} ({state.trust_tier}, tier {get_tier_level(state)})"
    tool_count = f"{len(mcp.all_tools)} MCP + {len(get_local_tools())} local"
    print(f"""
╭{"─" * (INNER_WIDTH)}╮
{row("OGnee Interactive Mode")}
{row("")}
{row(f"Model: {model_name}")}
{row(f"Trust: {tier_display}")}
{row(f"Tools: {tool_count}")}
{row("")}
{row("Type while it works: /p pause · /pf stop now · /tier N · plain text = steer")}
{row("Commands: /help /status /tools /think /exceptions /policy /quit")}
╰{"─" * (INNER_WIDTH)}╯
""")
    print_dim(f"  ⌨  input: {hub.describe()}")
    if hub.note:
        print_warning(f"  ⚠  {hub.note}")
    pol = get_policy()
    pol_src = "built-in defaults" if pol.source == "builtin" else pol.source
    print_dim(f"  🛡  shell policy: {pol_src} ({len(pol.deny)} deny · {len(pol.confirm)} confirm · "
              f"{len(pol.allow)} allow · match={pol.match} · default={pol.default})")
    if pol.errors:
        print_error(f"  ⚠  shell policy has errors — ALL shell commands are blocked until fixed: {pol.errors[0]}")

    last_ctrl_c = 0.0
    last_status_turn = 0
    try:
        while True:
            if state.quit_requested:
                runner.wait(10)
                break
            if (not runner.busy and state.turn_count >= last_status_turn + 5):
                last_status_turn = state.turn_count - (state.turn_count % 5)
                print_status_line(state.to_status())
            try:
                line = hub.read_line()
            except KeyboardInterrupt:
                print()
                if hub.pending() is not None or runner.busy:
                    print_warning("  ⏹  Ctrl-C — stopping the current turn (same as /pf)")
                    request_force_pause(state)
                    continue
                now = time.monotonic()
                if now - last_ctrl_c < 2.0:
                    break
                last_ctrl_c = now
                print_dim("  (Ctrl-C again within 2s — or /q — to quit)")
                continue
            except EOFError:
                print()
                if runner.busy:
                    request_force_pause(state)
                    runner.wait(10)
                break

            if hub.pending() is not None:
                hub.deliver(line)
                continue

            stripped = line.strip()
            if not stripped:
                continue
            if hub.recently_resolved() and hub.looks_like_gate_answer(stripped):
                print_dim("  (that approval was already resolved — ignored)")
                continue

            # ---- Slash commands: handled here, NEVER sent to the LLM ----
            if is_command(stripped):
                run_command_line(stripped)
                continue

            # ---- Regular input ----
            if runner.busy:
                depth = post_inbox(state, stripped)
                print_system(f"  📥 got it — OGneey sees this at the next turn boundary ({depth} waiting)")
                continue
            if state.mode is SessionState.PAUSED:
                depth = queue_user_message(state, stripped)
                print_system(
                    f"  📥 Paused — message queued ({depth} pending). "
                    f"It will be delivered on /resume."
                )
                continue

            print()  # move off the input line before printing output
            start_task(stripped)
    finally:
        # From here on we're leaving: a mashed Ctrl-C must not interrupt the
        # cleanup (it used to be able to skip stopping MCP servers).
        _ignore_sigint()
        if runner.busy:
            request_force_pause(state)
            runner.wait(10)
        runner.stop()
        hub.stop()


def _ignore_sigint():
    """Ignore Ctrl-C for the rest of the process (used only while exiting)."""
    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            pass

# =============================================================================
# ONE-SHOT MODE
# =============================================================================

async def one_shot(
    task: str,
    mcp: MCPManager,
    state: Session,
    logger: SessionLogger,
    system_prompt: str
) -> bool:
    """
    Execute a single task and exit.
    Good for scripted usage.
    """
    return await agent_loop(
        task=task,
        mcp=mcp,
        state=state,
        logger=logger,
        system_prompt=system_prompt
    )

# =============================================================================
# MAIN ENTRY POINTS
# =============================================================================

def run_interactive(
    model_name: str = None,
    trust_level: int = DEFAULT_TRUST_TIER,
    config_path: str = None
):
    """
    Run OGnee in interactive mode.
    """
    # Detect model if not specified
    if not model_name:
        model_name = get_current_model()

    # Load MCP
    mcp = MCPManager(config_path)
    if not mcp.load_config():
        print_error("Failed to load MCP config")
        return

    if mcp.start_all() == 0:
        print_warning("No MCP servers started. Continuing with local tools only.")

    try:
        # Startup sequence
        state, logger, system_prompt = startup(
            model_name=model_name,
            trust_level=trust_level
        )

        try:
            # Run interactive loop (keyboard on this thread, agent on a worker)
            interactive_session(mcp, state, logger, system_prompt)
        finally:
            # Always finalize logs/close fds, even on crash
            _ignore_sigint()
            shutdown(state, logger, reason="user_exit")

    finally:
        _ignore_sigint()
        mcp.stop_all()

def run_one_shot(
    task: str,
    model_name: str = None,
    trust_level: int = DEFAULT_TRUST_TIER,
    config_path: str = None
) -> bool:
    """
    Run OGnee with a single task.
    """
    # Detect model if not specified
    if not model_name:
        model_name = get_current_model()

    # Load MCP
    mcp = MCPManager(config_path)
    if not mcp.load_config():
        print_error("Failed to load MCP config")
        return False

    if mcp.start_all() == 0:
        print_warning("No MCP servers started. Continuing with local tools only.")

    success = False
    hub = get_hub()
    try:
        # Startup sequence
        state, logger, system_prompt = startup(
            model_name=model_name,
            trust_level=trust_level,
            quiet=True
        )
        # Line editing for approval prompts; no worker thread in one-shot mode.
        hub.start(history_path=INPUT_HISTORY_FILE, threaded=False)

        try:
            # Run task
            success = asyncio.run(one_shot(task, mcp, state, logger, system_prompt))
        finally:
            # Always finalize logs/close fds, even on crash
            shutdown(state, logger, reason="task_complete" if success else "task_failed")

        return success

    finally:
        hub.stop()
        mcp.stop_all()
