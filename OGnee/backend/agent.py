#!/usr/bin/env python3
"""
agent.py - Main agent loop

The core. Ties together MCP servers, tool execution, context tracking,
gates, logging, and the LLM. This is where OGneeys come to life.
"""

import asyncio
import json
import os
import re
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
)
from .context import (
    Session,
    SessionState,
    queue_user_message,
    drain_queued_messages,
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
from .gates import get_tier_name, check_and_enforce_ssh
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
from .session_log import SessionLogger, log_tool_to_global
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


def build_native_tools(mcp: MCPManager) -> list:
    """OpenAI ``tools`` schemas for MCP tools plus built-in local tools."""
    schemas = mcp.get_tools_schema()
    for local_tool in get_local_tools():
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

def visual_width(s: str) -> int:
    return max(0, wcswidth(s))

# width tracking using wcwidth() per character.
def pad_to_width(s: str, width: int) -> str:
    cumulative = 0
    out_chars = []
    for ch in s:
        ch_width = wcwidth.wcwidth(ch)
        if ch_width is None:
            ch_width = 0
        if cumulative + ch_width > width:
            break
        cumulative += ch_width
        out_chars.append(ch)
    result = "".join(out_chars)
    # Pad with spaces if shorter than width
    padding = width - cumulative
    if padding > 0:
        result += " " * padding
    return result

def row(content: str) -> str:
    return f"│ {pad_to_width(content, INNER_WIDTH - 2)} │"

# =============================================================================
# MID-TURN CONTROL
# =============================================================================
#
# The old `_check_stdin_for_pause` poller (select()+sys.stdin.readline()
# racing the terminal line discipline and readline) is gone. Mid-turn
# interruption is handled by KeyboardInterrupt; pause/plan state changes
# happen between turns through the slash commands and the single
# SessionState transition point (see context.SessionState).

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

# freezing the event loop. Made async and offloaded HTTP to thread pool.
async def chat(messages: list, tools: list = None) -> dict:
    """Send chat request to LM Studio (async — does not block the event loop).

    When ``tools`` is provided (native tool-calling profile), the schemas are
    sent via the OpenAI ``tools`` parameter with ``tool_choice="auto"``.
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
    response = await loop.run_in_executor(
        None,
        lambda: _llm_session.post(
            LM_STUDIO_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
    )
    response.raise_for_status()
    return response.json()

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

async def agent_loop(
    task: str,
    mcp: MCPManager,
    state: Session,
    logger: SessionLogger,
    system_prompt: str,
    messages: list = None
) -> bool:
    """
    Main agent loop with MCP tool execution.
    
    Returns True if task completed successfully, False otherwise.
    """
    
    # (removed — task header printed by interactive_loop only when needed)
    
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
        # Append new task
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
            response = await chat(messages)
            
            # Extract text and reasoning
            text = extract_response_text(response)
            reasoning = extract_reasoning_content(response)
            
            # Show reasoning if present
            if reasoning:
                print_dim(f"  💭 Thinking: {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}")
            
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
                if not _looks_like_tool_json(text):
                    print_ognee_response(text)
                    logger.log_assistant_message(text, usage.get("total_tokens", 0))
            else:
                print_dim("  (no text response)")
            
            return True
            
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
    native_tools = build_native_tools(mcp) if native_mode else None
    loop_detector = LoopDetector()
    loop_notice_sent = False  # deliver the "you're stuck" notice once

    if native_mode:
        print_dim(f"  🔧 Tool profile: native ({len(native_tools)} tools via API)")
    else:
        print_dim(f"  🔧 Tool profile: {profile} (text-parsed)")
    
    for iteration in range(MAX_ITERATIONS):
        # Never run while paused (the interactive loop enforces this too).
        if state.paused:
            print_system("⏸ Paused after turn. Use /resume to continue.")
            return True
        
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
        
        print_dim(f"\n--- Turn {iteration + 1}/{MAX_ITERATIONS} ---")
        
        try:
            # Call LLM — with native tool schemas when the model profile
            # supports them, otherwise rely on the profiled text format.
            print_ognee_thinking()
            response = await chat(messages_with_state, tools=native_tools)

            # Extract response text and reasoning
            text = extract_response_text(response)
            reasoning = extract_reasoning_content(response)

            # Show reasoning if present
            if reasoning:
                print_dim(f"  💭 Thinking: {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}")

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
                        read_results = await execute_tool_calls(
                            mcp_clients=mcp.servers,
                            tool_calls=read_calls,
                            state=state,
                            iteration=iteration
                        )

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
                    tool_results = await execute_tool_calls(
                        mcp_clients=mcp.servers,
                        tool_calls=tool_calls,
                        state=state,
                        iteration=iteration
                    )
                    
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
            
            # Check for pause after tool execution
            if state.paused:
                print_system("\n⏸ Paused after turn. Use /resume to continue.")
                return True
            
            if text:
                if not _looks_like_tool_json(text):
                    print_ognee_response(text)
                    logger.log_assistant_message(text, usage.get("total_tokens", 0))
            else:
                print_dim("  (no text response)")
            
            # Check for pause again
            if state.paused:
                print_system("\n⏸ Paused. Waiting for input...")
                return True
            
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


def _prompt_for_mode(state: Session) -> str:
    """REPL prompt string reflecting the current SessionState."""
    if state.mode is SessionState.PAUSED:
        return f"{COLORS.WARNING}⏸ {COLORS.RESET}"
    if state.mode is SessionState.PLAN_REVIEW:
        return f"{COLORS.SYSTEM}📋 {COLORS.RESET}"
    return f"{COLORS.COMMAND}› {COLORS.RESET}"


async def interactive_loop(
    mcp: MCPManager,
    state: Session,
    logger: SessionLogger,
    system_prompt: str
):
    """
    Interactive REPL for agent tasks.

    State machine (see context.SessionState): RUNNING sends input to the
    agent loop; PAUSED accepts only slash commands and queues other input;
    PLAN_REVIEW sends input to the agent loop with write tools queued.
    Slash commands NEVER fall through to the LLM as tasks.
    """

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

    # Get model info
    model_name = state.model_name
    tier_display = get_tier_name(state)
    
    tool_count = f"{len(mcp.all_tools)} MCP + {len(get_local_tools())} local"
    
    print(f"""
╭{"─" * (INNER_WIDTH)}╮
{row("OGnee Interactive Mode")}
{row("")}
{row(f"Model: {model_name}")}
{row(f"Trust: {tier_display}")}
{row(f"Tools: {tool_count}")}
{row("")}
{row("Commands: /help /pause /continue /tools-off /status /quit")}
╰{"─" * (INNER_WIDTH)}╯
""")
    
    while True:
        try:
            # Show status line periodically
            if state.turn_count > 0 and state.turn_count % 5 == 0:
                print_status_line(state.to_status())

            # Get input — prompt reflects the session state. read_input uses
            # prompt_toolkit (with a readline-safe fallback) so ANSI colors in
            # the prompt don't corrupt line wrapping.
            user_input = read_input(_prompt_for_mode(state)).strip()
            print()  # move off the input line before printing output

            if not user_input:
                continue

            # ---- Slash commands: handled here, NEVER sent to the LLM ----
            if is_command(user_input):
                result = execute_command(user_input, state)

                if result.result == CommandResult.QUIT:
                    break

                data = result.data or {}
                if "inject_user_message" in data:
                    # /continue, /recap: inject a message as the next task
                    user_input = data["inject_user_message"]
                elif "execute_plan" in data:
                    plan = data["execute_plan"]
                    results = await execute_plan_queue(plan, mcp, state, logger)
                    # Optional LLM narration: feed a compact result summary
                    # back so the model can tell the user what happened.
                    if results:
                        summary_lines = [
                            f"{'✓' if r.success else '✗'} "
                            f"{tc['server']}/{tc['tool']}: "
                            f"{(r.output or r.error or '')[:200]}"
                            for tc, r in zip(data["execute_plan"], results)
                        ]
                        user_input = (
                            "[PLAN EXECUTED] The queued plan was executed. Results:\n"
                            + "\n".join(summary_lines)
                            + "\nPlease summarize the outcome for the user."
                        )
                    else:
                        continue
                elif "ssh_command" in data:
                    # /ssh goes through the same gate/approval path as other
                    # shell tools — no bypass.
                    from .ssh import run_ssh_command
                    ssh_data = data["ssh_command"]
                    allowed, _ = check_and_enforce_ssh(
                        ssh_data["host"], ssh_data["command"], state
                    )
                    if not allowed:
                        continue
                    ssh_result = run_ssh_command(ssh_data["host"], ssh_data["command"])
                    if ssh_result.success:
                        print(ssh_result.stdout)
                    else:
                        print_error(ssh_result.stderr)
                    continue
                elif data.get("resume"):
                    # /resume: deliver messages queued while paused — these
                    # become the next task. No queued messages => no task.
                    queued = drain_queued_messages(state)
                    if not queued:
                        continue
                    for msg in queued:
                        print_system(f"  📤 Delivering queued message: {msg[:80]}")
                    user_input = "\n".join(queued)
                else:
                    continue
            else:
                # ---- Regular input ----
                if state.mode is SessionState.PAUSED:
                    depth = queue_user_message(state, user_input)
                    print_system(
                        f"  📥 Paused — message queued ({depth} pending). "
                        f"It will be delivered on /resume."
                    )
                    continue

            # Safety: never enter the agent loop while paused.
            if state.mode is SessionState.PAUSED:
                continue

            # Run the agent loop on the (possibly injected/queued) task
            await agent_loop(
                task=user_input,
                mcp=mcp,
                state=state,
                logger=logger,
                system_prompt=system_prompt,
                messages=messages
            )

        except (KeyboardInterrupt, EOFError):
            print("\n")
            break
        except Exception as e:
            print_error(f"Error: {e}")
            import traceback
            traceback.print_exc()

    return

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
            # Run interactive loop
            asyncio.run(interactive_loop(mcp, state, logger, system_prompt))
        finally:
            # Always finalize logs/close fds, even on crash
            shutdown(state, logger, reason="user_exit")

    finally:
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
    
    try:
        # Startup sequence
        state, logger, system_prompt = startup(
            model_name=model_name,
            trust_level=trust_level,
            quiet=True
        )

        try:
            # Run task
            success = asyncio.run(one_shot(task, mcp, state, logger, system_prompt))
        finally:
            # Always finalize logs/close fds, even on crash
            shutdown(state, logger, reason="task_complete" if success else "task_failed")

        return success

    finally:
        mcp.stop_all()
