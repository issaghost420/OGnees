#!/usr/bin/env python3
"""
tools.py - Tool dispatch, failure tracking, backoff logic

Executes MCP tool calls, tracks failures, backs off when tools keep failing.
No more infinite loops when something's broken.
"""

import asyncio
import json
import os
import time
import re
import requests
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import TOOL_TIMEOUT, TOOL_FAILURE_BACKOFF, RM_PATTERNS, get_tool_timeout
from .context import Session, SessionState, transition_state, record_tool_call, get_tool_failure_count
from .gates import gate_tool_call, get_tier_name
from .output_buffer import get_output_buffer, MAX_TOOL_RESULT_CHARS
from .display import (
    print_tool_call,
    print_tool_result,
    print_tool_error,
    print_tool_timeout,
    print_tool_denied,
    print_warning,
    print_error,
    print_system,
    print_dim,
    COLORS,
)

# =============================================================================
# BUILT-IN LOCAL TOOLS (no MCP server needed)
# =============================================================================

# local/speak POSTs assistant text to an external TTS endpoint. It is
# OPT-IN: disabled by default, enable with /speak-on or OGNEE_SPEAK=1.
# The endpoint is configurable via OGNEE_SPEAK_URL. Calls are gated as
# network requests (see gates.gate_tool_call).
CHATTERBOX_URL = os.environ.get(
    "OGNEE_SPEAK_URL", "https://dojos.duckdns.org/chatterbox/api/tts"
)
SPEAK_ENABLED = os.environ.get("OGNEE_SPEAK", "").lower() in ("1", "true", "yes", "on")

# FIX: Pre-compile regex patterns at module level to avoid recompiling on every
# clean_for_speech() call.  Was compiling 11 patterns per invocation.
_SPEECH_CLEAN_PATTERNS = [
    (re.compile(r'\*\*([^*]+)\*\*'), r'\1'),            # **bold**
    (re.compile(r'\*([^*]+)\*'), r'\1'),                  # *italic*
    (re.compile(r'`([^`]+)`'), r'\1'),                    # `code`
    (re.compile(r'```[\s\S]*?```'), ''),                  # code blocks
    (re.compile(r'^#{1,6}\s+', re.MULTILINE), ''),        # headers
    (re.compile(r'^\s*[-*]\s+', re.MULTILINE), ''),      # bullets
    (re.compile(r'^\s*\d+\.\s+', re.MULTILINE), ''),     # numbered lists
    (re.compile(r'\[([^\]]+)\]\([^)]+\)'), r'\1'),      # links
    (re.compile(r'\|[^\n]+\|'), ''),                     # table rows
    (re.compile(r'[-=]{3,}'), ''),                        # horizontal rules
    (re.compile(r'\n{3,}'), '\n\n'),                      # excess newlines
]


def clean_for_speech(text: str) -> str:
    """Strip markdown and technical formatting for TTS."""
    for pattern, repl in _SPEECH_CLEAN_PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


# FIX: Use a module-level requests.Session() for connection pooling in TTS calls.
# Was creating a brand-new TCP connection on every call_speak() invocation.
_TTS_SESSION = requests.Session()


async def call_speak(text: str, clean: bool = True) -> dict:
    """Send text to Chatterbox TTS."""
    global SPEAK_ENABLED
    
    if not SPEAK_ENABLED:
        return {"result": "[SPEAK DISABLED] Use /speak-on to enable"}
    
    if clean:
        text = clean_for_speech(text)
    
    if not text:
        return {"result": "[SKIPPED] No text to speak after cleaning"}
    
    try:
        # FIX: Replaced bare requests.post with session-pooled request.
        response = _TTS_SESSION.post(
            CHATTERBOX_URL,
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return {"result": f"[SPOKEN] {len(text)} chars sent to Chatterbox"}
        else:
            return {"error": f"Chatterbox returned {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": f"Chatterbox not running at {CHATTERBOX_URL}"}
    except Exception as e:
        return {"error": f"Speak failed: {str(e)}"}

def toggle_speak(enabled: bool):
    """Toggle speak functionality."""
    global SPEAK_ENABLED
    SPEAK_ENABLED = enabled

def is_speak_enabled() -> bool:
    return SPEAK_ENABLED

# Built-in tools that don't need MCP servers
LOCAL_TOOLS = {
    "local/speak": {
        "name": "speak",
        "full_name": "local/speak",
        "description": "Convert text to speech using local Chatterbox TTS. Use for verbal summaries, announcements, or responding vocally. Keep text conversational - avoid markdown, tables, code blocks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak aloud. Keep it conversational and TTS-friendly."
                }
            },
            "required": ["text"]
        },
        "handler": call_speak
    },
    "local/read_file": {
        "name": "read_file",
        "full_name": "local/read_file",
        "description": (
            "Read a specific range of lines from a text file. Use this for reading "
            "the middle of large files where MCP read_text_file only reads from the start. "
            "Parameters: path (required) — absolute file path; "
            "offset_line (optional, default 1) — first line to read (1-based); "
            "limit (optional, default 50) — max lines to return. "
            "Returns the requested lines with a header showing the range."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "offset_line": {"type": "integer", "description": "First line to read (1-based)", "default": 1},
                "limit": {"type": "integer", "description": "Max lines to read", "default": 50}
            },
            "required": ["path"]
        },
        "handler": None  # wired in call_local_tool
    },
    "local/read_output": {
        "name": "read_output",
        "full_name": "local/read_output",
        "description": (
            "Read a chunk of a previously stored large tool output by reference. "
            "When a tool returns more than ~20000 characters, the full result is stashed "
            "and you receive a preview with a reference handle (e.g. out_0001). "
            "Use this tool to paginate through the rest. Parameters: ref (string, required) — "
            "the reference handle from the preview; offset (integer, optional, default 0) — "
            "character offset to start reading from; max_chars (integer, optional, default 4000) — "
            "maximum characters in this chunk. Returns the next chunk and tells you if more follows."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Reference handle from the preview, e.g. 'out_0001'"
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start reading from",
                    "default": 0
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters in this chunk",
                    "default": 4000
                }
            },
            "required": ["ref"]
        },
        "handler": None  # wired below in call_local_tool
    }
}

def get_local_tools() -> list:
    """Get built-in local tools."""
    return list(LOCAL_TOOLS.values())

async def call_local_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a local built-in tool."""
    
    # Handle read_file — read specific line range from a file
    if tool_name == "local/read_file":
        path = arguments.get("path", "")
        offset_line = max(1, arguments.get("offset_line", 1))
        limit = max(1, arguments.get("limit", 50))
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            total = len(all_lines)
            start_idx = offset_line - 1
            end_idx = min(start_idx + limit, total)
            selected = all_lines[start_idx:end_idx]
            header = f"[Lines {offset_line}-{end_idx} of {total}] {path}\n"
            return {"result": header + "".join(selected)}
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}
        except PermissionError:
            return {"error": f"Permission denied: {path}"}
        except Exception as e:
            return {"error": f"Error reading {path}: {str(e)}"}

    # Handle read_output pagination tool (special: no handler in LOCAL_TOOLS)
    if tool_name == "local/read_output":
        buf = get_output_buffer()
        ref = arguments.get("ref", "")
        offset = arguments.get("offset", 0)
        max_chars = arguments.get("max_chars", 4000)
        chunk, has_more, next_offset = buf.read(ref, offset, max_chars)
        meta = buf.metadata(ref)
        total_chars = meta["total_chars"] if meta else "?"
        footer = f"\n\n[chunk {offset}-{offset+len(chunk)}/{total_chars} chars]"
        if has_more:
            footer += f" Use read_output(ref='{ref}', offset={next_offset}) for next chunk"
        return {"result": chunk + footer}
    
    tool = LOCAL_TOOLS.get(tool_name)
    if not tool:
        return {"error": f"Unknown local tool: {tool_name}"}
    
    handler = tool.get("handler")
    if not handler:
        return {"error": f"No handler for tool: {tool_name}"}
    
    # Extract just the text arg for speak
    if tool_name == "local/speak":
        text = arguments.get("text", "")
        return await handler(text)
    
    return await handler(**arguments)


# =============================================================================
# TOOL RESULT TYPES
# =============================================================================

@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    output: str
    error: Optional[str] = None
    duration_ms: int = 0
    was_denied: bool = False
    was_redirected: bool = False
    redirect_path: Optional[str] = None

# =============================================================================
# FAILURE TRACKING
# =============================================================================

@dataclass
class ToolFailureTracker:
    """Tracks tool failures for backoff logic.
    
    Key improvements over the naive per-tool tracking:
    - Path-aware: file tool failures on /mnt/titan don't lock /home/tower
    - Time decay: backoff auto-resets after BACKOFF_DECAY_SECONDS
    - Successful calls immediately reset the counter for that key
    """
    failures: Dict[str, List[datetime]] = field(default_factory=dict)
    consecutive: Dict[str, int] = field(default_factory=dict)
    last_failure_time: Dict[str, datetime] = field(default_factory=dict)
    
    # After this many seconds of no failures, auto-reset backoff
    BACKOFF_DECAY_SECONDS = 30
    
    # Tools where the first argument should be part of the backoff key
    # (so failures on one path don't lock the tool for other paths)
    PATH_AWARE_TOOLS = frozenset({
        "list_directory", "read_file", "read_text_file", "write_file",
        "get_file_info", "search_files", "list_directory_with_sizes",
        "create_directory", "execute_command",
    })
    
    @staticmethod
    def make_key(server: str, tool: str, args: dict = None) -> str:
        """Build a backoff key, optionally path-aware.
        
        For file tools, includes the path ROOT (first 2 components) so that
        failures on /mnt/titan/... don't lock /home/tower/..., but failures
        on /mnt/titan/A and /mnt/titan/B DO share a counter.
        """
        base_key = f"{server}/{tool}"
        if args and tool in ToolFailureTracker.PATH_AWARE_TOOLS:
            # Extract path from common arg names
            path = args.get("path", args.get("command", ""))
            if isinstance(path, str) and path.startswith("/"):
                # Use the first 2 path components as the scope
                parts = path.strip("/").split("/")[:2]
                scope = "/" + "/".join(parts) if parts else ""
                return f"{base_key}:{scope}"
        return base_key
    
    def record_failure(self, tool_key: str):
        now = datetime.now()
        if tool_key not in self.failures:
            self.failures[tool_key] = []
        self.failures[tool_key].append(now)
        self.consecutive[tool_key] = self.consecutive.get(tool_key, 0) + 1
        self.last_failure_time[tool_key] = now
    
    def record_success(self, tool_key: str):
        self.consecutive[tool_key] = 0
    
    def get_consecutive_failures(self, tool_key: str) -> int:
        return self.consecutive.get(tool_key, 0)
    
    def should_backoff(self, tool_key: str, threshold: int = TOOL_FAILURE_BACKOFF) -> bool:
        consecutive = self.get_consecutive_failures(tool_key)
        if consecutive < threshold:
            return False
        
        # Check time decay: if last failure was long enough ago, auto-reset
        last_fail = self.last_failure_time.get(tool_key)
        if last_fail:
            elapsed = (datetime.now() - last_fail).total_seconds()
            if elapsed > self.BACKOFF_DECAY_SECONDS:
                # Backoff expired — give the tool another chance
                self.consecutive[tool_key] = 0
                return False
        
        return True
    
    def get_recent_failures(self, tool_key: str, window_minutes: int = 5) -> int:
        if tool_key not in self.failures:
            return 0
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        return sum(1 for t in self.failures[tool_key] if t > cutoff)
    
    def clear(self, tool_key: str = None):
        if tool_key:
            self.failures.pop(tool_key, None)
            self.consecutive.pop(tool_key, None)
            self.last_failure_time.pop(tool_key, None)
        else:
            self.failures.clear()
            self.consecutive.clear()
            self.last_failure_time.clear()

# Global tracker instance
_failure_tracker = ToolFailureTracker()

def get_failure_tracker() -> ToolFailureTracker:
    return _failure_tracker

# =============================================================================
# TOOL EXECUTION
# =============================================================================

async def execute_tool_call(
    mcp_clients: Dict[str, Any],
    server: str,
    tool: str,
    args: Dict[str, Any],
    state: Session,
    iteration: int = 0
) -> ToolResult:
    """Execute a single tool call with all safety checks."""
    tool_key = ToolFailureTracker.make_key(server, tool, args)
    display_key = f"{server}/{tool}"  # For user-facing messages
    tracker = get_failure_tracker()
    start_time = time.time()
    
    # =========================================================================
    # LOCAL TOOLS - same gate/ACL checks as MCP tools, no MCP transport needed
    # =========================================================================
    if server == "local":
        print_tool_call(server, tool, args, iteration)

        # Local tools hit the filesystem and the network just like MCP
        # tools do, so they must pass the same gate layer (EXCLUDED_PATHS,
        # read-only tiers, network approval). Previously this branch
        # returned early and local/read_file could read ~/.ssh et al.
        allowed, modified_args = gate_tool_call(server, tool, args, state)
        if not allowed:
            print_tool_denied(display_key, "Denied by gate")
            return ToolResult(
                success=False,
                output="",
                error="Denied by permission gate",
                was_denied=True
            )
        if modified_args:
            args = modified_args

        result = await call_local_tool(display_key, args)  # local tools match on "local/tool_name"
        duration_ms = int((time.time() - start_time) * 1000)
        
        if "error" in result:
            print_tool_error(display_key, result["error"])
            return ToolResult(success=False, output="", error=result["error"], duration_ms=duration_ms)
        
        print_tool_result(result.get("result", ""), success=True)
        return ToolResult(success=True, output=result.get("result", ""), duration_ms=duration_ms)
    
    # =========================================================================
    # MCP TOOLS - gate checks, backoff, execution
    # =========================================================================
    
    # Print the tool call BEFORE gating so user always sees what was attempted.
    # The gate may block or prompt, but transparency comes first.
    print_tool_call(server, tool, args, iteration)
    
    # Check backoff first
    if tracker.should_backoff(tool_key):
        consecutive = tracker.get_consecutive_failures(tool_key)
        print_warning(f"  ⏸ Backing off {display_key} ({consecutive} consecutive failures)")
        print_system(f"    Use /enable {display_key} to reset and retry")
        return ToolResult(
            success=False,
            output="",
            error=f"Tool backed off after {consecutive} consecutive failures. Will auto-reset in {ToolFailureTracker.BACKOFF_DECAY_SECONDS}s.",
            was_denied=True
        )
    
    # Gate check
    allowed, modified_args = gate_tool_call(server, tool, args, state)
    
    if not allowed:
        print_tool_denied(display_key, "Denied by gate")
        return ToolResult(
            success=False,
            output="",
            error="Denied by permission gate",
            was_denied=True
        )
    
    # Use modified args if gate changed them (e.g., path redirect)
    if modified_args:
        args = modified_args
    
    # Get the MCP client
    client = mcp_clients.get(server)
    if not client:
        # Routing/validation error, not a tool failure — do NOT feed the
        # backoff tracker (hallucinated names must not burn real budgets).
        error = f"Unknown MCP server: {server}"
        print_tool_error(display_key, error)
        return ToolResult(success=False, output="", error=error)
    
    # Execute with timeout (per-server/per-tool configurable)
    tool_timeout = get_tool_timeout(server, tool)
    try:
        # call_tool is synchronous, run in executor for timeout support
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: client.call_tool(tool, args)),
            timeout=tool_timeout
        )
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Extract output text and the MCP isError flag
        output = ""
        is_error = False
        if hasattr(result, 'content'):
            for content in result.content:
                if hasattr(content, 'text'):
                    output += content.text
            if getattr(result, 'isError', False):
                is_error = True
        elif isinstance(result, str):
            output = result
        elif isinstance(result, dict):
            output = result.get("result", str(result))
            # The hand-rolled MCP client propagates the JSON-RPC isError flag
            # in the returned dict — trust it over keyword scanning.
            if result.get("isError"):
                is_error = True
        else:
            output = str(result)

        # Keyword scanning is a last resort: legitimate file content can
        # contain words like "error", "denied", "permission" (e.g., state.md
        # talking about permission gates). Only scan very short outputs that
        # look like error messages, not file content.
        if not is_error and len(output) < 500:
            error_indicators = ["ENOENT", "EACCES", "permission denied",
                              "No such file", "not found", "Unknown MCP"]
            output_lower = output.lower()
            is_error = any(ind.lower() in output_lower for ind in error_indicators)

        # ------------------------------------------------------------------
        # SMART OUTPUT: stash large outputs, give LLM a preview + reference
        # ------------------------------------------------------------------
        if len(output) > MAX_TOOL_RESULT_CHARS:
            buf = get_output_buffer()
            stashed_ref, preview = buf.store(server, tool, iteration, output)
            # The reference key must be inline in the returned text or the
            # model has no way to paginate the stashed output.
            output = f"{preview}\n\n[OUTPUT STORED — Reference: {stashed_ref}. Use local/read_output with ref=\"{stashed_ref}\" to read more.]"

        # Record in state
        record_tool_call(
            state=state,
            server=server,
            tool=tool,
            args=args,
            result=output,
            success=not is_error,
            duration_ms=duration_ms
        )
        
        # Update tracker
        if is_error:
            tracker.record_failure(tool_key)
        else:
            tracker.record_success(tool_key)
        
        # Print result (truncated)
        print_tool_result(output[:200], success=not is_error)
        
        return ToolResult(
            success=not is_error,
            output=output,
            error=output if is_error else None,
            duration_ms=duration_ms
        )
        
    except asyncio.TimeoutError:
        duration_ms = int((time.time() - start_time) * 1000)
        print_tool_timeout(tool_key, tool_timeout)
        tracker.record_failure(tool_key)
        
        record_tool_call(
            state=state,
            server=server,
            tool=tool,
            args=args,
            result=f"Timeout after {tool_timeout}s",
            success=False,
            duration_ms=duration_ms
        )
        
        return ToolResult(
            success=False,
            output="",
            error=f"Timeout after {tool_timeout}s",
            duration_ms=duration_ms
        )
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error = str(e)
        print_tool_error(tool_key, error)
        tracker.record_failure(tool_key)
        
        record_tool_call(
            state=state,
            server=server,
            tool=tool,
            args=args,
            result=error,
            success=False,
            duration_ms=duration_ms
        )
        
        return ToolResult(
            success=False,
            output="",
            error=error,
            duration_ms=duration_ms
        )

# =============================================================================
# BATCH TOOL EXECUTION
# =============================================================================

# Read-only tools that are safe to execute in parallel (no side effects)
READ_ONLY_TOOLS = frozenset({
    "read_file", "list_directory", "search_files", "get_file_info",
    "read_multiple_files", "list_allowed_directories", "directory_tree",
})

async def execute_tool_calls(
    mcp_clients: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    state: Session,
    iteration: int = 0
) -> List[ToolResult]:
    """Execute tool calls — read-only tools run in parallel, writes run sequentially."""
    if not tool_calls:
        return []
    
    # Separate into parallel-safe reads and sequential writes
    read_indices = []
    write_indices = []
    for i, call in enumerate(tool_calls):
        if call.get("tool", "") in READ_ONLY_TOOLS:
            read_indices.append(i)
        else:
            write_indices.append(i)
    
    results: List[Optional[ToolResult]] = [None] * len(tool_calls)
    
    # Execute read-only tools in parallel
    if read_indices:
        async def _exec_read(idx: int):
            call = tool_calls[idx]
            return idx, await execute_tool_call(
                mcp_clients=mcp_clients,
                server=call.get("server", ""),
                tool=call.get("tool", ""),
                args=call.get("args", {}),
                state=state,
                iteration=iteration + idx
            )
        
        read_results = await asyncio.gather(
            *[_exec_read(i) for i in read_indices],
            return_exceptions=True
        )
        for item in read_results:
            if isinstance(item, Exception):
                # Shouldn't happen — execute_tool_call has its own try/except
                continue
            idx, result = item
            results[idx] = result
            if result.was_denied:
                # Log it but don't bail — let the other read tools finish.
                # The denied tool's error message is already in results[idx].
                tool_key = f"{tool_calls[idx].get('server','')}/{tool_calls[idx].get('tool','')}"
                print_warning(f"  ⏸ {tool_key} denied/backed-off — continuing with remaining tools")
    
    # Execute write tools sequentially (order and side effects matter)
    for i in write_indices:
        call = tool_calls[i]
        server = call.get("server", "")
        tool = call.get("tool", "")
        args = call.get("args", {})
        
        result = await execute_tool_call(
            mcp_clients=mcp_clients,
            server=server,
            tool=tool,
            args=args,
            state=state,
            iteration=iteration + i
        )
        results[i] = result
        
        if result.was_denied:
            print_warning(f"  🛑 Stopping batch: {server}/{tool} was denied")
            break
        
        if state.step_mode:
            state.step_mode = False
            # Route through the single transition point so `mode` and the
            # `paused` flag never disagree.
            transition_state(state, SessionState.PAUSED)
            print_system("  👟 Step complete. Paused.")
            break
        
        tracker = get_failure_tracker()
        tool_key = ToolFailureTracker.make_key(server, tool, args)
        if tracker.should_backoff(tool_key):
            print_warning(f"  ⏸ Skipping {server}/{tool}: backed off after failures")
            # Add a fake result so the model knows this tool was skipped
            results[i] = ToolResult(
                success=False,
                output="",
                error=f"Tool {server}/{tool} is temporarily disabled due to repeated failures. Use /enable {server}/{tool} to reset, or try a different approach.",
                duration_ms=0,
                was_denied=True
            )
            continue  # Skip to next tool, don't break the batch
    
    # Every tool_call MUST get a result: native-mode backends reject
    # assistant messages whose tool_calls lack matching role:"tool"
    # replies (HTTP 400). If the batch was halted early (denial, step
    # mode), synthesize rejection results for the unexecuted calls so
    # results stay 1:1 with calls — no dangling tool_call_ids.
    for i, r in enumerate(results):
        if r is None:
            call = tool_calls[i]
            results[i] = ToolResult(
                success=False,
                output="",
                error=(
                    f"skipped: batch halted before "
                    f"{call.get('server', '')}/{call.get('tool', '')} executed "
                    f"(earlier call was denied or step mode paused execution)"
                ),
                duration_ms=0,
                was_denied=True
            )
    return results

# =============================================================================
# TOOL AVAILABILITY
# =============================================================================

def _get_tool_name(t) -> str:
    """Helper to safely extract a tool name whether t is a dict or an object."""
    if isinstance(t, dict):
        return t.get("name", "")
    return getattr(t, "name", "")


def get_available_tools(mcp_clients: Dict[str, Any]) -> Dict[str, List[str]]:
    """Get dict of server -> tool names, including local tools."""
    # FIX: Was crashing with AttributeError when client.tools contained dicts
    # instead of objects (t.name failed). Now safely handles both types.
    available = {"local": [t["name"] for t in LOCAL_TOOLS.values()]}
    
    for server_name, client in mcp_clients.items():
        if hasattr(client, 'tools'):
            available[server_name] = [_get_tool_name(t) for t in client.tools]
        elif hasattr(client, 'list_tools'):
            available[server_name] = []
    
    return available

def format_tools_for_prompt(mcp_clients: Dict[str, Any], state: Session) -> str:
    """Format available tools for inclusion in system prompt."""
    lines = ["Available tools:"]
    
    disabled = getattr(state, 'disabled_tools', set())
    
    # Add local tools first
    if "local" not in disabled:
        lines.append("\nlocal/ (built-in):")
        for tool in LOCAL_TOOLS.values():
            tool_key = tool["full_name"]
            if tool_key not in disabled:
                lines.append(f"  - {tool['name']}: {tool['description'][:60]}...")
    
    # Add MCP tools
    for server_name, client in mcp_clients.items():
        if server_name in disabled:
            continue
            
        if hasattr(client, 'tools'):
            tools = client.tools
            tool_names = []
            
            # FIX: Also use _get_tool_name here for consistency (same crash path
            # as get_available_tools if tools are dicts instead of objects).
            for t in tools:
                tname = _get_tool_name(t)
                tool_key = f"{server_name}/{tname}"
                if tool_key not in disabled:
                    tool_names.append(tname)
            
            if tool_names:
                lines.append(f"\n{server_name}/:")
                for name in tool_names:
                    lines.append(f"  - {name}")
    
    return "\n".join(lines)

# =============================================================================
# PLAN MODE SUPPORT
# =============================================================================

def queue_tool_call(state: Session, server: str, tool: str, args: Dict[str, Any]):
    """Queue a tool call for plan mode."""
    if not hasattr(state, 'pending_plan'):
        state.pending_plan = []
    
    state.pending_plan.append({
        "server": server,
        "tool": tool,
        "args": args,
        "queued_at": datetime.now().isoformat()
    })
    
    print_system(f"  📋 Queued: {server}/{tool}")

def get_pending_plan(state: Session) -> List[Dict]:
    return getattr(state, 'pending_plan', [])

def clear_pending_plan(state: Session):
    state.pending_plan = []

# =============================================================================
# TOOL RESULT FORMATTING
# =============================================================================

def format_tool_result_for_llm(
    result: ToolResult,
    tool_key: str,
    tool_call_id: str = "",
) -> Dict[str, Any]:
    """Format a tool result as an OpenAI ``role: "tool"`` message.

    ``tool_call_id`` is required by the OpenAI chat-completions API whenever
    the assistant turn carried native tool_calls; pass the id from the parsed
    call. ``tool_key`` is the ``server/tool`` display name.
    """
    message: Dict[str, Any] = {"role": "tool", "name": tool_key}
    if tool_call_id:
        message["tool_call_id"] = tool_call_id

    if result.was_denied:
        message["content"] = f"[DENIED] {result.error}"
    elif result.was_redirected:
        message["content"] = f"[REDIRECTED to shadow: {result.redirect_path}]\n{result.output}"
    elif not result.success:
        message["content"] = f"[ERROR] {result.error}"
    else:
        message["content"] = result.output
    return message

def format_tool_results_for_llm(
    results: List[ToolResult],
    tool_calls: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Format multiple tool results as ``role: "tool"`` messages."""
    formatted = []
    for result, call in zip(results, tool_calls):
        tool_key = f"{call['server']}/{call['tool']}"
        formatted.append(format_tool_result_for_llm(result, tool_key, call.get("id", "")))
    return formatted

# =============================================================================
# FAILURE SUMMARY
# =============================================================================

def get_failure_summary(state: Session) -> str:
    """Get a summary of tool failures for the model."""
    tracker = get_failure_tracker()
    
    if not tracker.consecutive:
        return ""
    
    lines = ["Recent tool issues:"]
    
    for tool_key, count in tracker.consecutive.items():
        if count > 0:
            if count >= TOOL_FAILURE_BACKOFF:
                lines.append(f"  ⛔ {tool_key}: BACKED OFF ({count} failures)")
            else:
                lines.append(f"  ⚠ {tool_key}: {count} consecutive failure(s)")
    
    return "\n".join(lines) if len(lines) > 1 else ""
