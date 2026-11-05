#!/usr/bin/env python3
"""
mcp.py - MCP server spawning, management, and tool-schema/prompt generation.

Handles Model Context Protocol servers over two transports:

- **stdio** (default): spawn a subprocess per server and speak JSON-RPC over
  stdin/stdout. Hardened with a dedicated stderr-drain thread, a single
  write lock for all stdin writes, EOF detection, and crash-loop-budgeted
  auto-restart.
- **proxy** (``MCP_MODE=proxy``): attach to an mcpo-style HTTP proxy for
  config entries that carry a ``url`` key instead of ``command``/``args``.

Security posture: an allowlist of spawnable commands, an environment
allowlist (no secrets leak into server subprocesses), and shell-metacharacter
rejection on arguments.
"""

import json
import subprocess
import os
import threading
import queue
import time
import select
from collections import deque
from typing import Optional, Dict, List, Any
from pathlib import Path

import requests

from .config import (
    MCP_CONFIG,
    MCP_CONFIG_FALLBACK,
    TOOL_TIMEOUT,
    get_tool_timeout,
)
from .display import (
    print_system,
    print_success,
    print_warning,
    print_error,
    print_dim,
    COLORS,
)

# =============================================================================
# SECURITY CONSTANTS
# =============================================================================

# Reject oversized JSON-RPC lines to bound memory growth from malicious or
# garbage output from a server subprocess.
MAX_LINE_LEN = 1024 * 1024  # 1MB

# Spawned commands not on this allowlist are rejected to prevent arbitrary
# command execution from mcp.json.
ALLOWED_MCP_COMMANDS = frozenset([
    "node",
    "python",
    "python3",
    "npx",
    "uvx",
    "deno",
    "bun",
    "docker",
    "deepresearch-mcp",
])

# Only these variables are inherited from the parent environment, preventing
# accidental exposure of secrets (API keys, tokens, passwords).
SAFE_ENV_VARS = frozenset([
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "PWD",
    "TMPDIR",
    "NODE_PATH",
    "PYTHONPATH",
    "UV_PYTHON",
    "DENO_DIR",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
])

# Shell metacharacters that could be used for command injection.
SHELL_METACHARACTERS = set(";|&$`<>*\n\r")

# NOTE: These are checked as defense-in-depth even though subprocess.Popen
# with a list (not shell=True) does not invoke a shell. The primary protection
# is the ALLOWED_MCP_COMMANDS allowlist above.

# =============================================================================
# TRANSPORT / RESTART TUNING
# =============================================================================

# MCP_MODE selects how `url`-style config entries are handled:
#   stdio (default) - only spawn command-based servers; url entries are skipped
#   proxy           - attach to url entries over HTTP (mcpo) instead of
#                     double-spawning servers the proxy already runs
MCP_MODE = os.environ.get("MCP_MODE", "stdio").strip().lower() or "stdio"

# Crash-loop budget: a server may be auto-restarted at most this many times
# within the window before being marked down for the rest of the session.
MAX_RESTARTS = 3
RESTART_WINDOW_SECONDS = 300

# How many lines of server stderr to keep for crash diagnostics.
STDERR_TAIL_LINES = 200

# OpenAI function-name grammar: ^[a-zA-Z0-9_-]{1,64}$. `server/tool` is not
# legal, so native tool schemas are exposed as `server__tool` and mapped back
# on dispatch.
NATIVE_NAME_SEP = "__"


def validate_mcp_command(command: str, args: List[str]) -> Optional[str]:
    """Validate a server command against the allowlist and its args for shell
    metacharacters that could enable command injection.

    Returns an error message string if invalid, None if valid.
    """
    # Get the base command name (strip path)
    base_cmd = os.path.basename(command)

    if base_cmd not in ALLOWED_MCP_COMMANDS:
        return (
            f"Command '{base_cmd}' is not in the allowlist. "
            f"Allowed commands: {', '.join(sorted(ALLOWED_MCP_COMMANDS))}"
        )

    # Check each argument for shell metacharacters
    for i, arg in enumerate(args):
        for ch in SHELL_METACHARACTERS:
            if ch in arg:
                return (
                    f"Argument {i} ('{arg[:50]}') contains shell "
                    f"metacharacter '{ch}' - possible injection attempt"
                )
        # Check for common dangerous patterns
        dangerous = ["$(", "${", "`", "||", "&&", ";", "|"]
        for pattern in dangerous:
            if pattern in arg:
                return (
                    f"Argument {i} ('{arg[:50]}') contains dangerous "
                    f"pattern '{pattern}'"
                )

    return None


def build_safe_env(user_env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Build a subprocess environment from allowlisted parent variables plus
    user-provided env. This prevents secrets (API keys, tokens, passwords)
    from leaking into the MCP server subprocess.
    """
    safe = {}
    for key, value in os.environ.items():
        if key in SAFE_ENV_VARS:
            safe[key] = value
    if user_env:
        safe.update(user_env)
    return safe


def make_native_name(server: str, tool: str) -> str:
    """Build an OpenAI-legal function name for a ``server/tool`` pair."""
    raw = f"{server}{NATIVE_NAME_SEP}{tool}"
    # Strip any characters outside the OpenAI function-name grammar.
    cleaned = "".join(c if (c.isalnum() or c in "_-") else "_" for c in raw)
    return cleaned[:64]


# =============================================================================
# MCP SERVER CLASS (stdio transport)
# =============================================================================

class MCPServer:
    """Manages a single MCP server subprocess with JSON-RPC communication.

    Robustness properties:
    - stderr is drained continuously by a daemon thread into a ring buffer
      (a full stderr pipe would otherwise deadlock node-based servers).
    - All stdin writes happen under a single write lock, so concurrent tool
      calls can never interleave JSON-RPC frames.
    - EOF on stdout breaks the reader loop and marks the server dead instead
      of spinning.
    """

    def __init__(self, name: str, command: str, args: list, env: dict = None,
                 cwd: str = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = build_safe_env(env)
        self.cwd = cwd  # optional working directory for the subprocess
        self.process: Optional[subprocess.Popen] = None
        self.tools: list = []
        self.request_id = 0
        self._read_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        # Recent stderr lines, kept for crash diagnostics.
        self.stderr_tail: deque = deque(maxlen=STDERR_TAIL_LINES)
        # Event-driven response delivery instead of CPU-wasting polling.
        self._pending_responses: Dict[int, Any] = {}
        self._pending_events: Dict[int, threading.Event] = {}
        self._lock = threading.Lock()        # guards pending maps + request id
        self._write_lock = threading.Lock()  # serializes ALL stdin writes
        self._running = False
        self.dead = False  # set by the reader thread when stdout hits EOF
        # Restart-race protection: each start()/stop() bumps the generation,
        # and reader/stderr threads capture (process, generation) at spawn.
        # A thread whose generation no longer matches ignores output and does
        # NOT touch self.dead — it belongs to a dead incarnation.
        self._generation = 0
        self._lifecycle_lock = threading.Lock()  # serializes start/stop

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the MCP server subprocess."""
        with self._lifecycle_lock:
            return self._start_locked()

    def _start_locked(self) -> bool:
        validation_error = validate_mcp_command(self.command, self.args)
        if validation_error:
            print_error(f"  ❌ {self.name}: {validation_error}")
            return False

        cwd = None
        if self.cwd:
            expanded = os.path.expanduser(self.cwd)
            if os.path.isdir(expanded):
                cwd = expanded
            else:
                print_warning(f"  ⚠️  {self.name}: cwd '{self.cwd}' does not exist, ignoring")

        try:
            full_command = [self.command] + self.args
            print_dim(f"  🔧 Starting {self.name}: {' '.join(full_command)}")

            self.process = subprocess.Popen(
                full_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                cwd=cwd,
                bufsize=0
            )

            # Start reader + stderr-drain threads, each bound to THIS
            # process/generation so a stale thread from a previous
            # incarnation can't steal frames or flip self.dead after a
            # restart.
            self._running = True
            self.dead = False
            self._generation += 1
            self._read_thread = threading.Thread(
                target=self._read_responses, daemon=True,
                args=(self.process, self._generation),
                name=f"mcp-reader-{self.name}",
            )
            self._read_thread.start()
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, daemon=True,
                args=(self.process, self._generation),
                name=f"mcp-stderr-{self.name}",
            )
            self._stderr_thread.start()

            # Give it a moment to start
            time.sleep(0.5)

            if self.process.poll() is not None:
                tail = "".join(self.stderr_tail)[-200:]
                print_error(f"  ❌ {self.name} failed to start: {tail}")
                # Reap the child and stop the reader/drain threads so a
                # failed start doesn't leak a zombie process or threads.
                self._stop_locked()
                return False

            return True

        except FileNotFoundError:
            print_error(f"  ❌ {self.name}: command not found: {self.command}")
            return False
        except Exception as e:
            print_error(f"  ❌ {self.name} error: {e}")
            return False

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _drain_stderr(self, proc=None, generation: int = -1):
        """Continuously drain the server's stderr into a ring buffer.

        Node MCP servers log to stderr; if the pipe is never read, the 64KB
        OS buffer fills and the server blocks on write, deadlocking every
        pending tool call. Lines are kept in ``stderr_tail`` for diagnostics.

        Bound to a specific (process, generation): after a restart this
        thread exits instead of draining the new process alongside its
        replacement thread.
        """
        if proc is None:
            proc = self.process
            generation = self._generation
        stream = proc.stderr if proc else None
        if stream is None:
            return
        while self._running and generation == self._generation:
            try:
                line = stream.readline()
                if not line:
                    break  # EOF — server exited
                self.stderr_tail.append(
                    line.decode("utf-8", errors="replace")
                )
            except (OSError, ValueError):
                break  # pipe closed during shutdown
            except Exception:
                break

    def _read_responses(self, proc=None, generation: int = -1):
        """Background thread to read JSON-RPC responses.

        Bound to the (process, generation) captured at thread start. If a
        restart swapped in a new process, this thread exits WITHOUT touching
        self.dead — marking the new incarnation dead was the phantom-dead
        bug — and never reads from the new process's stdout.
        """
        if proc is None:
            proc = self.process
            generation = self._generation
        stdout = proc.stdout if proc else None
        while (
            self._running
            and generation == self._generation
            and proc is not None
            and proc.poll() is None
        ):
            try:
                # select() with a timeout avoids blocking forever on readline()
                # if the server outputs garbage or hangs.
                readable, _, _ = select.select(
                    [stdout], [], [], 0.5
                )
                if not readable:
                    continue

                # Cap line length to bound memory growth from a malicious or
                # misbehaving server.
                raw = stdout.readline(MAX_LINE_LEN)
                if not raw:
                    # EOF: the server's stdout closed — it is gone. Break
                    # instead of spinning, and mark the server dead so the
                    # manager can restart it.
                    break

                # If the line hit the cap, drain the rest of it to stay in
                # sync with the stream — then skip it entirely. Parsing a
                # truncated line would only produce a guaranteed decode error.
                if len(raw) >= MAX_LINE_LEN and not raw.endswith(b"\n"):
                    while True:
                        chunk = stdout.read(1)
                        if not chunk or chunk == b"\n":
                            break
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                # Generation may have changed while we were blocked on the
                # read — drop output from a dead incarnation.
                if generation != self._generation:
                    break

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Invalid JSON, skip

                self._handle_message(msg)
            except Exception:
                break

        # Only the current generation's reader may mark the server dead;
        # a stale thread from before a restart must not clobber dead=False.
        if generation == self._generation and proc is self.process:
            self.dead = True

    def _handle_message(self, msg: dict):
        """Route one parsed JSON-RPC message from the server."""
        if "id" in msg and ("result" in msg or "error" in msg):
            # Response to one of our requests. Only store it if a waiter is
            # still registered — after a timeout the waiter is gone and a
            # late response would leak in _pending_responses forever.
            req_id = msg["id"]
            with self._lock:
                event = self._pending_events.pop(req_id, None)
                if event is not None:
                    self._pending_responses[req_id] = msg
            if event is not None:
                event.set()
        elif "id" in msg and "method" in msg:
            # Server-initiated request (sampling, roots, elicitation...).
            # We support none of them, but must reply or the server may hang
            # waiting for a response.
            self._send_raw({
                "jsonrpc": "2.0",
                "id": msg["id"],
                "error": {
                    "code": -32601,
                    "message": f"Method not supported by client: {msg.get('method')}",
                },
            })
        # else: notification (progress, tools/list_changed, ...) — no reply
        # required; nothing to do.

    # ------------------------------------------------------------------
    # Wire I/O
    # ------------------------------------------------------------------

    def _send_raw(self, payload: dict) -> Optional[Exception]:
        """Write one JSON-RPC frame under the write lock.

        The write lock is held across write()+flush() so concurrent callers
        can never interleave bytes from two frames.
        """
        try:
            data = (json.dumps(payload) + "\n").encode("utf-8")
            with self._write_lock:
                if not self.process or self.process.stdin is None:
                    return RuntimeError(f"Server {self.name} not running")
                self.process.stdin.write(data)
                self.process.stdin.flush()
            return None
        except Exception as e:
            return e

    def _send_request(self, method: str, params: dict = None, timeout: float = 30) -> dict:
        """Send JSON-RPC request and wait for response."""
        if not self.process or self.process.poll() is not None:
            return {"error": {"message": f"Server {self.name} not running"}}

        # Allocate the id and register the completion event under one lock so
        # the whole register-then-send sequence is race-free.
        req_id = None
        event = threading.Event()
        try:
            with self._lock:
                self.request_id += 1
                req_id = self.request_id
                request = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": method,
                }
                if params:
                    request["params"] = params

                # Register the event before sending to avoid the race where
                # the response arrives before we start waiting.
                self._pending_events[req_id] = event

            err = self._send_raw(request)
            if err is not None:
                with self._lock:
                    self._pending_events.pop(req_id, None)
                return {"error": {"message": str(err)}}

            if event.wait(timeout=timeout):
                with self._lock:
                    return self._pending_responses.pop(req_id, {})
            else:
                # Timeout: clean up the stale entry
                with self._lock:
                    self._pending_events.pop(req_id, None)
                    self._pending_responses.pop(req_id, None)
                return {"error": {"message": f"Timeout waiting for {method} response"}}

        except Exception as e:
            # Clean up on error
            if req_id is not None:
                with self._lock:
                    self._pending_events.pop(req_id, None)
                    self._pending_responses.pop(req_id, None)
            return {"error": {"message": str(e)}}

    def _send_notification(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC notification (no id, no reply expected).

        Used for 'notifications/initialized'. Returns an error dict on write
        failure.
        """
        if not self.process or self.process.poll() is not None:
            return {"error": {"message": f"Server {self.name} not running"}}

        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        err = self._send_raw(notification)
        return {"error": {"message": str(err)}} if err is not None else {}

    # ------------------------------------------------------------------
    # MCP protocol
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Initialize the MCP server and get capabilities."""
        response = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ognee", "version": "1.0.0"}
        })

        if "error" in response:
            print_warning(f"  ⚠️  {self.name} init error: {response['error'].get('message', response['error'])}")
            return False

        # Notifications have no 'id' and no response, so they must go through
        # _send_notification() — _send_request() would time out.
        self._send_notification("notifications/initialized", {})
        return True

    def list_tools(self) -> list:
        """Get available tools from the server."""
        response = self._send_request("tools/list", {})

        if "error" in response:
            print_warning(f"  ⚠️  {self.name} tools/list error: {response['error'].get('message', response['error'])}")
            return []

        self.tools = response.get("result", {}).get("tools", [])
        return self.tools

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool call."""
        response = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        }, timeout=get_tool_timeout(self.name, tool_name))

        if "error" in response:
            return {"error": response["error"].get("message", str(response["error"]))}

        result = response.get("result", {})
        content = result.get("content", [])

        # Extract text content
        texts = []
        for item in content:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))

        out = {"result": "\n".join(texts) if texts else str(result)}
        # Propagate the MCP isError flag so tool-level errors are reported as
        # failures instead of being mistaken for successful output.
        if result.get("isError"):
            out["isError"] = True
        return out

    def is_running(self) -> bool:
        """Check if server is still running."""
        return (
            not self.dead
            and self.process is not None
            and self.process.poll() is None
        )

    def stop(self):
        """Stop the MCP server."""
        with self._lifecycle_lock:
            self._stop_locked()

    def _stop_locked(self):
        # Invalidate the current generation first so reader/stderr threads
        # from this incarnation exit and never mark a future incarnation dead.
        self._generation += 1
        self._running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
                # A killed child that is never waited becomes a zombie.
                try:
                    self.process.wait(timeout=5)
                except Exception:
                    pass
            finally:
                # The stderr drain thread owns the stderr pipe during normal
                # operation; here we just close the descriptors to prevent
                # resource leaks and unblock the reader/drain threads.
                for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                    try:
                        if pipe:
                            pipe.close()
                    except Exception:
                        pass

                self.process = None

            with self._lock:
                for event in self._pending_events.values():
                    event.set()
                self._pending_events.clear()
                self._pending_responses.clear()

    def __repr__(self):
        state = "running" if self.is_running() else "stopped"
        return f"<MCPServer {self.name} ({state})>"


# =============================================================================
# MCP HTTP SERVER (mcpo-style proxy transport)
# =============================================================================

class MCPHttpServer:
    """Attaches to an mcpo-style HTTP proxy entry (``{"url": ...}``).

    mcpo exposes each MCP server as an OpenAPI HTTP endpoint: tools are
    invoked with ``POST {url}/{tool_name}`` and a JSON argument body, and the
    tool list can be discovered from ``GET {url}/openapi.json``. Used when
    ``MCP_MODE=proxy`` so OGnee talks to an already-running proxy instead of
    double-spawning servers.
    """

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url.rstrip("/")
        self.tools: list = []
        self.command = url  # for status display parity with MCPServer
        self._session = requests.Session()
        self._up = False

    def start(self) -> bool:
        """Check the proxy endpoint is reachable and discover tools."""
        try:
            resp = self._session.get(f"{self.url}/openapi.json", timeout=5)
            resp.raise_for_status()
        except Exception as e:
            print_error(f"  ❌ {self.name}: proxy unreachable at {self.url}: {e}")
            return False

        self._up = True
        # Derive tool list from the OpenAPI paths (each path is a tool).
        try:
            spec = resp.json()
            self.tools = []
            for path, ops in (spec.get("paths") or {}).items():
                tool_name = path.strip("/").split("/")[0]
                if not tool_name:
                    continue
                op = (ops or {}).get("post") or {}
                self.tools.append({
                    "name": tool_name,
                    "description": op.get("summary") or op.get("description") or "",
                    "inputSchema": {"type": "object", "properties": {}},
                })
        except Exception:
            # Discovery is best-effort; calls may still work.
            self.tools = []
        return True

    def initialize(self) -> bool:
        """No MCP handshake over the proxy — reachability check is enough."""
        return self._up

    def list_tools(self) -> list:
        return self.tools

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Invoke a tool through the HTTP proxy."""
        if not self._up:
            return {"error": f"Server {self.name} not running"}
        timeout = get_tool_timeout(self.name, tool_name)
        try:
            resp = self._session.post(
                f"{self.url}/{tool_name}",
                json=arguments or {},
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            return {"error": f"Timeout after {timeout}s"}
        except Exception as e:
            self._up = False
            return {"error": str(e)}

        if resp.status_code >= 400:
            return {"result": resp.text[:2000], "isError": True}

        try:
            data = resp.json()
        except ValueError:
            return {"result": resp.text}

        # mcpo wraps MCP results in the usual content/isError shape.
        if isinstance(data, dict) and "content" in data:
            texts = [
                item.get("text", "")
                for item in data.get("content", [])
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            out = {"result": "\n".join(texts) if texts else json.dumps(data)}
            if data.get("isError"):
                out["isError"] = True
            return out
        return {"result": data if isinstance(data, str) else json.dumps(data)}

    def is_running(self) -> bool:
        return self._up

    def stop(self):
        """Nothing to stop — the proxy owns the server processes."""
        self._up = False
        self._session.close()

    def __repr__(self):
        state = "up" if self.is_running() else "down"
        return f"<MCPHttpServer {self.name} ({state}) {self.url}>"


# =============================================================================
# MCP MANAGER
# =============================================================================

class MCPManager:
    """Manages multiple MCP servers from config."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._find_config()
        self.servers: Dict[str, Any] = {}  # MCPServer | MCPHttpServer
        self.all_tools: list = []
        self.tool_to_server: Dict[str, str] = {}
        # Auto-restart bookkeeping: restart timestamps per server, and the
        # set of servers that exhausted their crash-loop budget.
        self._restart_times: Dict[str, List[float]] = {}
        self._down_servers: set = set()
        # Serializes stop/start/restart so concurrent auto-restarts can't
        # double-spawn a server (the "double-spawned playwright" problem).
        self._lifecycle_lock = threading.Lock()

    def _find_config(self) -> str:
        """Find the MCP config file.

        The MCP_CONFIG environment variable wins (e.g. pointing at the shared
        ~/tools/mcp-servers/mcp.json), then the bundled config, then known
        fallback locations.
        """
        env_config = os.environ.get("MCP_CONFIG")
        if env_config:
            expanded = Path(os.path.expanduser(env_config))
            if expanded.exists():
                return str(expanded)

        if MCP_CONFIG.exists():
            return str(MCP_CONFIG)
        if MCP_CONFIG_FALLBACK.exists():
            return str(MCP_CONFIG_FALLBACK)

        # Check common locations
        home = Path.home()
        candidates = [
            home / "tools" / "mcp-servers" / "mcp.json",
            home / ".config" / "LM-Studio" / "mcp.json",
            home / ".config" / "lm-studio" / "mcp.json",
            home / "towerstack" / "OGnee" / "mcp.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return str(MCP_CONFIG)  # Default, even if doesn't exist

    def load_config(self) -> bool:
        """Load MCP configuration from file.

        Supports both stdio entries (``command``/``args``/``env``/``cwd``)
        and mcpo-style proxy entries (``url``). Proxy entries are only
        attached when ``MCP_MODE=proxy``; in the default stdio mode they are
        skipped with a warning rather than failing with a confusing
        allowlist error.
        """
        try:
            with open(os.path.expanduser(self.config_path)) as f:
                config = json.load(f)

            mcp_servers = config.get("mcpServers", {})

            for name, server_config in mcp_servers.items():
                url = server_config.get("url", "")
                if url:
                    if MCP_MODE == "proxy":
                        self.servers[name] = MCPHttpServer(name, url)
                    else:
                        print_dim(
                            f"  ⏭ {name}: url-based entry skipped "
                            f"(MCP_MODE={MCP_MODE}; set MCP_MODE=proxy to attach)"
                        )
                    continue

                command = server_config.get("command", "")
                args = server_config.get("args", [])
                env = server_config.get("env", {})
                cwd = server_config.get("cwd")

                self.servers[name] = MCPServer(name, command, args, env, cwd=cwd)

            print_system(f"📋 Loaded {len(self.servers)} MCP servers from {Path(self.config_path).name}")
            return True

        except FileNotFoundError:
            print_error(f"❌ Config not found: {self.config_path}")
            return False
        except json.JSONDecodeError as e:
            print_error(f"❌ Invalid JSON in config: {e}")
            return False

    def _register_tools(self, name: str, server) -> int:
        """(Re)register a server's tools in the manager indices."""
        # Drop stale entries for this server first (restart-safe).
        self.all_tools = [t for t in self.all_tools if t.get("server") != name]
        self.tool_to_server = {
            k: v for k, v in self.tool_to_server.items() if v != name
        }
        tools = server.list_tools()
        for tool in tools:
            tool_name = tool.get("name", "")
            # Prefix with server name for namespacing
            full_name = f"{name}/{tool_name}"
            tool["full_name"] = full_name
            tool["server"] = name
            self.all_tools.append(tool)
            self.tool_to_server[full_name] = name
        return len(tools)

    def start_all(self) -> int:
        """Start all MCP servers and collect tools."""
        print_system("\n🚀 Starting MCP servers...")
        started = 0

        for name, server in self.servers.items():
            if server.start():
                if server.initialize():
                    count = self._register_tools(name, server)
                    print_success(f"  ✅ {name}: {count} tools")
                    started += 1
                else:
                    print_warning(f"  ⚠️  {name}: started but failed to initialize")
            else:
                print_error(f"  ❌ {name}: failed to start")

        print_system(f"\n📦 Total: {len(self.all_tools)} tools from {started} servers")
        return started

    def start_server(self, name: str) -> bool:
        """Start a specific server."""
        if name not in self.servers:
            print_error(f"Unknown server: {name}")
            return False

        server = self.servers[name]
        if server.is_running():
            print_warning(f"{name} is already running")
            return True

        if server.start() and server.initialize():
            count = self._register_tools(name, server)
            self._down_servers.discard(name)
            print_success(f"✅ {name} started: {count} tools")
            return True

        return False

    def stop_server(self, name: str):
        """Stop a specific server."""
        if name in self.servers:
            self.servers[name].stop()
            # Remove tools from this server
            self.all_tools = [t for t in self.all_tools if t.get("server") != name]
            self.tool_to_server = {k: v for k, v in self.tool_to_server.items() if v != name}
            print_system(f"⏹ {name} stopped")

    def restart_server(self, name: str) -> bool:
        """Restart a specific server (manual restart — no budget check).

        Serialized under the lifecycle lock so concurrent restarts (e.g. two
        threads hitting a dead server at once) can't double-spawn it.
        """
        with self._lifecycle_lock:
            self.stop_server(name)
            time.sleep(0.5)
            return self.start_server(name)

    # ------------------------------------------------------------------
    # Crash-loop-budgeted auto-restart
    # ------------------------------------------------------------------

    def _restart_allowed(self, name: str) -> bool:
        """Check the crash-loop budget: at most MAX_RESTARTS restarts per
        RESTART_WINDOW_SECONDS; beyond that the server is marked down."""
        if name in self._down_servers:
            return False
        now = time.monotonic()
        times = [t for t in self._restart_times.get(name, [])
                 if now - t < RESTART_WINDOW_SECONDS]
        self._restart_times[name] = times
        if len(times) >= MAX_RESTARTS:
            self._down_servers.add(name)
            print_error(
                f"  ⛔ {name}: crash-loop budget exhausted "
                f"({MAX_RESTARTS} restarts in {RESTART_WINDOW_SECONDS // 60}min) — "
                f"server marked down for this session"
            )
            return False
        times.append(now)
        return True

    def _auto_restart(self, name: str) -> bool:
        """Try to restart a dead server within the crash-loop budget."""
        server = self.servers.get(name)
        # Proxy (HTTP) servers are managed externally — just re-probe.
        if isinstance(server, MCPHttpServer):
            return server.start()
        if not self._restart_allowed(name):
            return False
        print_warning(f"  ♻️ {name}: server died, attempting auto-restart...")
        ok = self.restart_server(name)
        if not ok and name in self.servers:
            tail = "".join(getattr(self.servers[name], "stderr_tail", []))
            if tail:
                print_dim(f"  last stderr from {name}:\n{tail[-400:]}")
        return ok

    def call_tool(self, full_name: str, arguments: dict) -> dict:
        """Route tool call to appropriate server."""
        server_name = self.tool_to_server.get(full_name)

        if not server_name:
            # Try without namespace prefix
            for registered_name, srv_name in self.tool_to_server.items():
                if registered_name.endswith(f"/{full_name}") or registered_name == full_name:
                    server_name = srv_name
                    full_name = registered_name
                    break

        if not server_name:
            return {"error": f"Unknown tool: {full_name}"}

        server = self.servers.get(server_name)
        if not server:
            return {"error": f"Server not found: {server_name}"}

        if not server.is_running():
            if not self._auto_restart(server_name):
                if server_name in self._down_servers:
                    return {"error": (
                        f"Server {server_name} is down (crash-loop budget "
                        f"exhausted). Do not retry its tools this session."
                    )}
                return {"error": f"Server not running: {server_name}"}
            server = self.servers.get(server_name)

        # Extract tool name without namespace
        tool_name = full_name.split("/")[-1] if "/" in full_name else full_name

        return server.call_tool(tool_name, arguments)

    # ------------------------------------------------------------------
    # Schemas and prompts
    # ------------------------------------------------------------------

    def native_name_map(self) -> Dict[str, str]:
        """Map OpenAI-legal native function names back to ``server/tool``."""
        return {
            make_native_name(t.get("server", ""), t.get("name", "")): t.get("full_name", "")
            for t in self.all_tools
        }

    def get_tools_schema(self) -> list:
        """Get tools in OpenAI function calling format.

        Names are emitted as ``server__tool`` (OpenAI's function-name grammar
        forbids ``/``); use ``native_name_map()`` to map back on dispatch.
        """
        tools = []
        for tool in self.all_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": make_native_name(tool.get("server", ""), tool.get("name", "")),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})
                }
            })
        return tools

    def get_tools_prompt(self, profile: str = "json",
                         extra_tools: Optional[list] = None) -> str:
        """Generate the tool instructions for the system prompt.

        Exactly ONE call format is described, chosen from the model's tool
        profile — offering a menu of formats makes weak local models mix
        syntaxes and produce unparseable calls.

        Args:
            profile: one of "native", "qwen-xml", "json", "gemma".
            extra_tools: additional tool dicts (e.g. built-in local tools)
                         with name/description/inputSchema keys.
        """
        all_tools = list(self.all_tools)
        for t in (extra_tools or []):
            entry = dict(t)
            entry.setdefault("server", entry.get("full_name", "local/").split("/")[0])
            entry.setdefault("full_name", f"{entry['server']}/{entry.get('name', '')}")
            all_tools.append(entry)

        if not all_tools:
            return ""

        if profile == "native":
            # Tools are sent via the API's `tools` parameter; the model only
            # needs a nudge to actually use them.
            return (
                "\n## Tools\n"
                "You have tools available via native function calling. "
                "Always act with tool calls instead of describing actions. "
                "Use the exact function names and required parameters provided "
                "in the tool definitions. After tool results arrive, summarize "
                "what you found in plain text before making further calls.\n"
            )

        format_block = {
            "qwen-xml": (
                "To use a tool, output EXACTLY this XML-wrapped JSON format:\n"
                '<tool_call>{"tool": "server/name", "arguments": {"arg1": "value1"}}</tool_call>\n'
            ),
            "gemma": (
                "To use a tool, output EXACTLY this format:\n"
                '<start_function_call>call:server/name{"arg1": "value1"}<end_function_call>\n'
            ),
            # "json" and unknown profiles fall back to a plain JSON object.
            "json": (
                "To use a tool, output EXACTLY this JSON object format:\n"
                '{"tool": "server/name", "arguments": {"arg1": "value1"}}\n'
            ),
        }.get(profile, (
            "To use a tool, output EXACTLY this JSON object format:\n"
            '{"tool": "server/name", "arguments": {"arg1": "value1"}}\n'
        ))

        lines = [
            "\n## Tools",
            format_block,
            "Rules:",
            "- Use ONLY the exact `server/name` identifiers listed below.",
            "- Output valid JSON arguments: double-quoted keys/strings, no trailing commas.",
            "- You may call multiple tools in one turn, one call per line.",
            "- After tool results arrive, summarize findings in plain text.",
            "",
        ]

        # Group by server, full schemas (compact menus caused models to
        # hallucinate parameters that were never shown to them).
        by_server: Dict[str, list] = {}
        for tool in all_tools:
            by_server.setdefault(tool.get("server", "unknown"), []).append(tool)

        for server, tools in by_server.items():
            lines.append(f"### {server}/")
            for tool in tools:
                name = tool.get("name", "")
                desc = tool.get("description", "No description")
                schema = tool.get("inputSchema", {})
                props = schema.get("properties", {})
                required = schema.get("required", [])

                lines.append(f"- **{server}/{name}**: {desc}")
                if props:
                    for prop_name, prop_info in props.items():
                        req = " (required)" if prop_name in required else ""
                        prop_type = prop_info.get("type", "any")
                        prop_desc = prop_info.get("description", "")
                        lines.append(f"    - `{prop_name}` ({prop_type}){req}: {prop_desc}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Status / display
    # ------------------------------------------------------------------

    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers."""
        status = {}
        for name, server in self.servers.items():
            status[name] = {
                "running": server.is_running(),
                "tools": len(server.tools),
                "command": server.command,
                "down": name in self._down_servers,
            }
        return status

    def print_status(self):
        """Print status of all MCP servers."""
        print()
        print_system("MCP Server Status")
        print_system("─" * 40)

        for name, server in self.servers.items():
            running = server.is_running()
            status = f"{COLORS.SUCCESS}●{COLORS.RESET}" if running else f"{COLORS.ERROR}●{COLORS.RESET}"
            tool_count = len(server.tools) if running else 0
            suffix = " [DOWN - crash-loop budget exhausted]" if name in self._down_servers else ""

            print(f"  {status} {COLORS.TOOL_NAME}{name:15}{COLORS.RESET} {tool_count} tools{suffix}")

        print()
        print_system(f"Total: {len(self.all_tools)} tools available")
        print()

    def print_tools(self):
        """Print all available tools grouped by server."""
        print()
        print_system("Available Tools")
        print_system("─" * 40)

        # Group by server
        by_server = {}
        for tool in self.all_tools:
            server = tool.get("server", "unknown")
            if server not in by_server:
                by_server[server] = []
            by_server[server].append(tool)

        for server, tools in by_server.items():
            print(f"\n  {COLORS.TOOL_NAME}{server}/{COLORS.RESET}")
            for tool in tools:
                name = tool.get("name", "")
                desc = tool.get("description", "")[:50]
                print(f"    • {name}: {desc}...")

        print()

    def stop_all(self):
        """Stop all MCP servers."""
        print_system("\n🛑 Stopping MCP servers...")
        for name, server in self.servers.items():
            server.stop()
            print_dim(f"  ⏹ {name}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_mcp(config_path: str = None) -> MCPManager:
    """Load and start MCP servers from config."""
    manager = MCPManager(config_path)
    if manager.load_config():
        manager.start_all()
    return manager

def get_tool_list(manager: MCPManager) -> List[str]:
    """Get flat list of all tool names."""
    return [t.get("full_name", t.get("name", "")) for t in manager.all_tools]

def find_tool(manager: MCPManager, name: str) -> Optional[Dict[str, Any]]:
    """Find a tool by name (partial match allowed)."""
    name_lower = name.lower()

    for tool in manager.all_tools:
        full_name = tool.get("full_name", "").lower()
        tool_name = tool.get("name", "").lower()

        if name_lower == full_name or name_lower == tool_name:
            return tool
        if name_lower in full_name or name_lower in tool_name:
            return tool

    return None
