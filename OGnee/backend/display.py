#!/usr/bin/env python3
"""
display.py - Terminal output formatting for OGnee

Colors, status bar, the pretty stuff.
Makes tool calls dim, conversation bright, errors red, etc.
"""

import re
import sys
import time
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from .config import COLORS, HOSTNAME

# =============================================================================
# USER INPUT (prompt_toolkit primary, readline-safe fallback)
# =============================================================================
#
# Raw ANSI escape codes in an input() prompt corrupt GNU readline's cursor
# math (every escape byte is counted as a printable column), which is what
# made typed input wrap on top of itself. prompt_toolkit handles colors and
# widths correctly and gives us proper line editing. If prompt_toolkit is
# not installed we fall back to input() with the escape sequences wrapped
# in readline's \001/\002 "ignore" markers so the cursor math stays correct.

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI as _PTANSI
    _HAS_PROMPT_TOOLKIT = True
except ImportError:  # pragma: no cover - depends on environment
    PromptSession = None
    _PTANSI = None
    _HAS_PROMPT_TOOLKIT = False

_prompt_session: "Optional[PromptSession]" = None

_ANSI_SEQ_RE = re.compile(r"(\x1b\[[0-9;]*m)")


def _get_prompt_session() -> "PromptSession":
    """Lazily create the shared PromptSession (multiline off by default)."""
    global _prompt_session
    if _prompt_session is None:
        _prompt_session = PromptSession(multiline=False)
    return _prompt_session


def _wrap_readline_escapes(prompt: str) -> str:
    """Wrap ANSI escapes in readline's \\001/\\002 non-printing markers."""
    return _ANSI_SEQ_RE.sub(lambda m: "\x01" + m.group(1) + "\x02", prompt)


def read_input(prompt: str = "") -> str:
    """Read one line of user input with an ANSI-colored prompt.

    Primary path: prompt_toolkit (correct width/color handling, persistent
    history). Fallback: builtin input() with \\001/\\002-wrapped escapes.
    Raises EOFError/KeyboardInterrupt just like input().
    """
    if _HAS_PROMPT_TOOLKIT and sys.stdin.isatty():
        return _get_prompt_session().prompt(_PTANSI(prompt))
    return input(_wrap_readline_escapes(prompt))


# =============================================================================
# CORE PRINTING
# =============================================================================

def print_color(text: str, color: str = "", end: str = "\n", flush: bool = True):
    """Print with color, auto-reset at end."""
    if color:
        sys.stdout.write(f"{color}{text}{COLORS.RESET}{end}")
    else:
        sys.stdout.write(f"{text}{end}")
    if flush:
        sys.stdout.flush()

def print_dim(text: str, end: str = "\n"):
    """Print internal/tool stuff - dim gray."""
    print_color(text, COLORS.INTERNAL, end)

def print_bright(text: str, end: str = "\n"):
    """Print conversation - bright white."""
    print_color(text, COLORS.CONVERSATION, end)

def print_warning(text: str, end: str = "\n"):
    """Print warnings - yellow."""
    print_color(text, COLORS.WARNING, end)

def print_error(text: str, end: str = "\n"):
    """Print errors - bright red."""
    print_color(text, COLORS.ERROR, end)

def print_system(text: str, end: str = "\n"):
    """Print system messages - cyan."""
    print_color(text, COLORS.SYSTEM, end)

def print_success(text: str, end: str = "\n"):
    """Print success - bright green."""
    print_color(text, COLORS.SUCCESS, end)

def print_command(text: str, end: str = "\n"):
    """Print slash commands - magenta."""
    print_color(text, COLORS.COMMAND, end)

def print_tool(tool_name: str, text: str, end: str = "\n"):
    """Print tool call info - tool name in blue, rest dim."""
    sys.stdout.write(f"{COLORS.TOOL_NAME}{tool_name}{COLORS.RESET} {COLORS.INTERNAL}{text}{COLORS.RESET}{end}")
    sys.stdout.flush()

def print_ognee(text: str, end: str = "\n"):
    """Print OGnee branding - magenta."""
    print_color(text, COLORS.OGNEE, end)

# =============================================================================
# FORMATTED SECTIONS
# =============================================================================

def header(text: str, char: str = "=", width: int = 70):
    """Print a section header."""
    line = char * width
    print_system(line)
    print_system(f" {text}")
    print_system(line)

def subheader(text: str, char: str = "-", width: int = 70):
    """Print a subsection header."""
    line = char * width
    print_dim(line)
    print_dim(f" {text}")
    print_dim(line)

def banner():
    """Print the OGnee startup banner."""
    print_ognee(r"""
   ____   _____                    
  / __ \ / ____|                   
 | |  | | |  __  _ __   ___  ___  
 | |  | | | |_ || '_ \ / _ \/ _ \ 
 | |__| | |__| || | | |  __/  __/ 
  \____/ \_____||_| |_|\___|\___| 
                                   
    """)
    print_system(f"  Running on {COLORS.HOST}{HOSTNAME}{COLORS.RESET}")
    print()

# =============================================================================
# STATUS BAR
# =============================================================================

@dataclass
class SessionStatus:
    """Current session state for status display."""
    start_time: datetime
    tokens_used: int = 0
    context_limit: int = 128_000
    tool_calls: int = 0
    tool_failures: int = 0
    model_name: str = "unknown"
    trust_tier: str = "supervised"
    paused: bool = False
    tools_enabled: bool = True

def format_duration(delta: timedelta) -> str:
    """Format timedelta as HH:MM:SS."""
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_tokens(used: int, limit: int) -> str:
    """Format token usage with color based on remaining."""
    remaining = limit - used
    pct = (used / limit) * 100 if limit > 0 else 0
    
    # Color based on how much runway left
    if remaining < 30_000:
        color = COLORS.ERROR
    elif remaining < 50_000:
        color = COLORS.WARNING
    else:
        color = COLORS.SUCCESS
    
    return f"{color}{used:,}{COLORS.RESET}/{limit:,} ({pct:.1f}%)"

def status_bar(status: SessionStatus) -> str:
    """Generate the status bar string."""
    elapsed = datetime.now() - status.start_time
    duration = format_duration(elapsed)
    tokens = format_tokens(status.tokens_used, status.context_limit)
    
    # Build status indicators
    indicators = []
    if status.paused:
        indicators.append(f"{COLORS.WARNING}⏸ PAUSED{COLORS.RESET}")
    if not status.tools_enabled:
        indicators.append(f"{COLORS.WARNING}🔒 TOOLS OFF{COLORS.RESET}")
    if status.tool_failures > 0:
        indicators.append(f"{COLORS.WARNING}⚠ {status.tool_failures} failures{COLORS.RESET}")
    
    indicator_str = " │ ".join(indicators) if indicators else ""
    
    bar = (
        f"{COLORS.SYSTEM}│ "
        f"⏱ {duration} │ "
        f"📊 {tokens} │ "
        f"🔧 {status.tool_calls} calls │ "
        f"🎭 {status.model_name} │ "
        f"🔐 {status.trust_tier}"
        f"{COLORS.RESET}"
    )
    
    if indicator_str:
        bar += f" │ {indicator_str}"
    
    bar += f"{COLORS.SYSTEM} │{COLORS.RESET}"
    
    return bar

def print_status(status: SessionStatus):
    """Print the status bar (in place; erases to end-of-line first)."""
    bar = status_bar(status)
    sys.stdout.write(f"\r{bar}\033[K")
    sys.stdout.flush()

def print_status_line(status: SessionStatus):
    """Print status bar on its own line."""
    bar = status_bar(status)
    print(bar)

# =============================================================================
# TOOL CALL FORMATTING
# =============================================================================

def format_tool_call(server: str, tool: str, args: dict, iteration: int) -> str:
    """Format a tool call for display — full args, no truncation."""
    # Show the full arguments so the user can see exactly what the model
    # is requesting before approving/denying at the gate.
    args_str = str(args)
    # Replace newlines so it doesn't break the single-line display format
    args_str = args_str.replace("\n", "↵ ")
    
    return (
        f"{COLORS.DIM}[{iteration}] "
        f"{COLORS.TOOL_NAME}{server}/{tool}{COLORS.RESET} "
        f"{COLORS.INTERNAL}{args_str}{COLORS.RESET}"
    )

def print_tool_call(server: str, tool: str, args: dict, iteration: int):
    """Print a tool call."""
    formatted = format_tool_call(server, tool, args, iteration)
    print(formatted)

def format_tool_result(result: str, success: bool = True, max_len: int = 200) -> str:
    """Format a tool result for display."""
    color = COLORS.DIM if success else COLORS.WARNING
    
    # Truncate if too long
    if len(result) > max_len:
        result = result[:max_len - 3] + "..."
    
    # Replace newlines with visible marker
    result = result.replace("\n", "↵ ")
    
    return f"{color}  → {result}{COLORS.RESET}"

def print_tool_result(result: str, success: bool = True):
    """Print a tool result."""
    formatted = format_tool_result(result, success)
    print(formatted)

def print_tool_error(tool: str, error: str):
    """Print a tool error."""
    print_error(f"  ✗ {tool}: {error}")

def print_tool_timeout(tool: str, timeout: int):
    """Print a tool timeout."""
    print_warning(f"  ⏰ {tool}: timed out after {timeout}s")

def print_tool_denied(tool: str, reason: str):
    """Print a tool denial."""
    print_warning(f"  🚫 {tool}: {reason}")

# =============================================================================
# GATE / APPROVAL PROMPTS
# =============================================================================

def prompt_approval(action: str, details: str, options: list[str] = None) -> str:
    """Prompt user for approval with options."""
    if options is None:
        options = ["y", "n"]
    
    print()
    print_warning("┌─ APPROVAL REQUIRED " + "─" * 50)
    print_warning(f"│ Action: {action}")
    print_warning(f"│ {details}")
    print_warning("│")
    
    options_str = "/".join(options)
    print_warning(f"│ Options: [{options_str}]")
    print_warning("└" + "─" * 69)
    
    try:
        response = read_input(f"{COLORS.COMMAND}  > {COLORS.RESET}").strip().lower()
        # Fail CLOSED: empty input defaults to deny, never approve.
        return response if response else "n"
    except (EOFError, KeyboardInterrupt):
        print()
        return "n"

def prompt_choice(title: str, options: dict[str, str]) -> str:
    """Prompt user to choose from numbered options.

    SECURITY: fails CLOSED. Trust-gate option dicts put "Allow" at "1" and
    "Deny" at "2"; empty input, EOF, and Ctrl-C all resolve to the deny
    option, and unrecognized input re-prompts instead of silently picking
    option 1. The deny key is "2" when present, otherwise the last key.
    """
    print()
    print_warning(f"┌─ {title} " + "─" * (60 - len(title)))

    for key, desc in options.items():
        print_warning(f"│  [{key}] {desc}")

    print_warning("└" + "─" * 69)

    keys = list(options.keys())
    deny_key = "2" if "2" in options else keys[-1]

    while True:
        try:
            response = read_input(f"{COLORS.COMMAND}  Choice: {COLORS.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            # Panic-Ctrl-C / piped stdin at a security gate => deny.
            print()
            return deny_key
        if not response:
            # Enter at a security gate => deny, never allow.
            return deny_key
        if response in options:
            return response
        print_warning(f"  Unrecognized choice '{response}' — enter one of: {', '.join(keys)}")

# =============================================================================
# CONVERSATION FORMATTING
# =============================================================================

def print_ognee_response(text: str):
    """Print OGneey response."""
    print(f"\n{COLORS.OGNEE}OGneey:{COLORS.RESET} {COLORS.CONVERSATION}{text}{COLORS.RESET}")

def print_ognee_thinking():
    """Print thinking indicator."""
    print_dim("  🤔 thinking...")

def print_ognee_acting(action: str):
    """Print action indicator."""
    print_dim(f"  ⚡ {action}")

# =============================================================================
# SESSION EVENTS
# =============================================================================

def print_session_start(model: str, tier: str):
    """Print session start message."""
    print()
    print_system("─" * 70)
    print_success(f"  Session started")
    print_system(f"  Model: {COLORS.BRIGHT_WHITE}{model}{COLORS.RESET}")
    print_system(f"  Trust: {COLORS.BRIGHT_WHITE}{tier}{COLORS.RESET}")
    print_system(f"  Time:  {COLORS.BRIGHT_WHITE}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{COLORS.RESET}")
    print_system("─" * 70)
    print()

def print_session_end(status: SessionStatus, reason: str = "user exit"):
    """Print session end summary."""
    elapsed = datetime.now() - status.start_time
    duration = format_duration(elapsed)
    
    print()
    print_system("─" * 70)
    print_system(f"  Session ended: {reason}")
    print_system(f"  Duration:      {duration}")
    print_system(f"  Tokens used:   {status.tokens_used:,} / {status.context_limit:,}")
    print_system(f"  Tool calls:    {status.tool_calls} ({status.tool_failures} failures)")
    print_system("─" * 70)
    print()

def print_hard_stop(tokens_remaining: int, threshold: int):
    """Print hard stop warning."""
    print()
    print_error("╔" + "═" * 68 + "╗")
    print_error("║  ⚠️  HARD STOP - CONTEXT LIMIT APPROACHING" + " " * 24 + "║")
    print_error("║" + " " * 68 + "║")
    print_error(f"║  Remaining tokens: {tokens_remaining:,}".ljust(69) + "║")
    print_error(f"║  Threshold:        {threshold:,}".ljust(69) + "║")
    print_error("║" + " " * 68 + "║")
    print_error("║  Session will now log and terminate.".ljust(69) + "║")
    print_error("║  Continuation scroll will be saved.".ljust(69) + "║")
    print_error("╚" + "═" * 68 + "╝")
    print()

def print_context_warning(tokens_remaining: int, threshold: int):
    """Print context warning (not yet hard stop)."""
    print()
    print_warning(f"  ⚠️  Context warning: {tokens_remaining:,} tokens remaining (threshold: {threshold:,})")
    print()

# =============================================================================
# SLASH COMMAND HELP
# =============================================================================

def print_commands_help():
    """Print available slash commands."""
    print()
    print_system("Available commands:")
    print()
    
    sections = [
        ("Control", [
            ("/pause", "Halt agent, wait for input"),
            ("/resume", "Resume from pause"),
            ("/quit", "End session gracefully"),
            ("/continue", "Load continuation from previous session"),
        ]),
        ("Tools", [
            ("/tools-off", "Disable all tool calls (chat-only mode)"),
            ("/tools-on", "Enable tool calls"),
            ("/disable <tool>", "Disable specific tool or server"),
            ("/enable <tool>", "Re-enable a disabled tool"),
            ("/disabled", "List disabled tools"),
            ("/tools-list", "List all available tools"),
        ]),
        ("Voice", [
            ("/speak-on", "Enable TTS output"),
            ("/speak-off", "Disable TTS output"),
            ("/say <text>", "Manually speak text"),
        ]),
        ("Plan Mode", [
            ("/plan", "Enter plan mode (propose, don't execute)"),
            ("/execute", "Execute the pending plan"),
            ("/clear", "Discard plan and exit plan mode"),
            ("/plan-off", "Exit plan mode without executing"),
            ("/step", "Execute one tool call, then pause"),
        ]),
        ("Shadow Files", [
            ("/diff", "Show pending shadow file changes"),
            ("/apply", "Apply shadow changes to real files"),
            ("/discard-shadow", "Discard all shadow changes"),
        ]),
        ("Session", [
            ("/status", "Show current session status"),
            ("/tokens", "Show detailed token usage"),
            ("/recap", "Request session summary from OGneey"),
            ("/save", "Force save session checkpoint"),
            ("/prompt", "View/modify system prompt"),
        ]),
        ("Remote", [
            ("/ssh <host> <cmd>", "Run command on remote host"),
            ("/hosts", "List available hosts"),
        ]),
        ("Other", [
            ("/tier [0-3]", "View or change trust tier"),
            ("/help", "Show this help"),
        ]),
    ]
    
    for section_name, commands in sections:
        print(f"  {COLORS.BOLD}{section_name}{COLORS.RESET}")
        for cmd, desc in commands:
            print(f"    {COLORS.COMMAND}{cmd:20}{COLORS.RESET} {desc}")
        print()

# =============================================================================
# UTILITY
# =============================================================================

def clear_line():
    """Clear the current line (uses ANSI erase-line, not hardcoded width)."""
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

def move_up(lines: int = 1):
    """Move cursor up N lines."""
    sys.stdout.write(f"\033[{lines}A")
    sys.stdout.flush()

def hide_cursor():
    """Hide the cursor."""
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

def show_cursor():
    """Show the cursor."""
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()

def get_terminal_width() -> int:
    """Get terminal width, default 80 if can't detect."""
    try:
        import shutil
        return shutil.get_terminal_size().columns
    except Exception:
        return 80
