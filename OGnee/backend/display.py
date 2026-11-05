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
# USER INPUT
# =============================================================================
#
# All terminal input goes through inputhub (readline by default — the old
# fallback never imported readline, which is why arrow keys printed ^[[D).
# read_input() is kept for older call sites.

from .inputhub import get_hub, strip_ansi  # noqa: E402


def read_input(prompt: str = "") -> str:
    """Read one line from the operator (raises EOFError/KeyboardInterrupt)."""
    return get_hub().read_line(prompt)


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
    """y/n style approval (used by /tier and /apply). Fails closed to "n"."""
    if options is None:
        options = ["y", "n"]
    print()
    print_warning("┌─ APPROVAL REQUIRED " + "─" * 50)
    print_warning(f"│ Action: {action}")
    print_warning(f"│ {details}")
    print_warning("│")
    print_warning(f"│ Options: [{'/'.join(options)}]")
    print_warning("└" + "─" * 69)
    ans = get_hub().ask(f"{COLORS.COMMAND}  [{'/'.join(options)}] › {COLORS.RESET}", kind="yn")
    if ans.kind != "line":
        return "n"
    response = ans.text.strip().lower()
    if response in ("yes",):
        response = "y"
    if response in ("no",):
        response = "n"
    return response if response else "n"


def prompt_choice(title: str, options: dict[str, str]) -> str:
    """Numbered choice (legacy). Fails CLOSED: empty/EOF/Ctrl-C -> deny key."""
    print()
    print_warning(f"┌─ {title} " + "─" * max(4, 60 - len(title)))
    for key, desc in options.items():
        print_warning(f"│  [{key}] {desc}")
    print_warning("└" + "─" * 69)
    keys = list(options.keys())
    deny_key = "2" if "2" in options else keys[-1]
    while True:
        ans = get_hub().ask(f"{COLORS.COMMAND}  Choice › {COLORS.RESET}", kind="choice")
        if ans.kind != "line":
            return deny_key
        response = ans.text.strip()
        if not response:
            return deny_key
        if response in options:
            return response
        print_warning(f"  Unrecognized choice '{response}' — enter one of: {', '.join(keys)}")

# =============================================================================
# CONVERSATION FORMATTING
# =============================================================================

def print_ognee_response(text: str, response_id: Optional[int] = None):
    """Print OGneey response. With a response_id, a footer under the reply
    names the /regenerate command for it (every reply keeps its own)."""
    print(f"\n{COLORS.OGNEE}OGneey:{COLORS.RESET} {COLORS.CONVERSATION}{text}{COLORS.RESET}")
    if response_id is not None:
        print_dim(f"  #{response_id} · ↻ /regen {response_id}")

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

def print_commands_help(sections=None):
    """Print slash-command help. ``sections`` is built from the live command
    registry by commands.py: [(section, [(usage, aliases, description, midturn)])]."""
    print()
    print_system("Available commands  (⚡ = works while OGneey is running)")
    print()
    for section_name, rows in (sections or []):
        print(f"  {COLORS.BOLD}{section_name}{COLORS.RESET}")
        for usage, aliases, desc, midturn in rows:
            bolt = "⚡" if midturn else " "
            head = usage if len(usage) <= 22 else usage.split(" ")[0] + " …"
            shown = " ".join("/" + a for a in aliases[:4])
            if len(aliases) > 4:
                shown += f" +{len(aliases) - 4}"
            alias_txt = f"  {COLORS.DIM}{shown}{COLORS.RESET}" if aliases else ""
            print(f"   {bolt} {COLORS.COMMAND}{head:24}{COLORS.RESET} {desc}{alias_txt}")
        print()
    print_dim("  /help <command> for details and every alias.")
    print_dim("  Typing while OGneey works: plain text is delivered at the next turn boundary.")
    print_dim("  Ctrl-C while it works = /pf (stop now). Ctrl-C twice at an idle prompt = quit.")
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
