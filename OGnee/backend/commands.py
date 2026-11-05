#!/usr/bin/env python3
"""
commands.py - Slash command parser and handlers

/pause /resume /tools-off /tools-on /plan /execute /step
/status /tokens /diff /recap /save /quit /help

User control over the OGneey.
"""

import re
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List

import requests

from .context import (
    Session,
    SessionState,
    transition_state,
    resume_target,
    save_continuation_scroll,
)
from .display import (
    print_system,
    print_success,
    print_warning,
    print_error,
    print_commands_help,
    print_status_line,
    COLORS,
    prompt_approval,
)
from .session_log import get_latest_continuation, load_continuation
from .shadow import list_shadow_changes, apply_shadow_changes, clear_shadow
from .startup import build_system_prompt
from .tools import toggle_speak, CHATTERBOX_URL
from .output_buffer import get_output_buffer
from .config import TIERS, HOSTS, HOSTNAME

# =============================================================================
# SECURITY CONSTANTS
# =============================================================================

# FIX #4: Add maximum input length to prevent DoS / memory exhaustion.
MAX_INPUT_LENGTH = 100_000

# FIX #2: Maximum length for /prompt append text.
#MAX_PROMPT_APPEND = 1_000

# FIX #2: Patterns that indicate prompt-injection attempts.
_FORBIDDEN_PATTERNS = [
    re.compile(r'ignore\s+(all\s+)?previous\s+(instructions|directives)', re.IGNORECASE),
    re.compile(r'ignore\s+(all\s+)?rules', re.IGNORECASE),
    re.compile(r'<\s*/\s*system\s*>', re.IGNORECASE),
    re.compile(r'override\s+(all\s+)?(constraints|restrictions)', re.IGNORECASE),
]


# =============================================================================
# COMMAND TYPES
# =============================================================================

class CommandResult(Enum):
    """What happened after executing a command."""
    CONTINUE = auto()       # Keep going normally
    QUIT = auto()           # End session gracefully
    HANDLED = auto()        # Command handled, skip normal processing this turn
    NOT_A_COMMAND = auto()  # Input wasn't a command

@dataclass
class CommandResponse:
    """Response from a command handler."""
    result: CommandResult
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

# =============================================================================
# COMMAND REGISTRY
# =============================================================================

# Commands registered here
_commands: Dict[str, Callable[[Session, List[str]], CommandResponse]] = {}

def command(name: str, aliases: List[str] = None):
    """Decorator to register a command."""
    def decorator(func):
        _commands[name] = func
        for alias in (aliases or []):
            _commands[alias] = func
        return func
    return decorator

def is_command(text: str) -> bool:
    """Check if input is a slash command."""
    return text.strip().startswith("/")

def parse_command(text: str) -> tuple[str, List[str]]:
    """Parse command and arguments from input."""
    text = text.strip()
    if not text.startswith("/"):
        return "", []
    
    parts = text[1:].split()
    if not parts:
        return "", []
    
    cmd = parts[0].lower()
    args = parts[1:]
    
    return cmd, args

def execute_command(text: str, state: Session) -> CommandResponse:
    """Execute a slash command."""
    # FIX #3: Reject oversized input before any processing.
    if len(text) > MAX_INPUT_LENGTH:
        print_error(f"Input too long ({len(text):,} chars). Max: {MAX_INPUT_LENGTH:,}")
        return CommandResponse(result=CommandResult.HANDLED)

    if not is_command(text):
        return CommandResponse(result=CommandResult.NOT_A_COMMAND)
    
    cmd, args = parse_command(text)
    
    if cmd in _commands:
        try:
            return _commands[cmd](state, args)
        except Exception as e:
            print_error(f"Command error: {e}")
            return CommandResponse(result=CommandResult.CONTINUE)
    else:
        print_warning(f"Unknown command: /{cmd}")
        print_system("Type /help for available commands")
        return CommandResponse(result=CommandResult.HANDLED)

# =============================================================================
# CONTROL COMMANDS
# =============================================================================

@command("pause", aliases=["p", "stop", "halt", "wait"])
def cmd_pause(state: Session, args: List[str]) -> CommandResponse:
    """Pause the agent. While paused, only slash commands are processed;
    other input is queued and surfaced on /resume (never sent to the LLM
    and never dropped)."""
    if state.mode is SessionState.PAUSED:
        print_system("Already paused. Type /resume to continue.")
        return CommandResponse(result=CommandResult.HANDLED)
    transition_state(state, SessionState.PAUSED)
    print_success("⏸  Paused. Type /resume to continue.")
    return CommandResponse(result=CommandResult.HANDLED)

@command("resume", aliases=["r", "go"])
def cmd_resume(state: Session, args: List[str]) -> CommandResponse:
    """Resume from pause. Returns data={"resume": True} so the interactive
    loop can deliver any messages queued while paused — this command itself
    is never sent to the LLM as a task."""
    if state.mode is not SessionState.PAUSED:
        print_system("Not paused.")
        return CommandResponse(result=CommandResult.HANDLED)

    transition_state(state, resume_target(state))
    print_success("▶  Resumed.")
    return CommandResponse(
        result=CommandResult.HANDLED,
        data={"resume": True},
    )

@command("quit", aliases=["q", "exit", "bye"])
def cmd_quit(state: Session, args: List[str]) -> CommandResponse:
    """End session gracefully."""
    print_system("Ending session...")
    return CommandResponse(result=CommandResult.QUIT)

@command("continue", aliases=["load-continuation", "cont"])
def cmd_pickup(state: Session, args: List[str]) -> CommandResponse:
    """Load the most recent continuation scroll into context."""
    # FIX #5: Late imports moved to top of file.
    latest = get_latest_continuation()
    if not latest:
        print_system("No continuation scroll found.")
        return CommandResponse(result=CommandResult.HANDLED)
    
    mtime = datetime.fromtimestamp(latest.stat().st_mtime)
    age_minutes = int((datetime.now() - mtime).total_seconds() / 60)
    
    content = load_continuation(latest)
    if not content:
        print_error("Failed to load continuation.")
        return CommandResponse(result=CommandResult.HANDLED)
    
    print_success(f"📜 Loaded continuation: {latest.name} ({age_minutes}m ago)")
    
    # Inject continuation as context for next message
    return CommandResponse(
        result=CommandResult.CONTINUE,
        data={"inject_user_message": f"Here is context from my previous session:\n\n{content}\n\nPlease acknowledge and continue from where we left off."}
    )

# =============================================================================
# TOOL CONTROL
# =============================================================================

@command("speak-on", aliases=["speakon", "voice-on", "tts-on"])
def cmd_speak_on(state: Session, args: List[str]) -> CommandResponse:
    """Enable the speak tool."""
    # FIX #5: Late import moved to top of file.
    toggle_speak(True)
    print_success("🔊 Speak enabled. OGneey can now use local/speak tool.")
    return CommandResponse(result=CommandResult.HANDLED)

@command("speak-off", aliases=["speakoff", "voice-off", "tts-off", "stfu", "shutup"])
def cmd_speak_off(state: Session, args: List[str]) -> CommandResponse:
    """Disable the speak tool."""
    # FIX #5: Late import moved to top of file.
    toggle_speak(False)
    print_success("🔇 Speak disabled. OGneey will not use TTS.")
    return CommandResponse(result=CommandResult.HANDLED)

@command("say", aliases=["speak"])
def cmd_say(state: Session, args: List[str]) -> CommandResponse:
    """Manually send text to Chatterbox. Usage: /say Hello world"""
    if not args:
        print_warning("Usage: /say <text to speak>")
        return CommandResponse(result=CommandResult.HANDLED)
    
    text = " ".join(args)

    # FIX #1: asyncio.run() crashes when called inside an already-running event
    # loop (interactive_loop is async). Since call_speak just does a sync
    # requests.post, call requests.post directly here instead of using
    # asyncio.run(call_speak(...)).
    try:
        response = requests.post(
            CHATTERBOX_URL,
            json={"text": text},
            timeout=10,
        )
        if response.status_code == 200:
            print_success(f"  [SPOKEN] {len(text)} chars sent to Chatterbox")
        else:
            print_error(f"  Chatterbox returned {response.status_code}")
    except requests.exceptions.ConnectionError:
        print_error(f"  Chatterbox not running at {CHATTERBOX_URL}")
    except Exception as e:
        print_error(f"  Speak failed: {e}")
    
    return CommandResponse(result=CommandResult.HANDLED)

@command("tools-off", aliases=["toolsoff", "notool", "notools", "chat"])
def cmd_tools_off(state: Session, args: List[str]) -> CommandResponse:
    """Disable all tool calls - chat only mode."""
    state.tools_enabled = False
    print_success("🔒 Tools disabled. OGneey is now chat-only.")
    return CommandResponse(result=CommandResult.HANDLED)

@command("tools-on", aliases=["toolson", "tools"])
def cmd_tools_on(state: Session, args: List[str]) -> CommandResponse:
    """Enable tool calls."""
    state.tools_enabled = True
    print_success("🔓 Tools enabled.")
    return CommandResponse(result=CommandResult.HANDLED)

@command("disable", aliases=["dis"])
def cmd_disable_tool(state: Session, args: List[str]) -> CommandResponse:
    """Disable a specific tool or server. Usage: /disable shell or /disable local-files/write_file"""
    if not args:
        print_warning("Usage: /disable <tool> or /disable <server/tool>")
        print_system("Example: /disable shell")
        print_system("Example: /disable local-files/write_file")
        return CommandResponse(result=CommandResult.HANDLED)
    
    target = args[0]
    
    # Store disabled tools in state
    if not hasattr(state, 'disabled_tools'):
        state.disabled_tools = set()
    
    state.disabled_tools.add(target)
    print_success(f"🚫 Disabled: {target}")
    return CommandResponse(result=CommandResult.HANDLED)

@command("enable", aliases=["en"])
def cmd_enable_tool(state: Session, args: List[str]) -> CommandResponse:
    """Re-enable a specific tool or server."""
    if not args:
        print_warning("Usage: /enable <tool>")
        return CommandResponse(result=CommandResult.HANDLED)
    
    target = args[0]
    
    if hasattr(state, 'disabled_tools') and target in state.disabled_tools:
        state.disabled_tools.remove(target)
        print_success(f"✓ Enabled: {target}")
    else:
        print_system(f"{target} was not disabled")
    
    return CommandResponse(result=CommandResult.HANDLED)

@command("disabled", aliases=["dis-list"])
def cmd_list_disabled(state: Session, args: List[str]) -> CommandResponse:
    """List disabled tools."""
    if not hasattr(state, 'disabled_tools') or not state.disabled_tools:
        print_system("No tools are disabled.")
    else:
        print_system("Disabled tools:")
        for t in sorted(state.disabled_tools):
            print_system(f"  🚫 {t}")
    return CommandResponse(result=CommandResult.HANDLED)

# =============================================================================
# PROMPT CONTROL
# =============================================================================

def _contains_injection(text: str) -> bool:
    """Check if text contains known prompt-injection patterns."""
    for pat in _FORBIDDEN_PATTERNS:
        if pat.search(text):
            return True
    return False

@command("prompt", aliases=["sys", "sysprompt", "system-prompt"])
def cmd_prompt(state: Session, args: List[str]) -> CommandResponse:
    """View or modify system prompt. Usage: /prompt [view|append <text>|clear|show-full]"""
    if not args:
        args = ["view"]
    
    action = args[0].lower()
    
    if action == "view":
        custom = getattr(state, 'custom_prompt', '')
        if custom:
            print_system("Custom prompt additions:")
            print(custom)
        else:
            print_system("No custom prompt additions.")
        print_system("\nCommands: /prompt append <text> | /prompt clear | /prompt show-full")
        return CommandResponse(result=CommandResult.HANDLED)
    
    elif action == "append":
        if len(args) < 2:
            print_warning("Usage: /prompt append <text to add>")
            return CommandResponse(result=CommandResult.HANDLED)
        
        text = " ".join(args[1:])

#        # FIX #2: Limit append length to prevent DoS / prompt stuffing.
#        if len(text) > MAX_PROMPT_APPEND:
#            print_error(
#                f"Append text too long ({len(text):,} chars). "
#                f"Max: {MAX_PROMPT_APPEND:,} chars."
#            )
#            return CommandResponse(result=CommandResult.HANDLED)

        # FIX #2: Block known prompt-injection patterns.
        if _contains_injection(text):
            print_error("Blocked: prompt text matches a known injection pattern.")
            return CommandResponse(result=CommandResult.HANDLED)

        if not hasattr(state, 'custom_prompt'):
            state.custom_prompt = ""
        state.custom_prompt += f"\n{text}"
        print_success(f"✓ Appended: {text[:60]}{'...' if len(text) > 60 else ''}")
        return CommandResponse(result=CommandResult.HANDLED)

    elif action in ("clear", "reset"):
        state.custom_prompt = ""
        print_success("✓ Custom prompt cleared")
        return CommandResponse(result=CommandResult.HANDLED)
    
    elif action == "show-full":
        # Show the full system prompt that would be sent
        print_system("=" * 60)
        print_system("FULL SYSTEM PROMPT (would be sent to model):")
        print_system("=" * 60)
        
        # FIX #5: Late import moved to top of file.
        full = build_system_prompt(state)
        
        custom = getattr(state, 'custom_prompt', '')
        if custom:
            full += f"\n\n## Additional Instructions\n{custom}"
        
        print(full[:2000])
        if len(full) > 2000:
            print_system(f"\n... [{len(full) - 2000} more characters]")
        print_system("=" * 60)
        return CommandResponse(result=CommandResult.HANDLED)
    
    else:
        print_warning(f"Unknown: {action}")
        print_system("Usage: /prompt [view|append <text>|clear|show-full]")
        return CommandResponse(result=CommandResult.HANDLED)

# =============================================================================
# PLAN MODE
# =============================================================================

@command("plan", aliases=["propose"])
def cmd_plan_mode(state: Session, args: List[str]) -> CommandResponse:
    """Enter plan mode - OGneey proposes actions but doesn't execute."""
    transition_state(state, SessionState.PLAN_REVIEW)
    state.pending_plan = []
    print_success("📋 Plan mode ON. OGneey will propose actions without executing.")
    print_system("   Use /execute to run the plan, /clear to discard.")
    return CommandResponse(result=CommandResult.HANDLED)

@command("execute", aliases=["exec", "run", "do"])
def cmd_execute_plan(state: Session, args: List[str]) -> CommandResponse:
    """Execute the pending plan.

    Returns HANDLED with data={"execute_plan": [...]} — the interactive
    loop runs the queue through the normal gated execution path. The
    literal string "/execute" is never sent to the LLM as a task.
    """
    if state.mode is not SessionState.PLAN_REVIEW and not state.pending_plan:
        print_warning("Not in plan mode. Use /plan to enter plan mode.")
        return CommandResponse(result=CommandResult.HANDLED)

    if not state.pending_plan:
        print_warning("No pending plan to execute. Exiting plan mode.")
        transition_state(state, SessionState.RUNNING)
        return CommandResponse(result=CommandResult.HANDLED)

    print_success(f"▶ Executing plan ({len(state.pending_plan)} steps)...")
    transition_state(state, SessionState.RUNNING)

    # Return the plan for execution
    return CommandResponse(
        result=CommandResult.HANDLED,
        data={"execute_plan": state.pending_plan}
    )

@command("clear", aliases=["discard"])
def cmd_clear_plan(state: Session, args: List[str]) -> CommandResponse:
    """Discard the pending plan and exit plan mode."""
    was_in_plan_mode = state.mode is SessionState.PLAN_REVIEW

    if state.pending_plan:
        count = len(state.pending_plan)
        state.pending_plan = []
        print_success(f"🗑  Discarded {count} planned steps.")

    if was_in_plan_mode:
        transition_state(state, SessionState.RUNNING)
        print_success("📋 Plan mode OFF.")
    elif not state.pending_plan:
        print_system("No pending plan.")

    return CommandResponse(result=CommandResult.HANDLED)

@command("plan-off", aliases=["planoff", "noplan"])
def cmd_plan_off(state: Session, args: List[str]) -> CommandResponse:
    """Exit plan mode without executing."""
    if state.mode is not SessionState.PLAN_REVIEW:
        print_system("Not in plan mode.")
        return CommandResponse(result=CommandResult.HANDLED)

    transition_state(state, SessionState.RUNNING)
    if state.pending_plan:
        count = len(state.pending_plan)
        state.pending_plan = []
        print_warning(f"🗑  Discarded {count} planned steps.")
    print_success("📋 Plan mode OFF.")
    return CommandResponse(result=CommandResult.HANDLED)

@command("step", aliases=["s", "next"])
def cmd_step(state: Session, args: List[str]) -> CommandResponse:
    """Execute one tool call, then pause. HANDLED — never sent to the LLM."""
    state.step_mode = True
    print_success("👟 Step mode: will pause after next tool call.")
    return CommandResponse(result=CommandResult.HANDLED)

# =============================================================================
# STATUS & INFO
# =============================================================================

@command("status", aliases=["stat", "st"])
def cmd_status(state: Session, args: List[str]) -> CommandResponse:
    """Show current session status."""
    print()
    print_status_line(state.to_status())
    print()
    
    # Additional details
    print_system(f"  Session ID:    {state.session_id}")
    print_system(f"  Model:         {state.model_name}")
    print_system(f"  Trust Tier:    {state.trust_tier}")
    print_system(f"  Elapsed:       {state.elapsed_formatted}")
    print_system(f"  Turn:          {state.turn_count}")
    print()
    
    # Modes
    modes = []
    if state.paused:
        modes.append("PAUSED")
    if not state.tools_enabled:
        modes.append("TOOLS OFF")
    if state.plan_mode:
        modes.append("PLAN MODE")
    if hasattr(state, 'step_mode') and state.step_mode:
        modes.append("STEP MODE")
    
    if modes:
        print_system(f"  Active modes:  {', '.join(modes)}")
        print()
    
    return CommandResponse(result=CommandResult.HANDLED)

@command("tokens", aliases=["tok", "context", "ctx"])
def cmd_tokens(state: Session, args: List[str]) -> CommandResponse:
    """Show detailed token usage."""
    print()
    print_system("Token Usage")
    print_system("─" * 40)
    
    limit = state.model_config.context_limit
    used = state.tokens_used
    remaining = state.tokens_remaining
    pct = (used / limit) * 100 if limit > 0 else 0
    
    # Color based on remaining
    if remaining < state.model_config.hard_stop_threshold:
        color = COLORS.ERROR
        status = "⚠️  CRITICAL"
    elif remaining < state.model_config.warning_threshold:
        color = COLORS.WARNING
        status = "⚠️  WARNING"
    else:
        color = COLORS.SUCCESS
        status = "✓ OK"
    
    print(f"  Total:       {color}{used:,}{COLORS.RESET} / {limit:,} ({pct:.1f}%)")
    print(f"  Remaining:   {color}{remaining:,}{COLORS.RESET}  {status}")
    print()
    print_system(f"  Conversation: ~{state.tokens_conversation:,}")
    print_system(f"  Tool calls:   ~{state.tokens_tool_calls:,}")
    print()
    print_system(f"  Runway:       ~{state.runway_turns} turns at current rate")
    print_system(f"  Hard stop:    {state.model_config.hard_stop_threshold:,} remaining")
    print()
    
    return CommandResponse(result=CommandResult.HANDLED)

@command("tools-list", aliases=["tl", "list-tools"])
def cmd_list_tools(state: Session, args: List[str]) -> CommandResponse:
    """List available tools (if we have tool info in state)."""
    if not hasattr(state, 'available_tools') or not state.available_tools:
        print_system("Tool list not available. Tools are loaded at startup.")
        return CommandResponse(result=CommandResult.HANDLED)
    
    print()
    print_system("Available Tools")
    print_system("─" * 40)
    
    for server, tools in state.available_tools.items():
        disabled = hasattr(state, 'disabled_tools') and server in state.disabled_tools
        status = f"{COLORS.WARNING}(disabled){COLORS.RESET}" if disabled else ""
        print(f"\n  {COLORS.TOOL_NAME}{server}{COLORS.RESET} {status}")
        for tool in tools:
            tool_key = f"{server}/{tool}"
            tool_disabled = hasattr(state, 'disabled_tools') and tool_key in state.disabled_tools
            if tool_disabled:
                print(f"    {COLORS.DIM}🚫 {tool}{COLORS.RESET}")
            else:
                print(f"    • {tool}")
    
    print()
    return CommandResponse(result=CommandResult.HANDLED)

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

@command("save", aliases=["checkpoint"])
def cmd_save(state: Session, args: List[str]) -> CommandResponse:
    """Force save session log and state."""
    print_system("Saving session...")
    
    # Save continuation scroll
    scroll_path = save_continuation_scroll(state, "Manual save checkpoint")
    print_success(f"✓ Saved continuation: {scroll_path.name}")
    
    return CommandResponse(result=CommandResult.HANDLED)

@command("recap", aliases=["summary", "sum"])
def cmd_recap(state: Session, args: List[str]) -> CommandResponse:
    """Request a recap of the current session."""
    print_system("Requesting session recap from OGneey...")
    
    # Return a signal to inject a recap request into the conversation
    return CommandResponse(
        result=CommandResult.CONTINUE,
        data={"inject_user_message": "Please provide a brief recap of what we've done this session so far."}
    )

@command("diff", aliases=["changes", "pending"])
def cmd_diff(state: Session, args: List[str]) -> CommandResponse:
    """Show pending shadow file changes."""
    # FIX #5: Late import moved to top of file.
    changes = list_shadow_changes()
    
    if not changes:
        print_system("No pending changes in shadow directory.")
        return CommandResponse(result=CommandResult.HANDLED)
    
    print()
    print_system("Pending Changes (shadow files)")
    print_system("─" * 40)
    
    for change in changes:
        print(f"  {COLORS.WARNING}M{COLORS.RESET} {change['shadow_path']}")
        print(f"    → {change['real_path']}")
    
    print()
    print_system(f"Total: {len(changes)} file(s)")
    print_system("Use /apply to apply changes, /discard-shadow to discard.")
    print()
    
    return CommandResponse(result=CommandResult.HANDLED)

@command("apply", aliases=["commit"])
def cmd_apply_shadow(state: Session, args: List[str]) -> CommandResponse:
    """Apply shadow changes to real files."""
    # FIX #5: Late imports moved to top of file.
    changes = list_shadow_changes()
    
    if not changes:
        print_system("No pending changes to apply.")
        return CommandResponse(result=CommandResult.HANDLED)
    
    # Show what will be applied
    print()
    print_warning("The following changes will be applied:")
    for change in changes:
        print(f"  {change['shadow_path']} → {change['real_path']}")
    print()
    
    response = prompt_approval(
        "Apply shadow changes",
        f"{len(changes)} file(s) will be modified",
        ["y", "n"]
    )
    
    if response == "y":
        results = apply_shadow_changes()
        applied = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        print_success(f"✓ Applied {len(applied)} change(s)")
        for path, _, msg in failed:
            print_error(f"  ✗ {path}: {msg}")
    else:
        print_system("Cancelled.")
    
    return CommandResponse(result=CommandResult.HANDLED)

@command("discard-shadow", aliases=["reset-shadow"])
def cmd_discard_shadow(state: Session, args: List[str]) -> CommandResponse:
    """Discard all shadow changes."""
    # FIX #5: Late imports moved to top of file.
    count = clear_shadow()
    if count > 0:
        print_success(f"🗑  Discarded {count} shadow file(s)")
    else:
        print_system("No shadow files to discard.")
    
    return CommandResponse(result=CommandResult.HANDLED)

# =============================================================================
# SSH / REMOTE
# =============================================================================

@command("ssh", aliases=["remote"])
def cmd_ssh(state: Session, args: List[str]) -> CommandResponse:
    """Run a command on a remote host. Usage: /ssh <host> <command>"""
    if len(args) < 2:
        print_warning("Usage: /ssh <host> <command>")
        print_system("Example: /ssh dacrib systemctl status pihole")
        return CommandResponse(result=CommandResult.HANDLED)
    
    host = args[0]
    command = " ".join(args[1:])
    
    # Signal to execute SSH command
    return CommandResponse(
        result=CommandResult.CONTINUE,
        data={"ssh_command": {"host": host, "command": command}}
    )

@command("hosts", aliases=["nodes"])
def cmd_hosts(state: Session, args: List[str]) -> CommandResponse:
    """List available hosts."""
    # FIX #5: Late import moved to top of file.
    print()
    print_system("Available Hosts")
    print_system("─" * 40)
    
    for name, config in HOSTS.items():
        current = " (current)" if name == HOSTNAME else ""
        gpu = "🎮" if config.get("has_gpu") else "  "
        lms = f"LMS:{config['lm_studio_port']}" if config.get("lm_studio_port") else ""
        
        print(f"  {gpu} {COLORS.HOST}{name:10}{COLORS.RESET} {config['ip']:15} {config['role']:10} {lms}{current}")
    
    print()
    return CommandResponse(result=CommandResult.HANDLED)

# =============================================================================
# HELP
# =============================================================================

@command("buffer", aliases=["buf", "stashed", "outputs"])
def cmd_buffer(state: Session, args: List[str]) -> CommandResponse:
    """Inspect stashed large tool outputs. Usage: /buffer [list|show <ref>|search <ref> <query>|drop <ref>]"""
    buf = get_output_buffer()

    if not args or args[0] == "list":
        refs = buf.list_refs()
        if not refs:
            print_system("No large outputs currently stashed.")
            return CommandResponse(result=CommandResult.HANDLED)

        print()
        print_system("Stashed Large Tool Outputs")
        print_system("─" * 50)
        for meta in refs:
            print(f"  {COLORS.TOOL_NAME}{meta['ref']}{COLORS.RESET}  "
                  f"{meta['server']}/{meta['tool']}  "
                  f"{meta['content_type']}  "
                  f"{meta['total_lines']:,} lines  "
                  f"{meta['total_chars']:,} chars")
        print()
        print_system(f"Total: {len(refs)} stashed output(s). Use /buffer show <ref> for preview.")
        print()
        return CommandResponse(result=CommandResult.HANDLED)

    action = args[0].lower()

    if action == "show" and len(args) >= 2:
        ref = args[1]
        meta = buf.metadata(ref)
        if not meta:
            print_error(f"Unknown reference: {ref}")
            return CommandResponse(result=CommandResult.HANDLED)

        chunk, has_more, next_off = buf.read(ref, 0, 4000)
        print()
        print_system(f"Output {ref} — {meta['server']}/{meta['tool']} ({meta['content_type']})")
        print_system("─" * 50)
        print(chunk)
        if has_more:
            print_system(
                f"... {meta['total_chars'] - len(chunk):,} more chars. "
                f"Use /buffer show {ref} or read_output(ref='{ref}', offset={next_off})"
            )
        print()
        return CommandResponse(result=CommandResult.HANDLED)

    if action == "search" and len(args) >= 3:
        ref = args[1]
        query = " ".join(args[2:])
        matches = buf.search(ref, query, context_lines=2)
        if not matches:
            print_system(f"No matches for '{query}' in {ref}")
            return CommandResponse(result=CommandResult.HANDLED)

        print()
        print_system(f"Search results for '{query}' in {ref}")
        print_system("─" * 50)
        for line_no, line in matches[:40]:  # cap display
            if line_no == 0 and line == "---":
                print("  ---")
            else:
                print(f"  {line}")
        if len(matches) > 40:
            print_system(f"  ... ({len(matches) - 40} more matches)")
        print()
        return CommandResponse(result=CommandResult.HANDLED)

    if action == "drop" and len(args) >= 2:
        ref = args[1]
        if buf.drop(ref):
            print_success(f"✓ Dropped {ref}")
        else:
            print_error(f"Unknown reference: {ref}")
        return CommandResponse(result=CommandResult.HANDLED)

    print_warning("Usage: /buffer list | /buffer show <ref> | /buffer search <ref> <query> | /buffer drop <ref>")
    return CommandResponse(result=CommandResult.HANDLED)


@command("help", aliases=["h", "?", "commands"])
def cmd_help(state: Session, args: List[str]) -> CommandResponse:
    """Show available commands."""
    print_commands_help()
    return CommandResponse(result=CommandResult.HANDLED)

# =============================================================================
# TRUST TIER (for testing)
# =============================================================================

@command("tier", aliases=["trust"])
def cmd_tier(state: Session, args: List[str]) -> CommandResponse:
    """View or change trust tier. Usage: /tier [0-3]"""
    # FIX #5: Late import moved to top of file.
    if not args:
        # Show current
        print()
        print_system(f"Current trust tier: {state.trust_tier}")
        print()
        print_system("Available tiers:")
        for level, tier in TIERS.items():
            current = " ← current" if tier.name == state.trust_tier else ""
            print(f"  [{level}] {tier.name}{current}")
        print()
        print_system("Usage: /tier <level> to change")
        return CommandResponse(result=CommandResult.HANDLED)
    
    try:
        level = int(args[0])
        if level not in TIERS:
            print_warning(f"Invalid tier: {level}. Must be 0-3.")
            return CommandResponse(result=CommandResult.HANDLED)
        
        tier = TIERS[level]

        # FIX #4: Require explicit confirmation for tier elevation to 2+.
        # Parse current tier level.
        current_level = None
        for lvl, t in TIERS.items():
            if t.name == state.trust_tier:
                current_level = lvl
                break

        if current_level is not None and level > current_level and level >= 2:
            # Build permission list for confirmation prompt.
            perms = []
            if tier.can_write_shadow:
                perms.append("shadow writes")
            if tier.can_write_direct:
                perms.append("direct writes")
            if tier.can_ssh:
                perms.append("SSH")
            if tier.can_sudo:
                perms.append("sudo")
            if tier.can_network:
                perms.append("network")

            perm_str = ", ".join(perms) if perms else "no special permissions"
            response = prompt_approval(
                f"Elevate trust tier to {tier.name} (level {level})",
                f"This grants: {perm_str}",
                ["y", "n"]
            )
            if response != "y":
                print_system("Tier elevation cancelled.")
                return CommandResponse(result=CommandResult.HANDLED)

        state.trust_tier = tier.name
        print_success(f"✓ Trust tier set to: {tier.name} (level {level})")
        
        # Show what this enables
        perms = []
        if tier.can_write_shadow:
            perms.append("shadow writes")
        if tier.can_write_direct:
            perms.append("direct writes")
        if tier.can_ssh:
            perms.append("SSH")
        if tier.can_sudo:
            perms.append("sudo")
        if tier.can_network:
            perms.append("network")
        
        if perms:
            print_system(f"  Enabled: {', '.join(perms)}")
        
    except ValueError:
        print_warning("Usage: /tier <level> where level is 0-3")
    
    return CommandResponse(result=CommandResult.HANDLED)
