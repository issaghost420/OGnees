#!/usr/bin/env python3
"""
commands.py - Slash command parser and handlers

User control over the OGneey. Every command is registered with a section,
usage line and aliases; /help is generated from this registry so it can't
drift. Commands run on the main thread and most of them work while the agent
is busy (see BUSY_REFUSED for the few that don't).
"""

import difflib
import re
import sys
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List, Tuple

import requests

from .context import (
    Session,
    SessionState,
    transition_state,
    resume_target,
    save_continuation_scroll,
    post_inbox,
    tier_level_of,
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
from .display import print_dim
from .inputhub import get_hub
from . import shell_policy

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
_canonical: Dict[str, str] = {}          # name or alias -> canonical name
_meta: Dict[str, Dict[str, Any]] = {}    # canonical name -> help metadata

# Commands that would race a running agent. Everything else works mid-turn.
BUSY_REFUSED = {"execute", "apply", "discard-shadow", "clear", "continue", "regenerate"}

_SECTION_ORDER = ["Control", "Tools & permissions", "Session", "Plan mode",
                  "Shadow files", "Remote", "Voice", "Other"]


def command(name: str, aliases: List[str] = None, section: str = "Other",
            usage: str = None, strict: bool = False):
    """Decorator to register a command.

    strict=True: the command takes no arguments, and refuses to run if it
    gets any (so a typo like `/apply foo` can't apply everything).
    """
    def decorator(func):
        for key in [name] + list(aliases or []):
            owner = _canonical.get(key)
            if owner is not None and owner != name:
                raise ValueError(f"slash-command alias collision: /{key} ({owner} vs {name})")
            _commands[key] = func
            _canonical[key] = name
        doc = (func.__doc__ or "").strip()
        flat = " ".join(line.strip() for line in doc.splitlines())
        m = re.match(r"(.+?[.!?])(\s|$)", flat)
        summary = (m.group(1) if m else flat)
        summary = re.sub(r"\s*Usage:.*$", "", summary).strip()
        _meta[name] = {
            "section": section,
            "usage": usage or f"/{name}",
            "aliases": list(aliases or []),
            "strict": strict,
            "summary": summary,
            "doc": doc,
        }
        return func
    return decorator


def canonical_command(text: str) -> str:
    """'/p' -> 'pause', '/pf' -> 'pause-force'; '' if unknown."""
    cmd, _ = parse_command(text)
    return _canonical.get(cmd, "")


def command_names() -> List[str]:
    """Every '/name' and '/alias' (for Tab completion)."""
    return ["/" + k for k in _commands]


def command_help_sections() -> List[Any]:
    by_section: Dict[str, List[Any]] = {}
    for name, meta in _meta.items():
        by_section.setdefault(meta["section"], []).append(
            (meta["usage"], meta["aliases"], meta["summary"], name not in BUSY_REFUSED))
    ordered = [s for s in _SECTION_ORDER if s in by_section]
    ordered += [s for s in by_section if s not in ordered]
    return [(sec, by_section[sec]) for sec in ordered]


def _print_command_help(name: str):
    meta = _meta.get(name, {})
    print()
    print_system(f"{meta.get('usage', '/' + name)}")
    if meta.get("aliases"):
        print_dim("  aliases: " + ", ".join("/" + a for a in meta["aliases"]))
    for line in (meta.get("doc") or "").splitlines():
        print(f"  {line.strip()}")
    print()


def _tool_catalog(state: Session) -> Dict[str, List[Dict[str, str]]]:
    provider = getattr(state, "tool_catalog_provider", None)
    if provider:
        try:
            return provider()
        except Exception:
            pass
    avail = getattr(state, "available_tools", {}) or {}
    if isinstance(avail, dict):
        return {srv: [{"name": t, "description": ""} for t in tools] for srv, tools in avail.items()}
    return {}


def _tool_keys(state: Session) -> List[str]:
    cat = _tool_catalog(state)
    keys = list(cat.keys())
    for srv, tools in cat.items():
        keys.extend(f"{srv}/{t['name']}" for t in tools)
    return keys


def resolve_tool_target(state: Session, target: str) -> Tuple[Optional[str], List[str]]:
    """Map what the operator typed to a real server or server/tool key.

    Accepts: server, server/tool, server__tool, or a bare tool name if it's
    unique. Case-insensitive. Returns (key, suggestions).
    """
    cat = _tool_catalog(state)
    t = target.strip().strip("/")
    if not t:
        return None, []
    servers = {s.lower(): s for s in cat}
    if t.lower() in servers:
        return servers[t.lower()], []
    keys = {f"{srv}/{tool['name']}".lower(): f"{srv}/{tool['name']}"
            for srv, tools in cat.items() for tool in tools}
    if t.lower() in keys:
        return keys[t.lower()], []
    if "__" in t:
        srv, _, tool = t.partition("__")
        cand = f"{srv}/{tool}".lower()
        if cand in keys:
            return keys[cand], []
    bare = [k for k in keys.values() if k.split("/", 1)[1].lower() == t.lower()]
    if len(bare) == 1:
        return bare[0], []
    pool = list(cat.keys()) + list(keys.values())
    if bare:
        return None, bare
    close = difflib.get_close_matches(t, pool, n=4, cutoff=0.55)
    if not close:
        lowered = {k.lower(): k for k in pool}
        close = [lowered[k] for k in difflib.get_close_matches(t.lower(), list(lowered), n=4, cutoff=0.55)]
    return None, close


def command_arg_choices(cmd: str, state: Session) -> List[str]:
    """Tab-completion candidates for a command's first argument."""
    name = _canonical.get(cmd, cmd)
    if name == "tier":
        return ["0", "1", "2", "3"]
    if name == "think":
        return ["short", "full", "off", "last", "toggle"]
    if name == "tools":
        return ["list", "on", "off", "disable", "enable", "status"]
    if name in ("disable", "enable", "tools-list"):
        extra = ["all"] if name == "enable" else []
        return _tool_keys(state) + extra
    if name == "regenerate":
        return [str(k) for k in sorted(getattr(state, "response_marks", {}) or {})]
    if name == "exceptions":
        return ["list", "clear"]
    if name == "policy":
        return ["show", "reload", "test", "path"]
    if name == "buffer":
        return ["list", "show", "search", "drop"]
    if name == "prompt":
        return ["view", "append", "clear", "show-full"]
    if name in ("ssh", "hosts"):
        return list(HOSTS)
    return []

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
        canon = _canonical.get(cmd, cmd)
        meta = _meta.get(canon, {})
        if args and args[0].lower() in ("--help", "-h", "help") and canon != "help":
            _print_command_help(canon)
            return CommandResponse(result=CommandResult.HANDLED)
        if meta.get("strict") and args:
            print_warning(f"/{canon} doesn't take arguments — usage: {meta.get('usage')}  "
                          f"(nothing was done)")
            return CommandResponse(result=CommandResult.HANDLED)
        try:
            return _commands[cmd](state, args)
        except Exception as e:
            print_error(f"Command error: {e}")
            return CommandResponse(result=CommandResult.CONTINUE)
    else:
        print_warning(f"Unknown command: /{cmd}")
        close = difflib.get_close_matches(cmd, list(_commands), n=3, cutoff=0.6)
        if close:
            print_system("Did you mean: " + ", ".join("/" + c for c in close) + " ?")
        print_system("Type /help for available commands")
        return CommandResponse(result=CommandResult.HANDLED)

# =============================================================================
# CONTROL COMMANDS
# =============================================================================

@command("pause", aliases=["p", "stop", "halt", "wait", "hold", "brb"], section="Control", strict=True)
def cmd_pause(state: Session, args: List[str]) -> CommandResponse:
    """Pause at the next turn boundary (gentle). The current LLM reply or tool
    batch finishes and its results are kept; then the agent stops. While
    paused, typed messages are queued and delivered on /resume."""
    if state.mode is SessionState.PAUSED:
        print_system("Already paused. Type /resume to continue.")
        return CommandResponse(result=CommandResult.HANDLED, data={"pause": True})
    transition_state(state, SessionState.PAUSED)
    if getattr(state, "agent_busy", False):
        print_success("⏸  Pausing at the next turn boundary (the current step finishes first). "
                      "/pf stops immediately · /r cancels the pause.")
    else:
        print_success("⏸  Paused. Type /resume to continue.")
    return CommandResponse(result=CommandResult.HANDLED, data={"pause": True})

@command("pause-force", aliases=["pf", "pause-now", "pausenow", "pauseforce", "force-pause",
                                  "forcepause", "fp", "stop-now", "stopnow", "abort", "halt-now"],
         section="Control", strict=True)
def cmd_pause_force(state: Session, args: List[str]) -> CommandResponse:
    """Stop RIGHT NOW. Cancels the LLM request mid-generation, cancels running
    MCP tool calls, and answers any open approval prompt with "no". The
    session stays open and paused; /r continues. (Ctrl-C while OGneey is
    working does the same thing.)"""
    print_warning("⏹  Force pause — stopping the current turn now")
    return CommandResponse(result=CommandResult.HANDLED, data={"force_pause": True})

@command("resume", aliases=["r", "go", "unpause", "res", "proceed", "carry-on"], section="Control", strict=True)
def cmd_resume(state: Session, args: List[str]) -> CommandResponse:
    """Resume after a pause. Continues the paused task, delivering anything
    you typed while paused. (Typed before the pause took effect: cancels it.)"""
    if state.mode is not SessionState.PAUSED:
        print_system("Not paused.")
        return CommandResponse(result=CommandResult.HANDLED)

    transition_state(state, resume_target(state))
    print_success("▶  Resumed.")
    return CommandResponse(
        result=CommandResult.HANDLED,
        data={"resume": True},
    )

@command("quit", aliases=["q", "exit", "bye", "leave", "q!"], section="Control")
def cmd_quit(state: Session, args: List[str]) -> CommandResponse:
    """End session gracefully."""
    print_system("Ending session...")
    return CommandResponse(result=CommandResult.QUIT)

@command("continue", aliases=["load-continuation", "cont"], section="Session")
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

@command("regenerate", aliases=["regen", "rg", "retry", "redo", "reroll"], section="Session",
         usage="/regenerate [N]")
def cmd_regenerate(state: Session, args: List[str]) -> CommandResponse:
    """Redo an OGneey reply. With no number it redoes the latest reply;
    /regenerate 3 redoes reply #3 (the number under each reply).
    Redoing an earlier reply rewinds the conversation to just before it, so
    the replies and messages after it leave OGneey's context — you're asked
    first, and everything dropped stays in the session log. Tool calls that
    already ran are not undone."""
    if len(args) > 1 or (args and not args[0].lstrip("#").isdigit()):
        print_warning("Usage: /regenerate [N]  — N is the #number under an OGneey reply")
        return CommandResponse(result=CommandResult.HANDLED)
    target = int(args[0].lstrip("#")) if args else None
    return CommandResponse(result=CommandResult.HANDLED, data={"regenerate": target})


@command("speak-on", aliases=["speakon", "voice-on", "tts-on"], section="Voice", strict=True)
def cmd_speak_on(state: Session, args: List[str]) -> CommandResponse:
    """Enable the speak tool."""
    # FIX #5: Late import moved to top of file.
    toggle_speak(True)
    print_success("🔊 Speak enabled. OGneey can now use local/speak tool.")
    return CommandResponse(result=CommandResult.HANDLED)

@command("speak-off", aliases=["speakoff", "voice-off", "tts-off", "stfu", "shutup"], section="Voice", strict=True)
def cmd_speak_off(state: Session, args: List[str]) -> CommandResponse:
    """Disable the speak tool."""
    # FIX #5: Late import moved to top of file.
    toggle_speak(False)
    print_success("🔇 Speak disabled. OGneey will not use TTS.")
    return CommandResponse(result=CommandResult.HANDLED)

@command("say", aliases=["speak"], section="Voice", usage="/say <text>")
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

@command("tools", section="Tools & permissions",
         usage="/tools [list [filter]|on|off|disable <x>|enable <x>|status]")
def cmd_tools(state: Session, args: List[str]) -> CommandResponse:
    """Tool control in one place. With no arguments it only shows status —
    it never changes anything unless you name a subcommand.
      /tools                  status
      /tools list [filter]    same as /tools-list
      /tools on | off         same as /tools-on | /tools-off
      /tools disable <x>...   same as /disable
      /tools enable <x>...    same as /enable"""
    if not args or args[0].lower() in ("status", "st"):
        cat = _tool_catalog(state)
        n_tools = sum(len(v) for v in cat.values())
        disabled = sorted(getattr(state, "disabled_tools", set()) or [])
        status = "🔓 enabled" if state.tools_enabled else "🔒 OFF (chat-only)"
        print()
        print_system(f"Tools: {status} · {len(cat)} servers · {n_tools} tools · "
                     f"{len(disabled)} disabled")
        if disabled:
            print_system("  disabled: " + ", ".join(disabled))
        print_dim("  /tools list · /tools off · /disable <server|server/tool> · /enable all")
        print()
        return CommandResponse(result=CommandResult.HANDLED)
    sub, rest = args[0].lower(), args[1:]
    if sub in ("list", "ls", "l"):
        return cmd_list_tools(state, rest)
    if sub in ("on", "enable-all"):
        return cmd_tools_on(state, [])
    if sub == "off":
        return cmd_tools_off(state, [])
    if sub in ("disable", "dis", "block"):
        return cmd_disable_tool(state, rest)
    if sub in ("enable", "en", "unblock"):
        return cmd_enable_tool(state, rest)
    print_warning(f"Unknown /tools subcommand '{args[0]}' — nothing was changed.")
    _print_command_help("tools")
    return CommandResponse(result=CommandResult.HANDLED)

@command("tools-off", aliases=["toolsoff", "notool", "notools", "chat", "chat-only"], section="Tools & permissions", strict=True)
def cmd_tools_off(state: Session, args: List[str]) -> CommandResponse:
    """Disable all tool calls - chat only mode."""
    state.tools_enabled = False
    print_success("🔒 Tools disabled. OGneey is now chat-only.")
    return CommandResponse(result=CommandResult.HANDLED)

@command("tools-on", aliases=["toolson", "tools-enable"], section="Tools & permissions", strict=True)
def cmd_tools_on(state: Session, args: List[str]) -> CommandResponse:
    """Enable tool calls."""
    state.tools_enabled = True
    print_success("🔓 Tools enabled.")
    return CommandResponse(result=CommandResult.HANDLED)

@command("disable", aliases=["dis", "disable-tool", "block-tool"], section="Tools & permissions", usage="/disable <server|server/tool> ...")
def cmd_disable_tool(state: Session, args: List[str]) -> CommandResponse:
    """Disable a server or a single tool for this session. The model stops
    seeing it at all (it's removed from the tool list sent to the model).
    Names are checked against the live tool list, with suggestions for typos.
    Examples: /disable browser   /disable git deep-research   /disable local-files/write_file"""
    if not args:
        print_warning("Usage: /disable <server|server/tool> [more...]   (/tools-list shows names)")
        return CommandResponse(result=CommandResult.HANDLED)
    cat = _tool_catalog(state)
    for target in args:
        key, suggestions = resolve_tool_target(state, target)
        if key is None:
            if suggestions:
                print_warning(f"  ✗ no tool or server called '{target}'. Did you mean: "
                              + ", ".join(suggestions) + " ?")
            else:
                print_warning(f"  ✗ no tool or server called '{target}' (see /tools-list)")
            continue
        state.disabled_tools.add(key)
        if "/" in key:
            print_success(f"🚫 Disabled tool {key}")
        else:
            n = len(cat.get(key, []))
            print_success(f"🚫 Disabled server {key} ({n} tool{'s' if n != 1 else ''})")
    return CommandResponse(result=CommandResult.HANDLED)

@command("enable", aliases=["en", "enable-tool", "unblock-tool"], section="Tools & permissions", usage="/enable <server|server/tool|all> ...")
def cmd_enable_tool(state: Session, args: List[str]) -> CommandResponse:
    """Re-enable a server or tool you disabled. /enable all clears the list."""
    if not args:
        print_warning("Usage: /enable <server|server/tool|all>")
        return CommandResponse(result=CommandResult.HANDLED)
    if args[0].lower() == "all":
        n = len(state.disabled_tools)
        state.disabled_tools.clear()
        print_success(f"✓ Re-enabled everything ({n} entr{'ies' if n != 1 else 'y'})")
        return CommandResponse(result=CommandResult.HANDLED)
    for target in args:
        key, suggestions = resolve_tool_target(state, target)
        key = key or target
        match = next((d for d in state.disabled_tools if d.lower() == key.lower()), None)
        if match:
            state.disabled_tools.discard(match)
            print_success(f"✓ Enabled: {match}")
        else:
            print_system(f"{target} was not disabled"
                         + (f" (disabled: {', '.join(sorted(state.disabled_tools))})"
                            if state.disabled_tools else ""))
    return CommandResponse(result=CommandResult.HANDLED)

@command("disabled", aliases=["dis-list", "disabled-list", "ls-disabled"], section="Tools & permissions", strict=True)
def cmd_list_disabled(state: Session, args: List[str]) -> CommandResponse:
    """List disabled tools."""
    if not hasattr(state, 'disabled_tools') or not state.disabled_tools:
        print_system("No tools are disabled.")
    else:
        cat = _tool_catalog(state)
        print_system("Disabled (hidden from the model and blocked at the gate):")
        for t in sorted(state.disabled_tools):
            extra = f"  ({len(cat.get(t, []))} tools)" if "/" not in t and t in cat else ""
            print_system(f"  🚫 {t}{extra}")
        print_dim("  /enable <name> or /enable all to restore")
    return CommandResponse(result=CommandResult.HANDLED)

@command("exceptions", aliases=["exc", "allowed", "waivers", "allowlist", "approvals"],
         section="Tools & permissions", usage="/exceptions [clear [task|session|all]]")
def cmd_exceptions(state: Session, args: List[str]) -> CommandResponse:
    """Show what you've waved through from approval prompts with [6] (until
    the task ends) and [7] (this session). /exceptions clear removes them."""
    ex = state.gate_exceptions
    labels = state.gate_exception_labels
    if args and args[0].lower() in ("clear", "reset", "rm", "drop"):
        scope = args[1].lower() if len(args) > 1 else "all"
        scopes = ["task", "session"] if scope == "all" else [scope]
        n = 0
        for sc in scopes:
            if sc in ex:
                n += len(ex[sc])
                ex[sc].clear()
        print_success(f"✓ Cleared {n} exception(s) ({scope})")
        return CommandResponse(result=CommandResult.HANDLED)
    print()
    any_shown = False
    for scope, title in (("task", "until this task ends  [6]"), ("session", "this session  [7]")):
        keys = sorted(ex.get(scope, set()))
        if not keys:
            continue
        any_shown = True
        print_system(f"Allowed {title}:")
        for k in keys:
            print_system(f"  ✓ {labels.get(k, k)}   {COLORS.DIM}{k}{COLORS.RESET}")
    if not any_shown:
        print_system("No approval exceptions yet. Pick [6] or [7] at an approval prompt to add one.")
    print()
    return CommandResponse(result=CommandResult.HANDLED)


@command("policy", aliases=["pol", "shell-policy", "perms", "permissions"],
         section="Tools & permissions", usage="/policy [show|reload|test <command>|path]")
def cmd_policy(state: Session, args: List[str]) -> CommandResponse:
    """The shell allow/confirm/deny policy (Zed-style).
      /policy               show the lists and settings
      /policy test <cmd>    dry-run: what would happen to this command right now
      /policy reload        re-read the file (it also auto-reloads on change)
      /policy path          which file is in use"""
    sub = args[0].lower() if args else "show"
    if sub == "reload":
        pol = shell_policy.reload_policy()
        if pol.errors:
            print_error("✗ policy has errors — ALL shell commands are blocked until fixed:")
            for e in pol.errors:
                print_error(f"   {e}")
        else:
            print_success(f"✓ reloaded {pol.source}: {len(pol.deny)} deny · {len(pol.confirm)} confirm · "
                          f"{len(pol.allow)} allow")
        return CommandResponse(result=CommandResult.HANDLED)
    pol = shell_policy.get_policy()
    if sub == "path":
        print_system(pol.source if pol.source != "builtin" else
                     "built-in defaults (no shell-policy.json found; set OGNEE_SHELL_POLICY or "
                     "put one next to the backend)")
        return CommandResponse(result=CommandResult.HANDLED)
    if sub in ("test", "check", "try", "dry-run"):
        cmd = " ".join(args[1:]).strip()
        if not cmd:
            print_warning("Usage: /policy test <command>")
            return CommandResponse(result=CommandResult.HANDLED)
        level = tier_level_of(state)
        d = shell_policy.evaluate(cmd, level, pol)
        color = {"allow": COLORS.SUCCESS, "confirm": COLORS.WARNING, "deny": COLORS.ERROR}[d.result]
        print()
        print(f"  {color}{d.result.upper()}{COLORS.RESET} at tier {level} — {d.reason}")
        if not d.parse_ok:
            print_dim("  (couldn't parse the command; treated as one unit, allow rules off)")
        for seg in d.segments:
            print_dim(f"    · {seg.collapsed[:100]}" + (f"   [{seg.origin}]" if seg.origin != "top" else ""))
        if d.exception_keys:
            print_dim("  [6]/[7] at the prompt would cover: " + ", ".join(lbl for _, lbl in d.exception_keys))
        print()
        return CommandResponse(result=CommandResult.HANDLED)
    print()
    print_system(f"Shell policy: {pol.source if pol.source != 'builtin' else 'built-in defaults'}")
    print_system(f"  match: {pol.match}   default: {pol.default}   (tier {tier_level_of(state)} now)")
    for line in pol.summary_lines():
        print_system("  " + line)
    if pol.errors:
        print_error("  ✗ errors (every shell command is blocked until fixed):")
        for e in pol.errors:
            print_error(f"     {e}")
    print_dim("  Order (same as Zed): deny if ANY sub-command matches deny → ask if ANY matches "
              "confirm → allow if EVERY sub-command matches allow → otherwise the default.")
    print_dim("  /policy test <command> to dry-run one")
    print()
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

@command("prompt", aliases=["sys", "sysprompt", "system-prompt"], section="Session", usage="/prompt [view|append <text>|clear|show-full]")
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

@command("plan", aliases=["propose", "planmode"], section="Plan mode", strict=True)
def cmd_plan_mode(state: Session, args: List[str]) -> CommandResponse:
    """Enter plan mode - OGneey proposes actions but doesn't execute."""
    transition_state(state, SessionState.PLAN_REVIEW)
    state.pending_plan = []
    print_success("📋 Plan mode ON. OGneey will propose actions without executing.")
    print_system("   Use /execute to run the plan, /clear to discard.")
    return CommandResponse(result=CommandResult.HANDLED)

@command("execute", aliases=["exec", "run", "do"], section="Plan mode", strict=True)
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

@command("clear", aliases=["discard", "plan-clear"], section="Plan mode", strict=True)
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

@command("plan-off", aliases=["planoff", "noplan", "unplan"], section="Plan mode", strict=True)
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

@command("step", aliases=["s", "next"], section="Control", strict=True)
def cmd_step(state: Session, args: List[str]) -> CommandResponse:
    """Execute one tool call, then pause. HANDLED — never sent to the LLM."""
    state.step_mode = True
    print_success("👟 Step mode: will pause after next tool call.")
    return CommandResponse(result=CommandResult.HANDLED)

# =============================================================================
# STATUS & INFO
# =============================================================================

@command("status", aliases=["stat", "st", "info"], section="Session", strict=True)
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

@command("tokens", aliases=["tok", "context", "ctx", "usage"], section="Session", strict=True)
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

@command("think", aliases=["thinking", "reasoning", "th", "thoughts"], section="Session",
         usage="/think [short|full|off|last|toggle]")
def cmd_think(state: Session, args: List[str]) -> CommandResponse:
    """How much of the model's reasoning to show.
      /think short   first 300 chars per turn (default)
      /think full    everything, every turn
      /think off     hide it
      /think last    print the last turn's reasoning in full (expand)
      /think toggle  flip between short and full
    Reasoning is always saved to the session log, whatever you show."""
    if not args:
        n = len(state.last_reasoning or "")
        print_system(f"Thinking display: {state.think_mode}" + (f" · last turn: {n:,} chars" if n else ""))
        print_dim("  /think short|full|off|last|toggle")
        return CommandResponse(result=CommandResult.HANDLED)
    sub = args[0].lower()
    if sub in ("last", "expand", "show", "more"):
        text = state.last_reasoning or ""
        if not text:
            print_system("No reasoning captured yet.")
        else:
            print()
            print_dim("── last reasoning " + "─" * 50)
            print(f"{COLORS.DIM}{text}{COLORS.RESET}")
            print_dim("─" * 68)
            print()
        return CommandResponse(result=CommandResult.HANDLED)
    if sub == "toggle":
        state.think_mode = "full" if state.think_mode != "full" else "short"
    elif sub in ("short", "full", "off"):
        state.think_mode = sub
    elif sub in ("on", "expand-all", "all"):
        state.think_mode = "full"
    elif sub in ("collapse", "brief"):
        state.think_mode = "short"
    else:
        print_warning("Usage: /think short|full|off|last|toggle")
        return CommandResponse(result=CommandResult.HANDLED)
    print_success(f"💭 Thinking display: {state.think_mode}")
    return CommandResponse(result=CommandResult.HANDLED)

@command("tools-list", aliases=["tl", "list-tools", "lt", "ls-tools", "tools-ls", "toollist", "listtools"], section="Tools & permissions", usage="/tools-list [filter]")
def cmd_list_tools(state: Session, args: List[str]) -> CommandResponse:
    """List every tool the model can call, grouped by server, with disabled
    ones marked. Optional filter: /tools-list read  ·  /tools-list git"""
    cat = _tool_catalog(state)
    if not cat:
        print_system("No tools are registered (did the MCP servers start?).")
        return CommandResponse(result=CommandResult.HANDLED)
    flt = " ".join(args).lower().strip()
    disabled = getattr(state, "disabled_tools", set()) or set()
    total = shown = 0
    print()
    print_system("Available tools" + (f" matching '{flt}'" if flt else ""))
    print_system("─" * 60)
    for server in sorted(cat, key=lambda x: (x != "local", x)):
        tools = cat[server]
        total += len(tools)
        rows = [t for t in tools if not flt or flt in server.lower() or flt in t["name"].lower()
                or flt in (t.get("description") or "").lower()]
        if not rows:
            continue
        srv_off = server in disabled
        tag = f" {COLORS.WARNING}(disabled){COLORS.RESET}" if srv_off else ""
        print(f"\n  {COLORS.TOOL_NAME}{server}{COLORS.RESET}  {COLORS.DIM}{len(tools)} tools{COLORS.RESET}{tag}")
        for t in rows:
            shown += 1
            key = f"{server}/{t['name']}"
            desc = (t.get("description") or "").strip().replace("\n", " ")
            desc = desc[:70] + ("…" if len(desc) > 70 else "")
            if srv_off or key in disabled:
                print(f"    {COLORS.DIM}🚫 {t['name']:28} {desc}{COLORS.RESET}")
            else:
                print(f"    • {t['name']:28} {COLORS.DIM}{desc}{COLORS.RESET}")
    print()
    print_dim(f"  {shown} shown · {total} total · {len(disabled)} disabled · /disable <server|server/tool>")
    print()
    return CommandResponse(result=CommandResult.HANDLED)

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

@command("save", aliases=["checkpoint", "ckpt"], section="Session", strict=True)
def cmd_save(state: Session, args: List[str]) -> CommandResponse:
    """Force save session log and state."""
    print_system("Saving session...")
    
    # Save continuation scroll
    scroll_path = save_continuation_scroll(state, "Manual save checkpoint")
    print_success(f"✓ Saved continuation: {scroll_path.name}")
    
    return CommandResponse(result=CommandResult.HANDLED)

@command("recap", aliases=["summary", "sum", "tldr"], section="Session", strict=True)
def cmd_recap(state: Session, args: List[str]) -> CommandResponse:
    """Request a recap of the current session."""
    print_system("Requesting session recap from OGneey...")
    
    # Return a signal to inject a recap request into the conversation
    return CommandResponse(
        result=CommandResult.CONTINUE,
        data={"inject_user_message": "Please provide a brief recap of what we've done this session so far."}
    )

@command("diff", aliases=["changes", "pending", "shadow-diff"], section="Shadow files", strict=True)
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

@command("apply", aliases=["commit", "shadow-apply"], section="Shadow files", strict=True)
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

@command("discard-shadow", aliases=["reset-shadow", "shadow-discard"], section="Shadow files", strict=True)
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

@command("ssh", aliases=["remote"], section="Remote", usage="/ssh <host> <command>")
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

@command("hosts", aliases=["nodes", "machines"], section="Remote", strict=True)
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

@command("buffer", aliases=["buf", "stashed", "outputs", "stash"], section="Session", usage="/buffer [list|show <ref>|search <ref> <q>|drop <ref>]")
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


@command("help", aliases=["h", "?", "commands", "cmds", "halp", "man"], section="Other", usage="/help [command]")
def cmd_help(state: Session, args: List[str]) -> CommandResponse:
    """Show available commands, or details for one: /help pause"""
    if args:
        name = _canonical.get(args[0].lstrip("/").lower())
        if name:
            _print_command_help(name)
        else:
            print_warning(f"Unknown command: /{args[0].lstrip('/')}")
        return CommandResponse(result=CommandResult.HANDLED)
    print_commands_help(command_help_sections())
    return CommandResponse(result=CommandResult.HANDLED)

# =============================================================================
# TRUST TIER (for testing)
# =============================================================================

@command("tier", aliases=["trust", "t", "level"], section="Tools & permissions", usage="/tier [0-3]")
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

        old_name = state.trust_tier
        state.trust_tier = tier.name
        print_success(f"✓ Trust tier set to: {tier.name} (level {level})")
        if old_name != tier.name:
            # Tell the model (delivered at the next turn boundary) and force a
            # fresh state block so the old "Trust Level" in the prompt is
            # overridden.
            caps = []
            if tier.can_write_direct:
                caps.append("write directly to allowed paths")
            elif tier.can_write_shadow:
                caps.append("writes are staged to shadow")
            if tier.can_ssh:
                caps.append("SSH")
            if tier.can_network:
                caps.append("network")
            if tier.can_sudo:
                caps.append("sudo (still subject to the shell policy)")
            post_inbox(state, (f"[SYSTEM NOTICE: the operator changed your trust tier from {old_name} "
                               f"to {tier.name} (level {level}). You can now: "
                               f"{', '.join(caps) or 'read only'}. This replaces the Trust Level in "
                               f"your instructions. Shell commands still follow the shell policy.]"))
            state.last_state_injection = -10**6
        
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


@command("input", aliases=["keyboard", "kb", "tty"], section="Other", strict=True)
def cmd_input(state: Session, args: List[str]) -> CommandResponse:
    """Show which line editor is active (and how to change it).
    If arrow keys print ^[[D instead of moving the cursor, this tells you why."""
    hub = get_hub()
    print()
    print_system(f"Input: {hub.describe()}")
    if hub.note:
        print_warning(f"  {hub.note}")
    print_dim(f"  history: {hub.history_path or '(none)'}")
    print_dim("  Switch with OGNEE_INPUT=readline (default) | ptk (prompt_toolkit, if installed) | basic")
    print_dim("  Keys: ←/→ move · Home/End or Ctrl-A/E · Ctrl-W/U delete · ↑/↓ history · Tab completes /commands")
    print()
    return CommandResponse(result=CommandResult.HANDLED)


@command("cls", aliases=["clear-screen", "clr"], section="Other", strict=True)
def cmd_cls(state: Session, args: List[str]) -> CommandResponse:
    """Clear the screen. (/clear is the plan-mode "discard plan" command.)"""
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()
    return CommandResponse(result=CommandResult.HANDLED)
