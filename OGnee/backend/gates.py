#!/usr/bin/env python3
"""
gates.py - Approval prompts and permission checks

The safety layer. Shell commands are decided by the shell policy file
(shell-policy.json, Zed-style allow/confirm/deny — see shell_policy.py);
file, move, network and SSH calls by the trust tier. Anything that needs a
human goes through the 1-7 approval menu, and whatever the operator types
there (notes, reasons, messages) reaches the model.
"""

import copy
import json
import os
import re
import shutil
import unicodedata  # FIX 2: Unicode normalization for interpreter bypass detection
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from .config import (
    get_trust_tier,
    is_path_read_only,
    is_path_writable,
    is_path_excluded,
    can_access_path,
    get_shadow_path,
    SHADOW_ROOT,
    BANNED_COMMANDS,
    SUDO_PATTERNS,
    RM_PATTERNS,
    TIERS,
)
from .context import Session
from .display import (
    print_warning,
    print_error,
    print_success,
    print_system,
    print_dim,
    prompt_approval,
    prompt_choice,
    COLORS,
)
from .inputhub import get_hub
from .shell_policy import (
    evaluate as policy_evaluate,
    get_policy,
    ALLOW as P_ALLOW,
    CONFIRM as P_CONFIRM,
    DENY as P_DENY,
)


# =============================================================================
# HELPER: Build banned-command regex with appropriate boundaries
# =============================================================================

def _make_banned_pattern(text: str) -> re.Pattern:
    """
    Build a regex matching a banned command string with safe boundaries.

    Uses negative lookaround assertions for word-character edges and
    whitespace anchoring for non-word-character edges. Patterns ending
    in '/' have no end boundary so that 'rm -rf /' also matches
    'rm -rf /home', 'rm -rf /etc/passwd', etc.
    """
    escaped = re.escape(text.lower())

    # Fork bomb: special flexible-whitespace pattern (FIX 5)
    if text == ":(){:|:&};:":
        return re.compile(r':\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:')

    first_char = text[0]
    last_char = text[-1]

    # Start boundary: (?<!\w) for word chars, (?:^|\s) for non-word chars
    if first_char.isalnum() or first_char == '_':
        start = r'(?<!\w)'
    else:
        start = r'(?:^|\s)'

    # End boundary: (?!\w) for word chars. For '/' no end boundary
    # (catches trailing path components). For other non-word chars,
    # require whitespace or end-of-string.
    if last_char == '/':
        end = r''  # No restriction — rm -rf / matches rm -rf /home etc.
    elif last_char.isalnum() or last_char == '_':
        end = r'(?!\w)'
    else:
        end = r'(?:\s|$)'

    return re.compile(start + escaped + end)


# =============================================================================
# SECURITY PATTERNS (PRE-COMPILED AT MODULE LEVEL FOR PERFORMANCE)
# =============================================================================

# FIX 1: Banned command patterns with proper word-boundary matching.
# Pre-compiled to avoid recompilation on every gate check.
_BANNED_PATTERNS: List[re.Pattern] = [
    _make_banned_pattern(cmd) for cmd in BANNED_COMMANDS
]

# FIX 5: Fork bomb regex — matches the classic bash fork bomb even with
# arbitrary whitespace between characters: : ( ) { : | : & } ; :
FORK_BOMB_RE = re.compile(
    r':\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:',
    re.IGNORECASE
)

# FIX 3: RM patterns using \b (word boundary) and \s (any whitespace).
# Pre-compiled from the string patterns in config, converted to regex.
# The old code used simple string inclusion (`pattern in command_lower`)
# which missed "rm\t-rf" because the pattern was "rm " (literal space)
# but the input contained a tab character.
_RM_PATTERNS_RE: List[re.Pattern] = []
for _rm_pat in RM_PATTERNS:
    _clean = _rm_pat.strip()
    _RM_PATTERNS_RE.append(
        re.compile(r'\b' + re.escape(_clean) + r'\s', re.IGNORECASE)
    )

# FIX 4: Sudo patterns using regex word boundaries.
# Covers: sudo, sudoedit, doas, pkexec, su -c (with flexible \s),
# and su root. The old SUDO_PATTERNS list used simple strings like
# "sudo " which missed "sudoedit" entirely (it doesn't contain "sudo ").
SUDO_PATTERNS_RE = [
    re.compile(r'\bsudo\b', re.IGNORECASE),              # sudo (any usage)
    re.compile(r'\bsudoedit\b', re.IGNORECASE),           # sudoedit bypass fix
    re.compile(r'\bdoas\b', re.IGNORECASE),              # doas
    re.compile(r'\bpkexec\b', re.IGNORECASE),            # pkexec
    re.compile(r'\bsu\s+-c\b', re.IGNORECASE),           # su -c (with \s)
    re.compile(r'\bsu\s+root\b', re.IGNORECASE),         # su root
]

# FIX 2: Interpreter-based file-write patterns.
# Language interpreters (python, node, ruby, perl, sqlite3, vim, emacs, ed)
# can write files without using shell redirections or standard write commands,
# completely bypassing regex-based write detection.
_INTERPRETER_WRITE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Python file writes
    (re.compile(
        r'\bpython\d*\s+(?:-[c\w]*\s+)?.*\bopen\s*\('
        r'[^)]*[\'"]\w*[\'"]\s*[,\)]',
        re.IGNORECASE | re.DOTALL
    ), 'python open/write'),
    (re.compile(
        r'\bpython\d*\s+(?:-[c\w]*\s+)?.*\.write\s*\(',
        re.IGNORECASE | re.DOTALL
    ), 'python .write()'),
    # Node.js file writes
    (re.compile(
        r'\bnode\d*\s+(?:-[e\w]*\s+)?.*\b'
        r'(?:writeFile|writeFileSync|appendFile|appendFileSync)\s*\(',
        re.IGNORECASE | re.DOTALL
    ), 'node.js fs.writeFile'),
    (re.compile(
        r'\bnode\d*\s+(?:-[e\w]*\s+)?.*\bfs\b.*\.\s*(?:write|append)',
        re.IGNORECASE | re.DOTALL
    ), 'node.js fs.write'),
    # Ruby file writes
    (re.compile(
        r'\bruby\d*\s+(?:-[e\w]*\s+)?.*\bFile\.(?:write|open|new)\b',
        re.IGNORECASE | re.DOTALL
    ), 'ruby File.write'),
    (re.compile(
        r'\bruby\d*\s+(?:-[e\w]*\s+)?.*\bIO\.(?:write|binwrite)\b',
        re.IGNORECASE | re.DOTALL
    ), 'ruby IO.write'),
    # Perl file writes
    (re.compile(
        r'\bperl\d*\s+(?:-[e\w]*\s+)?.*\bopen\s*\(\s*[^)]*[>+]+',
        re.IGNORECASE | re.DOTALL
    ), 'perl open(>)'),
    (re.compile(
        r'\bperl\d*\s+(?:-[e\w]*\s+)?.*\bprint\s+\S+\s*>>?',
        re.IGNORECASE | re.DOTALL
    ), 'perl print >>'),
    # sqlite3 file operations
    (re.compile(
        r'\bsqlite3\s+\S+.*\.(?:output|once|save)\b',
        re.IGNORECASE | re.DOTALL
    ), 'sqlite3 .output/.save'),
    # vim batch mode file writes
    (re.compile(
        r'\bvim?\s+(?:-[eEsw]+\s+).*-[cC]\s+.*\b(?:w|wq|x|saveas)\b',
        re.IGNORECASE | re.DOTALL
    ), 'vim batch write'),
    # emacs batch mode file writes
    (re.compile(
        r'\bemacs\b.*--batch.*'
        r'(?:write-file|save-buffer|with-temp-file|'
        r'insert-file-contents|append-to-file)',
        re.IGNORECASE | re.DOTALL
    ), 'emacs batch write'),
    # emacs --script mode (can contain arbitrary elisp including writes)
    (re.compile(
        r'\bemacs\b.*--script\b',
        re.IGNORECASE | re.DOTALL
    ), 'emacs --script'),
    # ed (line editor) file writes
    (re.compile(
        r'\bed\b.*[\'"\s](?:w|W)\s+[/\w]',
        re.IGNORECASE | re.DOTALL
    ), 'ed write'),
    # General heredoc / here-string via interpreters
    (re.compile(
        r'\b(?:python|python3|node|nodejs|ruby|perl)\d*\s+.*<<',
        re.IGNORECASE | re.DOTALL
    ), 'interpreter heredoc'),
]

# =============================================================================
# UNICODE NORMALIZATION HELPER
# =============================================================================

def _normalize_command(command: str) -> str:
    """
    Apply NFKC Unicode normalization to command strings.

    FIX 2: Attackers can use Unicode homoglyphs (e.g., U+FF50 FULLWIDTH
    LATIN SMALL LETTER P instead of 'p') to bypass pattern matching.
    NFKC normalization maps compatibility characters to their canonical
    equivalents, collapsing fullwidth/halfwidth and other confusables.

    Example: 'ｐｙｔｈｏｎ３' (fullwidth) → 'python3' (ASCII)
    """
    return unicodedata.normalize('NFKC', command)


# =============================================================================
# TILDE PATH EXPANSION HELPER
# =============================================================================

def _expand_tilde_path(path: str) -> str:
    """
    Expand ~ (tilde) to the user's home directory.

    FIX 6: Path.resolve() does NOT expand the tilde (~). A path like
    '~/Documents/file.txt' will not be matched against exclusion or
    read-only lists because the tilde remains literal. This helper
    must be called before any path validation that uses resolve().

    NOTE: config.py now expands ~ canonically in all path checks; this
    helper remains as defense-in-depth for gate-local path handling.
    """
    return os.path.expanduser(path)



# =============================================================================
# GATE RESULTS
# =============================================================================

class GateResult(Enum):
    """What the gate decided."""
    ALLOW = auto()              # Proceed as requested
    DENY = auto()               # Hard no, don't even ask
    REDIRECT = auto()           # Redirect to shadow/safe alternative
    PROMPT = auto()             # Ask user for approval
    MODIFY = auto()             # Allow with modifications


@dataclass
class GateDecision:
    """Decision from a gate check."""
    result: GateResult
    reason: str
    redirect_path: Optional[str] = None     # For REDIRECT / shadow-staged PROMPT
    modified_command: Optional[str] = None  # For MODIFY
    prompt_options: Optional[Dict] = None   # legacy; the approval menu ignores it
    # --- approval menu ---
    title: str = ""                 # e.g. "shell/execute_command"
    detail: str = ""                # the thing being approved — always shown in full
    allow_label: str = "allow"
    warning: str = ""               # e.g. "runs for REAL — not staged to shadow"
    # (key, label) pairs; options 6/7 add these as task/session exceptions
    exception_keys: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class GateVerdict:
    """What actually happens to a tool call after gating (and the operator).

    ``note`` / ``reason`` / ``message`` carry the operator's words back to the
    model so nothing typed at an approval prompt is ever lost.
    """
    allowed: bool
    args: Optional[Dict[str, Any]] = None
    redirect: Optional[str] = None
    note: str = ""          # [2] allow + note  (also text after [1]/[6]/[7])
    reason: str = ""        # [4] deny + reason, or the gate's own reason
    message: str = ""       # [5] say something instead
    aborted: bool = False   # /pf, Ctrl-C or /quit at the prompt
    paused: bool = False    # /p at the prompt
    regate: bool = False    # a slash command changed settings: re-check the call
    by_operator: bool = False
    auto: str = ""          # label of the exception that auto-approved it

    def denial_text(self) -> str:
        """Text the model sees when the call did not run."""
        if self.aborted:
            return ("Not executed — the operator force-paused the session (/pf). "
                    "Stop here and wait for the operator.")
        if self.paused:
            return ("Not executed — the operator paused the session (/p). Stop at this "
                    "point and wait; you'll be resumed with further instructions.")
        if self.message:
            return ("Not executed yet — the operator answered this approval prompt with a "
                    f"message instead: \"{self.message}\". Take it into account. You may "
                    "re-issue this call (or a different one) if it still makes sense.")
        if self.by_operator:
            if self.reason:
                return f"Denied by the operator. Reason: {self.reason}"
            return "Denied by the operator."
        return (f"Blocked by the permission gate: {self.reason}. Don't retry the same call; "
                "pick another approach or ask the operator.") if self.reason else \
            "Blocked by the permission gate."


# =============================================================================
# TRUST LEVEL HELPERS
# =============================================================================

def get_tier_name(state: Session) -> str:
    """Display name based on trust level (no ANSI — callers color it)."""
    tier_name = state.trust_tier.lower()
    return {
        "probation": "OGreenie",
        "supervised": "OSeemie",
        "trusted": "OGneey",
        "autonomous": "OFreeney",
    }.get(tier_name, "OGneey")


def get_tier_level(state: Session) -> int:
    """Get numeric tier level from state."""
    tier_name = state.trust_tier.lower()
    for level, tier in TIERS.items():
        if tier.name == tier_name:
            return level
    return 0


def get_tier_config(state: Session):
    """Get TrustTier config for current state."""
    level = get_tier_level(state)
    return TIERS.get(level, TIERS[0])


# =============================================================================
# COMMAND GATES
# =============================================================================

def check_banned_command(command: str) -> GateDecision:
    """Check if command is permanently banned (catastrophic patterns)."""
    command_normalized = _normalize_command(command).lower().strip()
    for pattern in _BANNED_PATTERNS:
        if pattern.search(command_normalized):
            return GateDecision(
                result=GateResult.DENY,
                reason="command contains a permanently banned pattern"
            )
    if FORK_BOMB_RE.search(command_normalized):
        return GateDecision(result=GateResult.DENY, reason="fork bomb pattern detected")
    return GateDecision(result=GateResult.ALLOW, reason="")


def check_rm_command(command: str, state: Session) -> GateDecision:
    """Legacy helper (the shell policy file now decides rm). Kept for callers."""
    command_normalized = _normalize_command(command).lower()
    for rm_re in _RM_PATTERNS_RE:
        if rm_re.search(command_normalized):
            return GateDecision(result=GateResult.DENY,
                                reason="rm/delete commands are banned")
    return GateDecision(result=GateResult.ALLOW, reason="")


def gate_shell_command(command: str, state: Session) -> GateDecision:
    """Gate for shell commands — resolved by the shell policy (shell-policy.json).

    Zed's order: built-in rules, then deny-any, confirm-any, allow-all, default.
    See shell_policy.py for the details.
    """
    decision = check_banned_command(command)
    if decision.result == GateResult.DENY:
        return decision

    tier = get_tier_config(state)
    pd = policy_evaluate(command, get_tier_level(state), get_policy())

    if pd.result == P_DENY:
        return GateDecision(result=GateResult.DENY, reason=pd.reason, title="shell")
    if pd.result == P_ALLOW:
        return GateDecision(result=GateResult.ALLOW, reason=pd.reason, title="shell")

    warning = ""
    if pd.writes and not tier.can_write_direct:
        warning = "this runs for REAL — shell commands are not staged to shadow"
    return GateDecision(
        result=GateResult.PROMPT,
        reason=pd.reason,
        title="shell",
        detail=command,
        allow_label="allow — run it",
        warning=warning,
        exception_keys=list(pd.exception_keys),
    )


# =============================================================================
# FILE ACCESS GATES
# =============================================================================

def gate_file_read(path: str, state: Session) -> GateDecision:
    """Gate for file read operations."""
    path = _expand_tilde_path(path)
    if is_path_excluded(path):
        return GateDecision(result=GateResult.DENY,
                            reason=f"path is in the privacy exclusion list: {path}")
    if not can_access_path(path):
        return GateDecision(result=GateResult.DENY,
                            reason=f"path is outside the allowed directories: {path}")
    return GateDecision(result=GateResult.ALLOW, reason="read permitted")


def _parent_key(path: str) -> Tuple[str, str]:
    parent = str(Path(path).expanduser().resolve().parent)
    return parent, parent.replace(str(Path.home()), "~", 1)


def gate_file_write(path: str, state: Session, content_preview: str = "") -> GateDecision:
    """Gate for file write operations."""
    tier = get_tier_config(state)
    path = _expand_tilde_path(path)

    # Writes into the shadow staging area itself are always allowed.
    if Path(path).resolve().is_relative_to(SHADOW_ROOT):
        return GateDecision(result=GateResult.ALLOW, reason="shadow staging area")

    if is_path_excluded(path):
        return GateDecision(result=GateResult.DENY,
                            reason=f"path is in the privacy exclusion list: {path}")

    parent, parent_disp = _parent_key(path)
    keys = [(f"write:{parent}", f"writes in {parent_disp}/")]
    detail = path + (f"\n\n{content_preview}" if content_preview else "")

    if is_path_writable(path) or is_path_read_only(path):
        if tier.can_write_direct and is_path_writable(path):
            return GateDecision(result=GateResult.ALLOW, reason="direct write permitted")
        shadow = get_shadow_path(path)
        where = "writable" if is_path_writable(path) else "read-only"
        return GateDecision(
            result=GateResult.PROMPT,
            reason=f"{state.trust_tier} tier stages writes ({where} path)",
            redirect_path=str(shadow),
            title="write file",
            detail=detail,
            allow_label="allow (stage to shadow)",
            warning=(f"will be written to shadow: {shadow} — type /tier 2 here to write it "
                     f"for real" if not tier.can_write_direct else
                     f"read-only path: will be written to shadow: {shadow}"),
            exception_keys=keys,
        )

    return GateDecision(result=GateResult.DENY,
                        reason=f"path is not in the allowed directories: {path}")


def gate_file_move(src: str, dst: str, state: Session) -> GateDecision:
    """Gate for move/rename.

    A move cannot be staged: rewriting only the destination into shadow (the
    old behavior) really MOVED the real file into the shadow tree and
    reported success. Moves are now direct-or-prompt, never redirected.
    """
    tier = get_tier_config(state)
    src = _expand_tilde_path(src or "")
    dst = _expand_tilde_path(dst or "")
    for p in (src, dst):
        if p and is_path_excluded(p):
            return GateDecision(result=GateResult.DENY,
                                reason=f"path is in the privacy exclusion list: {p}")
        if p and not can_access_path(p):
            return GateDecision(result=GateResult.DENY,
                                reason=f"path is outside the allowed directories: {p}")
    if dst and Path(dst).resolve().is_relative_to(SHADOW_ROOT):
        return GateDecision(result=GateResult.ALLOW, reason="move into the shadow staging area")
    if tier.can_write_direct and (not dst or is_path_writable(dst)) and (not src or is_path_writable(src)):
        return GateDecision(result=GateResult.ALLOW, reason="direct move permitted")
    parent, parent_disp = _parent_key(dst or src)
    return GateDecision(
        result=GateResult.PROMPT,
        reason=f"moves can't be staged to shadow at {state.trust_tier} tier",
        title="move file",
        detail=f"{src}\n  → {dst}",
        allow_label="allow — do the REAL move",
        warning="this moves the actual file (not a staged copy)",
        exception_keys=[(f"move:{parent}", f"moves into {parent_disp}/")],
    )


def gate_multi_file_write(paths: List[str], state: Session) -> GateDecision:
    """Gate for operations touching multiple files (kept for compatibility)."""
    tier = get_tier_config(state)
    if len(paths) < 3 or tier.level >= 2:
        return GateDecision(result=GateResult.ALLOW, reason="batch permitted")
    return GateDecision(result=GateResult.PROMPT,
                        reason=f"batch write of {len(paths)} files",
                        title="batch write", detail="\n".join(paths),
                        exception_keys=[("batch:write", "batch writes")])


# =============================================================================
# NETWORK / SSH GATES
# =============================================================================

def gate_network_request(url: str, state: Session, first_request: bool = False,
                         tool_key: str = "") -> GateDecision:
    """Gate for network requests. Below tier 2 it asks (was: hard deny)."""
    tier = get_tier_config(state)
    if tier.can_network:
        return GateDecision(result=GateResult.ALLOW, reason="network permitted")
    key = tool_key or url
    return GateDecision(
        result=GateResult.PROMPT,
        reason=f"network access isn't granted at {state.trust_tier} tier",
        title=f"network · {key}",
        detail=url,
        exception_keys=[(f"net:{key}", f"`{key}`")],
    )


def gate_ssh_command(host: str, command: str, state: Session) -> GateDecision:
    """Gate for SSH commands to other hosts (policy applies to the remote command)."""
    tier = get_tier_config(state)
    cmd_gate = gate_shell_command(command, state)
    if cmd_gate.result == GateResult.DENY:
        return cmd_gate
    needs_ssh_ok = not tier.can_ssh
    if not needs_ssh_ok and cmd_gate.result == GateResult.ALLOW:
        print_system(f"  🔗 SSH to {host}: {command[:60]}")
        return GateDecision(result=GateResult.ALLOW, reason="SSH permitted")
    reasons = []
    keys: List[Tuple[str, str]] = []
    if needs_ssh_ok:
        reasons.append(f"SSH isn't granted at {state.trust_tier} tier")
        keys.append((f"ssh:{host}", f"SSH to {host}"))
    if cmd_gate.result == GateResult.PROMPT:
        reasons.append(cmd_gate.reason)
        keys.extend(cmd_gate.exception_keys)
    return GateDecision(
        result=GateResult.PROMPT,
        reason="; ".join(reasons),
        title=f"ssh · {host}",
        detail=command,
        allow_label=f"allow — run on {host}",
        warning=cmd_gate.warning,
        exception_keys=keys,
    )


# =============================================================================
# APPROVAL MENU (1-7)
# =============================================================================

_MENU_HELP = """\
  1  allow                     run it
  2  allow + note              run it, and tell OGneey something (e.g. "back it up first")
  3  deny                      don't run it
  4  deny + reason             don't run it, and tell OGneey why
  5  say something             don't run it yet — send a message; OGneey decides again
  6  allow once                allow this kind of action until the current task ends
  7  allow this session        allow this kind of action until you /quit
  v  view                      show the full command / content
  You can put text right after the number:  "4 use the backup dir instead"
  Anything that isn't a choice is sent as [5] say something (never as an allow).
  Slash commands work here: /p pauses at the boundary, /pf stops now,
  /tier N (or /disable, /tools-off, ...) changes settings and re-checks this call."""

_PREVIEW_LINES = 40
_ALIASES = {
    "y": "1", "yes": "1", "a": "1", "allow": "1", "ok": "1",
    "n": "3", "no": "3", "d": "3", "deny": "3",
    "o": "6", "once": "6",
    "s": "7", "session": "7",
    "v": "v", "view": "v",
    "?": "?", "h": "?", "help": "?",
}


def _parse_choice(text: str) -> Tuple[Optional[str], str]:
    s = (text or "").strip()
    if not s:
        return None, ""
    m = re.match(r"^([1-7])(?:[\s.:,)\-]+(.*))?$", s, re.S)
    if m:
        return m.group(1), (m.group(2) or "").strip()
    low = s.lower()
    if low in _ALIASES:
        return _ALIASES[low], ""
    m = re.match(r"^(no|n|deny)[\s,:;.\-]+(.+)$", s, re.I | re.S)
    if m:
        return "4", m.group(2).strip()
    m = re.match(r"^(yes|y|ok|allow)[\s,:;.\-]+(.+)$", s, re.I | re.S)
    if m:
        return "2", m.group(2).strip()
    return "say", s


def _term_width() -> int:
    try:
        return max(40, min(100, shutil.get_terminal_size().columns - 2))
    except Exception:
        return 78


def _render_menu(decision: GateDecision, state: Session, full: bool = False):
    width = _term_width()
    title = decision.title or "action"
    head = f"┌─ APPROVAL · {title} "
    print()
    print_warning(head + "─" * max(4, width - len(head)))
    if decision.reason:
        print_warning(f"│ why: {decision.reason}")
    if decision.warning:
        print_warning(f"│ ⚠  {decision.warning}")
    lines = (decision.detail or "").splitlines() or [""]
    shown = lines if full else lines[:_PREVIEW_LINES]
    print_warning("│")
    for ln in shown:
        print(f"{COLORS.WARNING}│{COLORS.RESET}   {COLORS.BRIGHT_WHITE}{ln}{COLORS.RESET}")
    if len(shown) < len(lines):
        print_warning(f"│   … {len(lines) - len(shown)} more lines — [v] shows everything")
    print_warning("│")
    label = " + ".join(lbl for _, lbl in decision.exception_keys)
    print_warning(f"│ [1] {decision.allow_label}   [2] allow + note   [3] deny")
    print_warning("│ [4] deny + reason   [5] say something instead")
    if label:
        print_warning(f"│ [6] allow {label} until this task ends")
        print_warning(f"│ [7] allow {label} for this session")
    print_warning("│ [v] view full · [?] help · /p pause · /pf stop now · /tier N re-checks")
    print_warning("└" + "─" * (width - 1))


def _followup(label: str) -> Tuple[Optional[str], bool]:
    """Ask for the note/reason/message text. Returns (text, aborted)."""
    ans = get_hub().ask(f"{COLORS.COMMAND}  {label} › {COLORS.RESET}", kind="followup")
    if ans.kind in ("abort", "eof"):
        return None, True
    if ans.kind == "command":
        return None, ans.command in ("pause-force", "quit", "pause")
    return ans.text.strip(), False


def _exceptions_cover(state: Session, keys: List[Tuple[str, str]]) -> bool:
    if not keys:
        return False
    ex = getattr(state, "gate_exceptions", None) or {}
    active = set(ex.get("task", set())) | set(ex.get("session", set()))
    return all(k in active for k, _ in keys)


def add_gate_exceptions(state: Session, keys: List[Tuple[str, str]], scope: str):
    ex = state.gate_exceptions.setdefault(scope, set())
    for k, label in keys:
        ex.add(k)
        state.gate_exception_labels[k] = label


def _ask_operator(decision: GateDecision, state: Session) -> GateVerdict:
    hub = get_hub()
    _render_menu(decision, state)
    while True:
        ans = hub.ask(f"{COLORS.COMMAND}  approve [1-7] › {COLORS.RESET}", kind="gate")
        if ans.kind == "abort":
            print_warning("  ⏹ approval cancelled (force pause)")
            return GateVerdict(False, aborted=True, by_operator=True)
        if ans.kind == "eof":
            return GateVerdict(False, by_operator=True, reason="no answer (input closed)")
        if ans.kind == "command":
            if ans.command == "pause":
                return GateVerdict(False, paused=True, by_operator=True)
            if ans.command in ("pause-force", "quit"):
                return GateVerdict(False, aborted=True, by_operator=True)
            if not ans.command:
                print_dim("  (unknown command — still waiting on this approval)")
                continue
            return GateVerdict(False, regate=True)

        choice, text = _parse_choice(ans.text)
        if choice is None:
            print_dim("  type 1-7 (or ? for help)")
            continue
        if choice == "?":
            print_system(_MENU_HELP)
            continue
        if choice == "v":
            _render_menu(decision, state, full=True)
            continue
        if choice == "1":
            return GateVerdict(True, by_operator=True, note=text)
        if choice == "2":
            if not text:
                text, aborted = _followup("note for OGneey")
                if aborted:
                    return GateVerdict(False, aborted=True, by_operator=True)
            return GateVerdict(True, by_operator=True, note=text or "")
        if choice == "3":
            return GateVerdict(False, by_operator=True, reason=text)
        if choice == "4":
            if not text:
                text, aborted = _followup("reason")
                if aborted:
                    return GateVerdict(False, aborted=True, by_operator=True)
            return GateVerdict(False, by_operator=True, reason=text or "")
        if choice == "5":
            if not text:
                text, aborted = _followup("message for OGneey")
                if aborted:
                    return GateVerdict(False, aborted=True, by_operator=True)
                if not text:
                    continue
            return GateVerdict(False, by_operator=True, message=text)
        if choice in ("6", "7"):
            if not decision.exception_keys:
                print_warning("  6/7 aren't available for this kind of action — pick 1-5")
                continue
            scope = "task" if choice == "6" else "session"
            add_gate_exceptions(state, decision.exception_keys, scope)
            label = " + ".join(lbl for _, lbl in decision.exception_keys)
            until = "this task ends" if scope == "task" else "you /quit"
            print_success(f"  ✓ won't ask again for {label} until {until}  (/exceptions to review)")
            return GateVerdict(True, by_operator=True, note=text)
        if choice == "say":
            print_dim("  ↪ sent as [5] say something (the call was NOT run)")
            return GateVerdict(False, by_operator=True, message=text)


def enforce_decision(decision: GateDecision, state: Session) -> GateVerdict:
    """Apply a gate decision, asking the operator when needed."""
    if decision.result == GateResult.ALLOW:
        return GateVerdict(True)
    if decision.result == GateResult.DENY:
        print_error(f"  🚫 DENIED: {decision.reason}")
        return GateVerdict(False, reason=decision.reason)
    if decision.result == GateResult.REDIRECT:
        print_warning(f"  ↪ Redirected: {decision.reason}")
        return GateVerdict(True, redirect=decision.redirect_path)
    if decision.result == GateResult.MODIFY:
        return GateVerdict(True)
    if decision.result == GateResult.PROMPT:
        if _exceptions_cover(state, decision.exception_keys):
            label = " + ".join(lbl for _, lbl in decision.exception_keys)
            print_dim(f"  ✓ auto-approved ({label} — exception active)")
            return GateVerdict(True, redirect=decision.redirect_path, auto=label)
        verdict = _ask_operator(decision, state)
        if verdict.allowed:
            verdict.redirect = decision.redirect_path
        return verdict
    print_error(f"  🚫 DENIED: unhandled gate result {decision.result} (bug). {decision.reason}")
    return GateVerdict(False, reason="internal gate error")


def enforce_gate(decision: GateDecision, context: str = "", state: Session = None) -> Tuple[bool, Optional[str]]:
    """Legacy tuple interface: (allowed, redirect_path)."""
    if state is None:
        # Old call sites had no state; exceptions just don't apply there.
        state = _NullState()
    v = enforce_decision(decision, state)
    return v.allowed, v.redirect


class _NullState:
    trust_tier = "probation"
    gate_exceptions = {"task": set(), "session": set()}
    gate_exception_labels: Dict[str, str] = {}


# =============================================================================
# CONVENIENCE WRAPPERS (legacy signatures)
# =============================================================================

def check_and_enforce_shell(command: str, state: Session) -> Tuple[bool, Optional[str]]:
    v = enforce_decision(gate_shell_command(command, state), state)
    return v.allowed, None


def check_and_enforce_read(path: str, state: Session) -> Tuple[bool, None]:
    v = enforce_decision(gate_file_read(path, state), state)
    return v.allowed, None


def check_and_enforce_write(path: str, state: Session) -> Tuple[bool, Optional[str]]:
    v = enforce_decision(gate_file_write(path, state), state)
    return v.allowed, v.redirect


def check_and_enforce_ssh(host: str, command: str, state: Session) -> Tuple[bool, None]:
    """Used by the /ssh slash command. Re-checks after a slash command at the prompt."""
    for _ in range(6):
        v = enforce_decision(gate_ssh_command(host, command, state), state)
        if not v.regate:
            return v.allowed, None
    return False, None


# =============================================================================
# TOOL CALL GATE (HIGH LEVEL)
# =============================================================================

_FILE_READ_TOOLS = frozenset({
    "read_file", "read_text_file", "get_file", "view",
    "read_multiple_files", "search_files", "directory_tree",
    "get_file_info", "list_directory",
})
_FILE_WRITE_TOOLS = frozenset({
    "write_file", "create_file", "edit_file", "save_file",
    "create_directory",
})
_FILE_MOVE_TOOLS = frozenset({
    "move_file", "rename_file",
})
_SHELL_TOOLS = frozenset({
    "execute_command", "run_command", "shell", "bash",
})
# Network-egress tools — includes local/speak, which POSTs assistant text to
# an external TTS endpoint.
_NETWORK_TOOLS = frozenset({
    "speak", "navigate", "browser_navigate", "fetch", "web_fetch",
    "web_search", "searxng_web_search", "http_request", "download",
    "download_file", "get_url",
})
_PATH_ARG_KEYS = ("path", "paths", "source", "destination", "directory", "file_path")


def _extract_paths(args: Dict[str, Any]) -> List[str]:
    """Pull every filesystem path out of a tool-call args dict."""
    paths = []
    for key in _PATH_ARG_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val:
            paths.append(val)
        elif isinstance(val, list):
            paths.extend(p for p in val if isinstance(p, str) and p)
    return paths


def _content_preview(tool: str, args: Dict[str, Any]) -> str:
    if "content" in args and isinstance(args["content"], str):
        return args["content"]
    if "edits" in args and isinstance(args["edits"], list):
        out = []
        for i, e in enumerate(args["edits"], 1):
            if not isinstance(e, dict):
                continue
            old = str(e.get("oldText", e.get("old_text", "")))
            new = str(e.get("newText", e.get("new_text", "")))
            out.append(f"--- edit {i}: replace")
            out.extend("  - " + ln for ln in old.splitlines() or [""])
            out.append("    with")
            out.extend("  + " + ln for ln in new.splitlines() or [""])
        return "\n".join(out)
    return ""


def _emit(state: Session, server: str, tool: str, verdict: GateVerdict):
    hook = getattr(state, "gate_event_hook", None)
    if not hook:
        return
    try:
        hook({
            "tool": f"{server}/{tool}", "allowed": verdict.allowed,
            "by_operator": verdict.by_operator, "auto": verdict.auto,
            "note": verdict.note, "reason": verdict.reason, "message": verdict.message,
            "aborted": verdict.aborted, "paused": verdict.paused,
            "tier": state.trust_tier,
        })
    except Exception:
        pass


def gate_tool_call(server: str, tool: str, args: Dict[str, Any], state: Session) -> GateVerdict:
    """Main gate for any tool call.

    Returns a GateVerdict (allowed? final args? operator's words?). If the
    operator runs a slash command at the approval prompt (/tier 3, /disable,
    /tools-off, ...) the call is re-checked under the new settings.
    """
    original = copy.deepcopy(args)
    for _ in range(6):
        verdict = _gate_tool_call_once(server, tool, copy.deepcopy(original), state)
        if verdict.regate:
            print_dim(f"  ↻ re-checking {server}/{tool} with the new settings…")
            continue
        _emit(state, server, tool, verdict)
        return verdict
    return GateVerdict(False, reason="approval re-check limit reached")


def _merge(a: GateVerdict, b: GateVerdict) -> GateVerdict:
    if b.note:
        a.note = (a.note + "\n" + b.note).strip()
    a.by_operator = a.by_operator or b.by_operator
    a.auto = a.auto or b.auto
    return a


def _gate_tool_call_once(server: str, tool: str, args: Dict[str, Any], state: Session) -> GateVerdict:
    tier = get_tier_config(state)
    tool_key = f"{server}/{tool}"

    disabled = getattr(state, "disabled_tools", set()) or set()
    if server in disabled or tool_key in disabled:
        print_warning(f"  🚫 Tool disabled: {tool_key}")
        return GateVerdict(False, reason=f"{tool_key} is disabled by the operator (/enable to restore)")

    if not state.tools_enabled:
        print_warning("  🔒 Tools disabled. Use /tools-on to enable.")
        return GateVerdict(False, reason="tools are switched off by the operator (chat-only mode)")

    # Shell commands: the policy file decides.
    if tool in _SHELL_TOOLS:
        command = args.get("command", "")
        decision = gate_shell_command(command, state)
        decision.title = tool_key
        v = enforce_decision(decision, state)
        v.args = args if v.allowed else None
        return v

    # File reads
    if tool in _FILE_READ_TOOLS:
        for path in _extract_paths(args):
            v = enforce_decision(gate_file_read(path, state), state)
            if not v.allowed:
                return v
        return GateVerdict(True, args=args)

    # File writes (writes, edits, directory creation)
    if tool in _FILE_WRITE_TOOLS:
        result = GateVerdict(True, args=args)
        preview = _content_preview(tool, args)
        for path in _extract_paths(args):
            decision = gate_file_write(path, state, preview)
            decision.title = f"{tool_key}"
            v = enforce_decision(decision, state)
            if not v.allowed:
                return v
            _merge(result, v)
            if v.redirect and v.redirect != "EXPLAIN":
                for key in _PATH_ARG_KEYS:
                    if args.get(key) == path:
                        args[key] = v.redirect
                Path(v.redirect).parent.mkdir(parents=True, exist_ok=True)
        result.args = args
        return result

    # Move/rename: never redirected into shadow (that relocated real files).
    if tool in _FILE_MOVE_TOOLS:
        src = args.get("source", "") or args.get("path", "")
        dst = args.get("destination", "") or args.get("new_path", "")
        decision = gate_file_move(src, dst, state)
        decision.title = tool_key
        v = enforce_decision(decision, state)
        v.args = args if v.allowed else None
        v.redirect = None
        return v

    # Network egress (web tools, browser, local/speak TTS, ...)
    if tool in _NETWORK_TOOLS:
        url = (args.get("url") or args.get("uri") or args.get("endpoint")
               or args.get("query") or tool_key)
        v = enforce_decision(gate_network_request(str(url), state, tool_key=tool_key), state)
        v.args = args if v.allowed else None
        return v

    # Tier 0: every tool needs approval
    if tier.requires_tool_approval:
        try:
            detail = json.dumps(args, indent=2, default=str)
        except (TypeError, ValueError):
            detail = str(args)
        decision = GateDecision(
            result=GateResult.PROMPT,
            reason=f"{state.trust_tier} tier: every tool call needs approval",
            title=tool_key,
            detail=detail,
            exception_keys=[(f"tool:{tool_key}", f"`{tool_key}`")],
        )
        v = enforce_decision(decision, state)
        v.args = args if v.allowed else None
        return v

    return GateVerdict(True, args=args)
