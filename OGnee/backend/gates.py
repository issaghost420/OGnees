#!/usr/bin/env python3
"""
gates.py - Approval prompts and permission checks

The safety layer. Catches writes, sudo, rm, multi-file ops.
Turns OSeemies into OGreenies through demonstrated competence.
"""

import os
import re
import unicodedata  # FIX 2: Unicode normalization for interpreter bypass detection
from enum import Enum, auto
from dataclasses import dataclass
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
    prompt_approval,
    prompt_choice,
    COLORS,
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
    redirect_path: Optional[str] = None     # For REDIRECT
    modified_command: Optional[str] = None  # For MODIFY
    prompt_options: Optional[Dict] = None   # For PROMPT

# =============================================================================
# TRUST LEVEL HELPERS
# =============================================================================

def get_tier_name(state: Session) -> str:
    """Get display name based on trust level."""
    tier_name = state.trust_tier.lower()
    
    if tier_name == "probation":
        return f"{COLORS.WARNING}OGreenie{COLORS.RESET}"
    elif tier_name == "supervised":
        return f"{COLORS.SYSTEM}OSeemie{COLORS.RESET}"
    elif tier_name == "trusted":
        return f"{COLORS.SUCCESS}OGneey{COLORS.RESET}"
    elif tier_name == "autonomous":
        return f"{COLORS.SUCCESS}OFreeney 🌟{COLORS.RESET}"
    else:
        return f"OGneey"

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
    """Check if command is permanently banned."""
    # FIX 1 + FIX 2: Normalize Unicode and strip before matching.
    command_normalized = _normalize_command(command).lower().strip()
    
    # FIX 1: Use pre-compiled boundary-aware regex patterns instead of
    # substring matching (`banned.lower() in command_lower`).
    # The old code would match "echo 'rm -rf / is bad'" because
    # "rm -rf /" is a literal substring of the quoted text.
    # The new patterns use (?<!\w) and (?!\w) boundaries to ensure
    # banned commands are matched as whole tokens, not substrings.
    for pattern in _BANNED_PATTERNS:
        if pattern.search(command_normalized):
            return GateDecision(
                result=GateResult.DENY,
                reason="Command contains banned pattern"
            )
    
    # FIX 5: Check fork bomb with regex that tolerates arbitrary whitespace.
    # Pattern: : ( ) { : | : & } ; : — each token can be space/tab separated.
    # The old code used an exact string match for ":(){:|:&};:" which was
    # trivially bypassed by adding spaces: ": ( ) { : | : & } ; :"
    if FORK_BOMB_RE.search(command_normalized):
        return GateDecision(
            result=GateResult.DENY,
            reason="Fork bomb pattern detected"
        )
    
    return GateDecision(result=GateResult.ALLOW, reason="")

def check_rm_command(command: str, state: Session) -> GateDecision:
    """Check for rm/delete commands - ALWAYS denied."""
    # FIX 2: Normalize Unicode to catch homoglyph evasion.
    command_normalized = _normalize_command(command).lower()
    
    for rm_re in _RM_PATTERNS_RE:
        if rm_re.search(command_normalized):
            return GateDecision(
                result=GateResult.DENY,
                reason="rm/delete commands are permanently banned. "
                       "No OGneey gets rm privileges. Ever. "
                       "Pattern matched"
            )
    
    return GateDecision(result=GateResult.ALLOW, reason="")

def check_sudo_command(command: str, state: Session) -> GateDecision:
    """Check for sudo/privilege escalation."""
    command_normalized = _normalize_command(command).lower()
    tier = get_tier_config(state)
    
    #   - \bsudo\b       : matches "sudo" as a word
    #   - \bsudoedit\b   : matches "sudoedit" (the old bypass)
    #   - \bdoas\b       : matches "doas" privilege escalation
    #   - \bpkexec\b     : matches "pkexec"
    #   - \bsu\s+-c\b   : matches "su -c" with any whitespace
    #   - \bsu\s+root\b : matches "su root" with any whitespace
    for pattern_re in SUDO_PATTERNS_RE:
        if pattern_re.search(command_normalized):
            if tier.can_sudo:
                # Tier 3 can sudo but we still log it
                print_warning(f"  ⚡ sudo permitted (tier: {state.trust_tier})")
                return GateDecision(result=GateResult.ALLOW, reason="Tier permits sudo")
            else:
                # Need approval
                return GateDecision(
                    result=GateResult.PROMPT,
                    reason=f"sudo requires approval (current tier: {state.trust_tier})",
                    prompt_options={
                        "1": "Allow this once",
                        "2": "Deny",
                        "3": "Explain to OGneey why this needs sudo",
                    }
                )
    
    return GateDecision(result=GateResult.ALLOW, reason="")

def gate_shell_command(command: str, state: Session) -> GateDecision:
    """Main gate for shell commands."""
    
    # Check banned first - no exceptions
    decision = check_banned_command(command)
    if decision.result == GateResult.DENY:
        return decision
    
    # Check rm - NEVER allowed
    decision = check_rm_command(command, state)
    if decision.result == GateResult.DENY:
        return decision
    
    # Check sudo
    decision = check_sudo_command(command, state)
    if decision.result != GateResult.ALLOW:
        return decision
    
    # Check for writes via shell (redirects, sed -i, tee, patch, etc.)
    tier = get_tier_config(state)
    if not tier.can_write_direct:
        write_patterns = [
            (r'[^2]?>\s*\S', 'output redirect'),               # > or >> to file
            (r'\btee\s+', 'tee write'),                         # tee to file
            (r'\bsed\s+.*-i', 'sed in-place edit'),             # sed -i
            (r'\bpatch\b', 'patch apply'),                      # patch
            (r'\bdd\s+', 'dd write'),                           # dd write
            (r'\bcp\s+', 'cp copy'),                            # cp (creates files)
            (r'\bmv\s+', 'mv move'),                            # mv (modifies filesystem)
            (r'\bchmod\b', 'chmod'),                            # chmod
            (r'\bchown\b', 'chown'),                            # chown
            (r'\bmkdir\b', 'mkdir'),                            # mkdir
            (r'\btouch\b', 'touch'),                            # touch
            (r'\bln\s+', 'symlink'),                            # ln
            (r'\bgit\s+push', 'git push'),                      # git push
            (r'\bgit\s+commit', 'git commit'),                  # git commit
            (r'\bnixos-rebuild', 'nixos-rebuild'),              # nixos-rebuild
            (r'\bsystemctl\s+(start|stop|restart|enable|disable)', 'systemctl'),
        ]
        
        # FIX 2: Normalize Unicode to catch homoglyph evasion.
        command_str = _normalize_command(command).lower()
        for pattern, desc in write_patterns:
            if re.search(pattern, command_str):
                cmd_preview = command[:100] + ("..." if len(command) > 100 else "")
                return GateDecision(
                    result=GateResult.PROMPT,
                    reason=f"Shell command would modify the filesystem ({desc})",
                    prompt_options={
                        "1": f"Allow: {cmd_preview}",
                        "2": "Deny",
                    }
                )
        
        # FIX 2: Check interpreter-based file-write bypasses.
        # Language interpreters (python, node, ruby, perl, sqlite3, vim,
        # emacs, ed) can write files without using shell redirections or
        # standard filesystem-modifying commands, completely bypassing
        # the regex patterns above. Unicode NFKC normalization is applied
        # upstream in command_str to defeat homoglyph attacks.
        for interp_re, interp_desc in _INTERPRETER_WRITE_PATTERNS:
            if interp_re.search(command_str):
                cmd_preview = command[:100] + ("..." if len(command) > 100 else "")
                return GateDecision(
                    result=GateResult.PROMPT,
                    reason=f"Shell command would modify the filesystem ({interp_desc})",
                    prompt_options={
                        "1": f"Allow: {cmd_preview}",
                        "2": "Deny",
                    }
                )
    
    # All clear
    return GateDecision(result=GateResult.ALLOW, reason="Command permitted")

# =============================================================================
# FILE ACCESS GATES
# =============================================================================

def gate_file_read(path: str, state: Session) -> GateDecision:
    """Gate for file read operations."""
    
    # FIX 6: Expand tilde before validation to ensure ~/ paths are
    # properly resolved and matched against exclusion lists.
    # Path.resolve() does NOT expand ~ — it remains a literal tilde.
    path = _expand_tilde_path(path)
    
    # Check if path is excluded (Documents, Downloads, Trash, etc)
    if is_path_excluded(path):
        return GateDecision(
            result=GateResult.DENY,
            reason=f"Path is in exclusion list (privacy): {path}"
        )
    
    # Check if we can access at all
    if not can_access_path(path):
        return GateDecision(
            result=GateResult.DENY,
            reason=f"Path is outside allowed directories: {path}"
        )
    
    return GateDecision(result=GateResult.ALLOW, reason="Read permitted")

def gate_file_write(path: str, state: Session, content_preview: str = "") -> GateDecision:
    """Gate for file write operations."""
    tier = get_tier_config(state)
    
    # FIX 6: Expand tilde before validation.
    path = _expand_tilde_path(path)

    # Writes targeting the shadow staging area itself are ALWAYS allowed:
    # the file is already staged for review. Re-redirecting it through
    # get_shadow_path() is what produced the doubled
    # "shadow/home/.../shadow/home/..." paths.
    if Path(path).resolve().is_relative_to(SHADOW_ROOT):
        return GateDecision(result=GateResult.ALLOW, reason="Shadow staging area")

    # Check exclusions first
    if is_path_excluded(path):
        return GateDecision(
            result=GateResult.DENY,
            reason=f"Path is in exclusion list: {path}"
        )
    
    # If path is in writable list, check tier
    if is_path_writable(path):
        if tier.can_write_direct:
            return GateDecision(result=GateResult.ALLOW, reason="Direct write permitted")
        elif tier.can_write_shadow:
            # Prompt before writing — let user see what's happening
            shadow = get_shadow_path(path)
            return GateDecision(
                result=GateResult.PROMPT,
                reason=f"Write to {path} (will go to shadow)",
                redirect_path=str(shadow),
                prompt_options={
                    "1": "Allow (writes to shadow)",
                    "2": "Deny",
                }
            )
        else:
            # OSeemie tier - needs approval
            return GateDecision(
                result=GateResult.PROMPT,
                reason=f"Write requires approval (tier: {state.trust_tier})",
                prompt_options={
                    "1": "Allow write to shadow",
                    "2": "Deny",
                    "3": "Show me what it wants to write",
                }
            )
    
    # If path is read-only, always redirect to shadow
    if is_path_read_only(path):
        if tier.can_write_shadow:
            shadow = get_shadow_path(path)
            return GateDecision(
                result=GateResult.PROMPT,
                reason=f"Write to read-only path {path} (will go to shadow)",
                redirect_path=str(shadow),
                prompt_options={
                    "1": "Allow (writes to shadow)",
                    "2": "Deny",
                }
            )
        else:
            return GateDecision(
                result=GateResult.PROMPT,
                reason=f"OSeemie wants to modify read-only file",
                prompt_options={
                    "1": "Allow to shadow",
                    "2": "Deny",
                    "3": "Explain the situation",
                }
            )
    
    # Path not in any allowed list
    return GateDecision(
        result=GateResult.DENY,
        reason=f"Path not in allowed directories: {path}"
    )

def gate_multi_file_write(paths: List[str], state: Session) -> GateDecision:
    """Gate for operations touching multiple files."""
    tier = get_tier_config(state)
    
    if len(paths) < 3:
        # Less than 3 files, handle individually
        return GateDecision(result=GateResult.ALLOW, reason="Small batch, proceed")
    
    # 3+ files - require review
    if tier.level >= 2:
        # Trusted+ can batch write
        print_warning(f"  📦 Batch write: {len(paths)} files")
        return GateDecision(result=GateResult.ALLOW, reason="Tier permits batch")
    
    # Lower tiers need approval
    return GateDecision(
        result=GateResult.PROMPT,
        reason=f"Batch write of {len(paths)} files requires approval",
        prompt_options={
            "1": f"Allow all {len(paths)} files",
            "2": "Review files one by one",
            "3": "Deny all",
        }
    )

# =============================================================================
# NETWORK GATES
# =============================================================================

def gate_network_request(url: str, state: Session, first_request: bool = False) -> GateDecision:
    """Gate for network requests."""
    tier = get_tier_config(state)
    
    if tier.can_network:
        return GateDecision(result=GateResult.ALLOW, reason="Network permitted")
    
    if first_request:
        # First network request of session needs approval
        return GateDecision(
            result=GateResult.PROMPT,
            reason=f"First network request requires approval",
            prompt_options={
                "1": "Allow network access for this session",
                "2": "Allow this request only",
                "3": "Deny network access",
            }
        )
    
    return GateDecision(result=GateResult.DENY, reason="Network access not permitted")

def gate_ssh_command(host: str, command: str, state: Session) -> GateDecision:
    """Gate for SSH commands to other hosts."""
    tier = get_tier_config(state)
    
    # First check the command itself for banned patterns
    cmd_gate = gate_shell_command(command, state)
    if cmd_gate.result == GateResult.DENY:
        return cmd_gate
    
    if tier.can_ssh:
        print_system(f"  🔗 SSH to {host}: {command[:50]}...")
        return GateDecision(result=GateResult.ALLOW, reason="SSH permitted")
    
    return GateDecision(
        result=GateResult.PROMPT,
        reason=f"SSH to {host} requires approval",
        prompt_options={
            "1": f"Allow SSH to {host}",
            "2": "Deny",
            "3": "See full command first",
        }
    )

# =============================================================================
# GATE ENFORCEMENT
# =============================================================================

def enforce_gate(decision: GateDecision, context: str = "") -> Tuple[bool, Optional[str]]:
    """
    Enforce a gate decision, prompting user if needed.
    Returns (allowed, redirect_path_or_none)
    """
    
    if decision.result == GateResult.ALLOW:
        return True, None
    
    elif decision.result == GateResult.DENY:
        print_error(f"  🚫 DENIED: {decision.reason}")
        return False, None
    
    elif decision.result == GateResult.REDIRECT:
        print_warning(f"  ↪ Redirected: {decision.reason}")
        if decision.redirect_path:
            print_system(f"    → {decision.redirect_path}")
        return True, decision.redirect_path
    
    elif decision.result == GateResult.PROMPT:
        # Ask user
        if decision.prompt_options:
            choice = prompt_choice(
                f"GATE: {decision.reason}",
                decision.prompt_options
            )
            
            if choice == "1":
                return True, decision.redirect_path
            elif choice == "2":
                return False, None
            elif choice == "3":
                # User wants more info - return special signal
                return False, "EXPLAIN"
            else:
                return False, None
        else:
            response = prompt_approval(
                context or "Action",
                decision.reason
            )
            return response == "y", None
    
    elif decision.result == GateResult.MODIFY:
        print_warning(f"  ✏️ Modified: {decision.reason}")
        if decision.modified_command:
            print_system(f"    → {decision.modified_command}")
        return True, decision.modified_command
    
    else:
        # FIX 7: Explicit else clause catches any unhandled GateResult values
        # (e.g., a new enum member added elsewhere but not handled here).
        # Previously this silently fell through with `return False, None`,
        # masking logic errors and making debugging nearly impossible.
        # Now it logs an error and denies, failing safe.
        print_error(f"  🚫 DENIED: Unhandled gate result {decision.result}. "
                    f"This is a bug - report it. Reason: {decision.reason}")
        return False, None

# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================

def check_and_enforce_shell(command: str, state: Session) -> Tuple[bool, Optional[str]]:
    """Check and enforce shell command gate. Returns (allowed, modified_command)."""
    decision = gate_shell_command(command, state)
    return enforce_gate(decision, f"Shell: {command[:60]}...")

def check_and_enforce_read(path: str, state: Session) -> Tuple[bool, None]:
    """Check and enforce file read gate."""
    decision = gate_file_read(path, state)
    allowed, _ = enforce_gate(decision, f"Read: {path}")
    return allowed, None

def check_and_enforce_write(path: str, state: Session) -> Tuple[bool, Optional[str]]:
    """Check and enforce file write gate. Returns (allowed, redirect_path)."""
    decision = gate_file_write(path, state)
    return enforce_gate(decision, f"Write: {path}")

def check_and_enforce_ssh(host: str, command: str, state: Session) -> Tuple[bool, None]:
    """Check and enforce SSH gate."""
    decision = gate_ssh_command(host, command, state)
    allowed, _ = enforce_gate(decision, f"SSH {host}: {command[:40]}...")
    return allowed, None

# =============================================================================
# TOOL CALL GATE (HIGH LEVEL)
# =============================================================================

# Capability classes — gate by what a tool DOES, not one exact name string.
# Unknown write-capable tools fall through to the tier approval check.
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


def gate_tool_call(
    server: str,
    tool: str,
    args: Dict[str, Any],
    state: Session
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Main gate for any tool call.
    Returns (allowed, modified_args_or_none)
    """
    tier = get_tier_config(state)
    tool_key = f"{server}/{tool}"
    
    # Check if tool is disabled
    if hasattr(state, 'disabled_tools'):
        if server in state.disabled_tools or tool_key in state.disabled_tools:
            print_warning(f"  🚫 Tool disabled: {tool_key}")
            return False, None
    
    # Check if tools are globally off
    if not state.tools_enabled:
        print_warning(f"  🔒 Tools disabled. Use /tools-on to enable.")
        return False, None
    
    # Shell commands get special handling
    if tool in _SHELL_TOOLS:
        command = args.get("command", "")
        allowed, modified = check_and_enforce_shell(command, state)
        if modified and modified != "EXPLAIN":
            args["command"] = modified
        return allowed, args if allowed else None

    # File read tools (single path, path lists, directory listings)
    if tool in _FILE_READ_TOOLS:
        for path in _extract_paths(args):
            allowed, _ = check_and_enforce_read(path, state)
            if not allowed:
                return False, None
        return True, args

    # File write tools (writes, edits, directory creation)
    if tool in _FILE_WRITE_TOOLS:
        for path in _extract_paths(args):
            allowed, redirect = check_and_enforce_write(path, state)
            if not allowed:
                return False, None
            if redirect and redirect != "EXPLAIN":
                # Apply the shadow redirect to whichever arg held this path
                for key in _PATH_ARG_KEYS:
                    if args.get(key) == path:
                        args[key] = redirect
                Path(redirect).parent.mkdir(parents=True, exist_ok=True)
        return True, args

    # Move/rename: source must be readable, destination must pass write gate
    if tool in _FILE_MOVE_TOOLS:
        src = args.get("source", "") or args.get("path", "")
        dst = args.get("destination", "") or args.get("new_path", "")
        if src:
            allowed, _ = check_and_enforce_read(src, state)
            if not allowed:
                return False, None
        if dst:
            allowed, redirect = check_and_enforce_write(dst, state)
            if not allowed:
                return False, None
            if redirect and redirect != "EXPLAIN":
                for key in ("destination", "new_path"):
                    if args.get(key) == dst:
                        args[key] = redirect
                Path(redirect).parent.mkdir(parents=True, exist_ok=True)
        return True, args

    # Network egress (web tools, browser, local/speak TTS, ...)
    if tool in _NETWORK_TOOLS:
        url = (args.get("url") or args.get("uri") or args.get("endpoint")
               or tool_key)
        decision = gate_network_request(url, state)
        allowed, _ = enforce_gate(decision, f"Network: {tool_key}")
        return allowed, args if allowed else None
    
    # If tier requires approval for all tools
    if tier.requires_tool_approval:
        response = prompt_approval(
            f"Tool call: {tool_key}",
            f"Args: {str(args)[:100]}...",
            ["y", "n", "details"]
        )
        if response == "y":
            return True, args
        elif response == "details":
            print_system(f"Full args: {args}")
            return False, None
        else:
            return False, None
    
    # Default: allow
    return True, args
