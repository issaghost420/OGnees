#!/usr/bin/env python3
"""
shell_policy.py - Zed-style allow / confirm / deny policy for shell commands

This is the brain behind every ``execute_command`` gate decision. It reads a
policy file in the same shape Zed uses for its terminal tool permissions and
resolves it the same way Zed's resolver does
(zed: crates/agent/src/tool_permissions.rs):

  1. Built-in hard rules (recursive delete of / ~ $HOME . ..) -> DENY, always.
  2. Any invalid regex in the policy                         -> DENY (fail closed).
  3. Split the command into sub-commands (&&, ||, ;, |, &, |&, newlines,
     $(...), backticks, <(...), >(...), subshells, bash -c '...', eval).
  4. ANY sub-command matches an always_deny pattern          -> DENY
  5. ANY sub-command matches an always_confirm pattern       -> CONFIRM
  6. EVERY sub-command matches some always_allow pattern     -> ALLOW
  7. Otherwise the default.
  If the command can't be parsed (unbalanced quotes etc.) allow rules are
  switched off for it — exactly what Zed does.

Two places where OGnee goes past Zed, both switchable in the policy file:

  * ``"match": "word"`` (default). Patterns must start at a word boundary, so
    ``rm`` matches ``rm``, ``rmdir`` and ``/bin/rm`` but not ``format``,
    ``confirm`` or ``docker run --rm``; ``sed`` doesn't fire on ``used``.
    Allow patterns must match the START of a sub-command and end on a word
    boundary (``ls`` allows ``ls -la`` but not ``lsof`` or ``python x ls``),
    and an allowed sub-command that writes a file with ``>`` stops counting
    as allowed (``cat x > /etc/foo`` is not "just cat").
    ``"match": "zed"`` gives you byte-for-byte Zed semantics instead
    (unanchored, case-insensitive regex search on each sub-command).

  * ``"default": "tier"`` (default). Commands no list mentions are decided by
    OGnee's trust tier: tier 0 asks, tier 1 asks only if the command writes
    something (redirect, cp/mv/sed -i/git commit/..., or runs an interpreter),
    tiers 2-3 allow. ``"confirm"`` / ``"allow"`` / ``"deny"`` behave like Zed.

Heredoc bodies are data, not commands (``cat > f <<EOF ... EOF`` doesn't get
its body scanned), unless the heredoc is fed to a shell (``bash <<EOF``), in
which case the body IS parsed as commands.

Pure module: no OGnee imports, so it can be tested on its own.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ALLOW = "allow"
CONFIRM = "confirm"
DENY = "deny"
_RANK = {ALLOW: 0, CONFIRM: 1, DENY: 2}

MATCH_WORD = "word"
MATCH_ZED = "zed"
DEFAULT_TIER = "tier"

_MAX_DEPTH = 8

# Characters that make a plain string pattern "regex" rather than literal.
# A bare '.' does NOT count — `.cache` and `.local` are meant literally.
_REGEX_META = set("()[]{}|*+?^$\\")

# Left boundary used in word mode: the pattern can't be glued onto the end of
# another word, a $VAR, a dotted name, or a --flag.
_LEFT_BOUNDARY = r"(?<![\w$.\-])"
# Right boundary for allow patterns: `ls` must not allow `lsof`.
_RIGHT_BOUNDARY = r"(?![\w\-])"


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class Redirect:
    op: str
    target: str
    fd: str = ""


@dataclass
class Segment:
    """One simple command extracted from a (possibly compound) command line."""
    text: str
    redirects: List[Redirect] = field(default_factory=list)
    heredocs: List[Tuple[str, str]] = field(default_factory=list)  # (delim, body)
    origin: str = "top"   # top | subst | subshell | wrapper | heredoc-shell

    @property
    def collapsed(self) -> str:
        return re.sub(r"\s+", " ", self.text).strip()

    def words(self) -> List[str]:
        return shell_words(self.text)


@dataclass
class Rule:
    kind: str          # deny | confirm | allow
    source: str        # the pattern exactly as written in the policy file
    regex: Optional[re.Pattern]
    note: str = ""
    error: str = ""


@dataclass
class Trigger:
    kind: str          # deny | confirm | default | builtin | invalid | parse
    pattern: str
    segment: str
    detail: str = ""


@dataclass
class PolicyDecision:
    result: str                                  # allow | confirm | deny
    reason: str                                  # one-line human summary
    triggers: List[Trigger] = field(default_factory=list)
    exception_keys: List[Tuple[str, str]] = field(default_factory=list)  # (key, label)
    segments: List[Segment] = field(default_factory=list)
    parse_ok: bool = True
    writes: List[Tuple[str, str]] = field(default_factory=list)  # (segment, why)
    source: str = "builtin"


# =============================================================================
# SHELL PARSING
# =============================================================================

def shell_words(text: str) -> List[str]:
    """Best-effort POSIX word split (quotes removed). Never raises."""
    try:
        lex = shlex.shlex(text, posix=True)
        lex.whitespace_split = True
        lex.commenters = ""
        return list(lex)
    except ValueError:
        return text.split()


def _find_balanced(text: str, start: int, open_ch: str = "(", close_ch: str = ")") -> int:
    """Index just past the close_ch matching an already-consumed open_ch.

    ``start`` points at the first character inside the opening bracket.
    Quotes are respected. Returns -1 if unbalanced.
    """
    depth = 1
    i = start
    n = len(text)
    quote = None
    while i < n:
        c = text[i]
        if quote == "'":
            if c == "'":
                quote = None
            i += 1
            continue
        if quote == '"':
            if c == "\\":
                i += 2
                continue
            if c == '"':
                quote = None
            i += 1
            continue
        if c == "\\":
            i += 2
            continue
        if c in ("'", '"'):
            quote = c
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _find_backtick(text: str, start: int) -> int:
    """Index just past the closing backtick (start = first char inside)."""
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "`":
            return i + 1
        i += 1
    return -1


def _read_word(text: str, i: int) -> Tuple[str, int, bool]:
    """Read one shell word starting at i (after skipping blanks).

    Returns (raw_word, next_index, was_quoted).
    """
    n = len(text)
    while i < n and text[i] in " \t":
        i += 1
    start = i
    quoted = False
    quote = None
    while i < n:
        c = text[i]
        if quote:
            if quote == '"' and c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            quoted = True
            i += 1
            continue
        if c == "\\":
            quoted = True
            i += 2
            continue
        if c in " \t\n;&|<>()":
            break
        i += 1
    return text[start:i], i, quoted


def _unquote(word: str) -> str:
    parts = shell_words(word)
    return parts[0] if parts else word


_REDIRECT_OPS = ("&>>", "&>", "<<<", "<<-", "<<", "<>", "<&", ">>", ">|", ">&", "<", ">")


def _parse_into(text: str, out: List[Segment], origin: str, depth: int) -> bool:
    """Split *text* into simple commands, appending them to *out*.

    Returns False if the text could not be parsed (unbalanced quotes or
    brackets, or nesting deeper than _MAX_DEPTH).
    """
    if depth > _MAX_DEPTH:
        return False

    ok = True
    n = len(text)
    i = 0
    cur: List[str] = []
    redirects: List[Redirect] = []
    # heredocs waiting for their body: [delim, strip_tabs, owner_segment_or_None]
    pending: List[list] = []
    quote: Optional[str] = None

    def flush():
        seg_text = "".join(cur).strip()
        seg = None
        if seg_text:
            seg = Segment(seg_text, list(redirects), [], origin)
            out.append(seg)
        for p in pending:
            if p[2] is None:
                p[2] = seg
        cur.clear()
        redirects.clear()
        return seg

    def recurse(inner: str, sub_origin: str):
        nonlocal ok
        if not _parse_into(inner, out, sub_origin, depth + 1):
            ok = False

    while i < n:
        c = text[i]

        # ---------- inside single quotes: literal ----------
        if quote == "'":
            cur.append(c)
            if c == "'":
                quote = None
            i += 1
            continue

        # ---------- inside double quotes: $() and `` still expand ----------
        if quote == '"':
            if c == "\\" and i + 1 < n:
                cur.append(text[i:i + 2])
                i += 2
                continue
            if c == '"':
                quote = None
                cur.append(c)
                i += 1
                continue
            if text.startswith("$((", i):
                j = _find_balanced(text, i + 3)
                j = _find_balanced(text, j) if j != -1 else -1
                if j == -1:
                    return False
                cur.append(text[i:j])
                i = j
                continue
            if text.startswith("$(", i):
                j = _find_balanced(text, i + 2)
                if j == -1:
                    return False
                recurse(text[i + 2:j - 1], "subst")
                cur.append(text[i:j])
                i = j
                continue
            if c == "`":
                j = _find_backtick(text, i + 1)
                if j == -1:
                    return False
                recurse(text[i + 1:j - 1], "subst")
                cur.append(text[i:j])
                i = j
                continue
            cur.append(c)
            i += 1
            continue

        # ---------- unquoted ----------
        if c == "\\":
            if i + 1 < n and text[i + 1] == "\n":
                i += 2          # line continuation
                continue
            cur.append(text[i:i + 2])
            i += 2
            continue

        if c in ("'", '"'):
            quote = c
            cur.append(c)
            i += 1
            continue

        if c == "#" and (not cur or cur[-1][-1:] in (" ", "\t", "\n", ";", "&", "|", "(")):
            # comment to end of line (the newline itself is handled below)
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue

        if text.startswith("$((", i):
            j = _find_balanced(text, i + 3)
            j = _find_balanced(text, j) if j != -1 else -1
            if j == -1:
                return False
            cur.append(text[i:j])
            i = j
            continue

        if text.startswith("$(", i):
            j = _find_balanced(text, i + 2)
            if j == -1:
                return False
            recurse(text[i + 2:j - 1], "subst")
            cur.append(text[i:j])
            i = j
            continue

        if c == "`":
            j = _find_backtick(text, i + 1)
            if j == -1:
                return False
            recurse(text[i + 1:j - 1], "subst")
            cur.append(text[i:j])
            i = j
            continue

        if text.startswith(("<(", ">("), i):
            j = _find_balanced(text, i + 2)
            if j == -1:
                return False
            recurse(text[i + 2:j - 1], "subst")
            cur.append(text[i:j])
            i = j
            continue

        if c == "(":
            if "".join(cur).strip() == "":
                # subshell / group at the start of a command
                j = _find_balanced(text, i + 1)
                if j == -1:
                    return False
                flush()
                recurse(text[i + 1:j - 1], "subshell")
                i = j
                continue
            if cur and cur[-1].endswith("="):
                # array assignment: arr=(a b c)
                j = _find_balanced(text, i + 1)
                if j == -1:
                    return False
                cur.append(text[i:j])
                i = j
                continue
            cur.append(c)       # e.g. function name()
            i += 1
            continue

        if c == ")":
            # stray close (case patterns `a)`): keep as text
            cur.append(c)
            i += 1
            continue

        # ---------- redirections ----------
        fd = ""
        k = i
        while k < n and text[k].isdigit():
            k += 1
        if k > i and k < n and text[k] in "<>" and (i == 0 or text[i - 1] in " \t\n;&|("):
            fd = text[i:k]
        else:
            k = i
        op = None
        if k < n and (text[k] in "<>" or text.startswith("&>", k)):
            for cand in _REDIRECT_OPS:
                if text.startswith(cand, k):
                    op = cand
                    break
        if op is not None:
            j = k + len(op)
            word, j2, was_quoted = _read_word(text, j)
            cur.append(text[i:j2])
            cur.append(" ")
            if op in ("<<", "<<-"):
                delim = _unquote(word) if word else ""
                if delim:
                    pending.append([delim, op == "<<-", None])
            elif op in (">&", "<&") and (word.isdigit() or word == "-"):
                pass            # fd duplication, not a file
            elif op == "<<<":
                pass            # here-string
            else:
                redirects.append(Redirect(op, _unquote(word) if word else "", fd))
            i = j2
            continue

        # ---------- separators ----------
        if c == "\n":
            flush()
            i += 1
            for p in pending:
                delim, strip, owner = p
                body: List[str] = []
                while i < n:
                    j = text.find("\n", i)
                    line = text[i:] if j == -1 else text[i:j]
                    i = n if j == -1 else j + 1
                    cmp = line.lstrip("\t") if strip else line
                    if cmp == delim:
                        break
                    body.append(line)
                if owner is not None:
                    owner.heredocs.append((delim, "\n".join(body)))
            pending.clear()
            continue

        if text.startswith(("&&", "||", "|&", ";;"), i):
            flush()
            i += 2
            continue
        if c in ";|&":
            flush()
            i += 1
            continue

        cur.append(c)
        i += 1

    if quote is not None:
        return False
    flush()
    return ok


_SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "mksh", "ash"}
_RESERVED_LEAD = {"if", "then", "else", "elif", "do", "while", "until", "!", "time", "{", "}"}
_RESERVED_ONLY = {"fi", "done", "esac", "}", ")", "then", "else", "do", "{", "in"}

# Wrapper programs: (name, option flags that take an argument)
_WRAPPERS = {
    "sudo": {"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-T", "-U", "--user", "--group", "--chdir"},
    "doas": {"-u", "-C"},
    "env": {"-u", "-C", "-S", "--unset", "--chdir"},
    "nohup": set(),
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p", "-P", "-u", "--class", "--classdata"},
    "time": {"-f", "-o", "--format", "--output"},
    "command": set(),
    "builtin": set(),
    "exec": {"-a"},
    "stdbuf": {"-i", "-o", "-e"},
    "timeout": {"-s", "-k", "--signal", "--kill-after"},
    "unbuffer": set(),
    "xargs": {"-a", "-d", "-E", "-I", "-L", "-n", "-P", "-s", "--arg-file", "--delimiter",
              "--max-args", "--max-procs", "--max-chars", "--replace"},
    "watch": {"-n", "-d", "--interval"},
    "flock": {"-w", "-E", "--timeout"},
    "chrt": set(),
    "taskset": set(),
    "systemd-run": {"-u", "-p", "--unit", "--property", "--user"},
}
_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _strip_lead(words: List[str]) -> List[str]:
    """Drop leading env assignments and reserved words."""
    out = list(words)
    while out and (_ASSIGN_RE.match(out[0]) or out[0] in _RESERVED_LEAD):
        out.pop(0)
    return out


def effective_argv(words: List[str]) -> List[str]:
    """argv of the program that actually runs, wrappers and env stripped.

    ``sudo -u x env FOO=1 nice -n 5 cp a b`` -> ``['cp', 'a', 'b']``.
    """
    w = _strip_lead(words)
    guard = 0
    while w and guard < 10:
        guard += 1
        prog = os.path.basename(w[0])
        if prog not in _WRAPPERS:
            break
        takes_arg = _WRAPPERS[prog]
        w = w[1:]
        if prog == "timeout":
            # options, then the duration
            while w and w[0].startswith("-"):
                opt = w.pop(0)
                if opt in takes_arg and w:
                    w.pop(0)
            if w:
                w.pop(0)
            continue
        while w:
            if w[0] == "--":
                w.pop(0)
                break
            if _ASSIGN_RE.match(w[0]):
                w.pop(0)
                continue
            if w[0].startswith("-"):
                opt = w.pop(0)
                if opt in takes_arg and w:
                    w.pop(0)
                continue
            break
        w = _strip_lead(w)
    if w:
        w = [os.path.basename(w[0])] + w[1:]
    return w


def split_command(command: str) -> Tuple[List[Segment], bool]:
    """Split a command line into simple commands (recursively).

    Returns (segments, parse_ok). On parse failure the whole command is
    returned as a single segment and parse_ok is False.
    """
    command = unicodedata.normalize("NFKC", command or "")
    segments: List[Segment] = []
    ok = _parse_into(command, segments, "top", 0)
    if not ok:
        return [Segment(command.strip(), [], [], "top")], False

    # Expand wrappers: bash -c '...', eval ..., and shells fed by heredoc.
    i = 0
    guard = 0
    while i < len(segments) and guard < 200:
        guard += 1
        seg = segments[i]
        i += 1
        words = seg.words()
        argv = effective_argv(words)
        if not argv:
            continue
        prog = argv[0]
        inner = None
        if prog in _SHELLS:
            for k, a in enumerate(argv[1:], start=1):
                if a == "-c" or (a.startswith("-") and not a.startswith("--") and "c" in a[1:]):
                    if k + 1 < len(argv):
                        inner = argv[k + 1]
                    break
            if inner is None and seg.heredocs:
                inner = "\n".join(body for _, body in seg.heredocs)
                if inner:
                    sub: List[Segment] = []
                    if not _parse_into(inner, sub, "heredoc-shell", 1):
                        ok = False
                    segments.extend(sub)
                inner = None
        elif prog == "eval":
            inner = " ".join(argv[1:])
        elif prog == "ssh":
            # ssh [opts] host [command...] -> the remote command is still a command
            rest = argv[1:]
            j = 0
            ssh_takes_arg = set("bcDEeFIiJLlmOoPpQRSWw")
            while j < len(rest) and rest[j].startswith("-"):
                flag = rest[j]
                j += 1
                if len(flag) == 2 and flag[1] in ssh_takes_arg:
                    j += 1
            if j < len(rest) - 1:
                inner = " ".join(rest[j + 1:])
            elif seg.heredocs:
                inner = "\n".join(body for _, body in seg.heredocs)
        if inner:
            sub = []
            if not _parse_into(inner, sub, "wrapper", 1):
                ok = False
            segments.extend(sub)

    # Drop pure keyword segments (fi, done, ...)
    segments = [s for s in segments if s.collapsed not in _RESERVED_ONLY]
    if not ok:
        return [Segment(command.strip(), [], [], "top")], False
    return segments, True


# =============================================================================
# WRITE DETECTION (used for the tier-1 default)
# =============================================================================

_NON_FILE_TARGETS = {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "-"}

_WRITE_PROGRAMS = {
    "cp", "mv", "rm", "rmdir", "unlink", "shred", "touch", "mkdir", "ln", "link",
    "chmod", "chown", "chgrp", "chattr", "setfacl", "dd", "truncate", "tee", "install",
    "patch", "rsync", "scp", "sftp", "mkfifo", "mknod", "mktemp", "unzip", "zip", "7z",
    "7za", "unrar", "ar", "cpio", "passwd", "chpasswd", "useradd", "userdel", "usermod",
    "groupadd", "groupdel", "mount", "umount", "swapon", "swapoff", "iptables",
    "ip6tables", "nft", "ufw", "kill", "pkill", "killall", "reboot", "shutdown",
    "poweroff", "halt", "nano", "micro", "vim", "vi", "nvim", "emacs", "ed", "ex",
    "nixos-rebuild", "nix-build", "nix-shell", "nix-collect-garbage", "home-manager",
    "flatpak", "apt", "apt-get", "dpkg", "snap", "brew", "inotifywait", "inotifywatch",
    "sqlite3", "psql", "mysql", "redis-cli", "crontab", "at", "batch", "hf",
    "huggingface-cli", "wget",
}
_INTERPRETERS = {
    "python", "python3", "python2", "pypy", "pypy3", "node", "nodejs", "deno", "bun",
    "ruby", "perl", "php", "lua", "luajit", "rscript", "julia", "osascript", "pwsh",
    "powershell",
}
_PKG_MANAGERS = {"pip", "pip3", "pipx", "npm", "pnpm", "yarn", "uv", "cargo", "go", "gem", "conda", "mamba"}
_PKG_READ = {"list", "show", "freeze", "check", "search", "view", "outdated", "info",
             "--version", "-v", "-V", "version", "help", "--help", "-h", "ls", "why", "audit"}
_GIT_READ = {"status", "log", "diff", "show", "ls-files", "ls-tree", "ls-remote", "rev-parse",
             "rev-list", "describe", "blame", "grep", "shortlog", "cat-file", "reflog",
             "whatchanged", "show-ref", "for-each-ref", "name-rev", "help", "version",
             "--version", "count-objects", "check-ignore", "var", "merge-base"}
_GIT_LISTING = {"branch", "tag", "remote", "stash", "config", "worktree", "submodule"}
_SYSTEMCTL_WRITE = {"start", "stop", "restart", "reload", "enable", "disable", "mask", "unmask",
                    "kill", "daemon-reload", "set-property", "edit", "isolate", "reboot",
                    "poweroff", "halt", "suspend", "hibernate", "try-restart",
                    "reload-or-restart", "reset-failed", "link", "revert", "preset",
                    "set-default", "set-environment", "import-environment"}
_CONTAINER_READ = {"ps", "images", "inspect", "logs", "version", "info", "stats", "top", "port",
                   "history", "search", "events", "--version", "help", "--help"}
_NIX_READ = {"eval", "search", "show-config", "path-info", "why-depends", "log", "hash",
             "--version", "help", "--help", "repl"}
_COMPRESSORS = {"gzip", "gunzip", "bzip2", "bunzip2", "xz", "unxz", "zstd", "unzstd", "lz4"}


def _short_flags(argv: List[str]) -> str:
    """Concatenate all single-dash short flag letters (``-rf -x`` -> ``rfx``)."""
    letters = []
    for a in argv[1:]:
        if a.startswith("-") and not a.startswith("--") and len(a) > 1:
            letters.append(a[1:])
    return "".join(letters)


def _first_positional(argv: List[str]) -> str:
    for a in argv[1:]:
        if not a.startswith("-"):
            return a
    return ""


def segment_write_reason(seg: Segment) -> Optional[str]:
    """Why this sub-command counts as a write (None if it looks read-only)."""
    for r in seg.redirects:
        if r.op in (">", ">>", ">|", "&>", "&>>", "<>"):
            tgt = r.target.strip()
            if tgt and tgt not in _NON_FILE_TARGETS and not tgt.startswith("&"):
                return f"writes a file (redirect -> {tgt})"
    argv = effective_argv(seg.words())
    if not argv:
        return None
    prog = argv[0].lower()
    args = argv[1:]
    flags = _short_flags(argv)
    sub = _first_positional(argv)

    if prog in _WRITE_PROGRAMS:
        if prog == "crontab" and "l" in flags and len(args) == 1:
            return None
        if prog == "wget" and ("--spider" in args or "-O-" in args or ("-O" in args and "-" in args)):
            return None
        return f"`{prog}` modifies files or system state"
    if prog in _COMPRESSORS:
        if any(f in flags for f in "cltk") or "--stdout" in args or "--list" in args or "--test" in args:
            return None
        return f"`{prog}` compresses/decompresses in place"
    if prog in _INTERPRETERS or re.match(r"^python\d(\.\d+)?$", prog):
        if args and all(a in ("--version", "-V", "--help", "-h") for a in args):
            return None
        if len(args) >= 2 and args[0] == "-m" and args[1] in ("json.tool", "py_compile", "pydoc"):
            return None
        return f"runs code (`{prog}`)"
    if prog in _SHELLS:
        if "n" in flags:
            return None         # syntax check only
        if "c" in flags or seg.heredocs:
            return None         # the inner commands are analyzed on their own
        if not args or all(a.startswith("-") for a in args):
            # `curl ... | sh` — a shell reading commands from stdin runs
            # whatever arrives on the pipe.
            return f"runs commands from stdin (`{prog}`)"
        return f"runs a script (`{prog}`)"
    if prog == "sed" or prog == "gsed":
        if "i" in flags or any(a.startswith("--in-place") for a in args):
            return "`sed -i` edits a file in place"
        return None
    if prog in ("awk", "gawk", "mawk"):
        if "inplace" in args:
            return "`awk -i inplace` edits a file in place"
        if re.search(r"\bprintf?\b[^;]*>\s*\"", seg.text) or "system(" in seg.text:
            return f"`{prog}` program writes files or runs commands"
        return None
    if prog == "perl":
        if "i" in flags:
            return "`perl -i` edits a file in place"
        return "runs code (`perl`)"
    if prog == "find":
        for a in args:
            if a in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0",
                     "-fprintf", "-fls"):
                return f"`find {a}` modifies files or runs commands"
        return None
    if prog in ("tar", "bsdtar"):
        first = args[0] if args else ""
        modes = first.lstrip("-") if first and not first.startswith("--") else ""
        if (any(m in modes for m in "cxru") or any(a in ("--create", "--extract", "--get",
                                                          "--append", "--update", "--delete")
                                                    for a in args)):
            return "`tar` creates or extracts files"
        return None
    if prog == "curl":
        for a in args:
            if a in ("-o", "-O", "--output", "--remote-name", "--remote-name-all", "-J",
                     "--remote-header-name", "-T", "--upload-file") or (a.startswith("-o") and len(a) > 2):
                return "`curl` writes or uploads a file"
        if "O" in flags or "o" in flags:
            return "`curl` writes a file"
        return None
    if prog == "git":
        if not sub or sub in _GIT_READ:
            return None
        if sub in _GIT_LISTING:
            rest = args[args.index(sub) + 1:] if sub in args else []
            if not rest or all(r in ("-a", "-r", "-v", "-vv", "--list", "-l", "--all", "list",
                                      "show", "--get", "--get-all", "--show-origin", "status")
                               or r.startswith("--get") or r.startswith("--list") for r in rest):
                return None
        return f"`git {sub}` changes the repository"
    if prog == "systemctl":
        if sub in _SYSTEMCTL_WRITE:
            return f"`systemctl {sub}` changes a service"
        return None
    if prog == "journalctl":
        if any(a.startswith("--vacuum") or a in ("--rotate", "--flush", "--sync", "--relinquish-var")
               for a in args):
            return "`journalctl` maintenance changes logs"
        return None
    if prog in ("docker", "podman", "nerdctl"):
        if not sub or sub in _CONTAINER_READ:
            return None
        if sub in ("container", "image", "volume", "network", "system", "pod") and len(args) >= 2:
            if args[args.index(sub) + 1] in ("ls", "list", "inspect", "df", "info", "ps"):
                return None
        return f"`{prog} {sub}` changes containers/images"
    if prog == "nix":
        if not sub or sub in _NIX_READ:
            return None
        if sub == "flake" and len(args) >= 2 and args[args.index(sub) + 1] in ("show", "metadata", "info", "check"):
            return None
        if sub == "store" and len(args) >= 2 and args[args.index(sub) + 1] in ("ls", "cat", "ping", "info"):
            return None
        return f"`nix {sub}` builds or changes the store/profile"
    if prog in ("nix-env",):
        if "q" in flags or "--query" in args:
            return None
        return "`nix-env` changes a profile"
    if prog in _PKG_MANAGERS:
        if not sub or sub in _PKG_READ:
            return None
        return f"`{prog} {sub}` installs or changes packages"
    if prog in ("sysctl",) and ("w" in flags or any("=" in a for a in args)):
        return "`sysctl` changes kernel settings"
    if prog in ("hostnamectl", "timedatectl", "localectl") and sub.startswith("set"):
        return f"`{prog} {sub}` changes system settings"
    if prog == "ip" and any(a in ("add", "del", "delete", "set", "flush", "change", "replace") for a in args):
        return "`ip` changes network configuration"
    return None


# =============================================================================
# POLICY
# =============================================================================

def _is_regexish(pattern: str) -> bool:
    return any(ch in _REGEX_META for ch in pattern)


def _compile_rule(kind: str, entry: Any, match_mode: str) -> List[Rule]:
    """Turn one policy entry (string or {pattern: ...} object) into rules."""
    case_sensitive = False
    note = ""
    force_regex: Optional[bool] = None
    if isinstance(entry, dict):
        raw = entry.get("pattern", "")
        case_sensitive = bool(entry.get("case_sensitive", False))
        note = str(entry.get("note", "") or "")
        if "regex" in entry:
            force_regex = bool(entry.get("regex"))
        patterns = raw if isinstance(raw, list) else [raw]
    else:
        patterns = [entry]

    rules = []
    for p in patterns:
        if not isinstance(p, str) or not p.strip():
            continue
        src = p
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            if match_mode == MATCH_ZED:
                rx = re.compile(src, flags)
            else:
                regexish = _is_regexish(src) if force_regex is None else force_regex
                body = src if regexish else re.escape(src)
                if kind == ALLOW:
                    rx = re.compile(r"^\s*(?:" + body + r")" + _RIGHT_BOUNDARY, flags)
                else:
                    rx = re.compile(_LEFT_BOUNDARY + r"(?:" + body + r")", flags)
            rules.append(Rule(kind, src, rx, note))
        except re.error as e:
            rules.append(Rule(kind, src, None, note, error=str(e)))
    return rules


def _find_terminal_block(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Locate the terminal permissions block in any of the accepted shapes.

    Returns (terminal_block, global_block_for_defaults).
    """
    if not isinstance(data, dict):
        return {}, {}
    candidates = []
    agent = data.get("agent") if isinstance(data.get("agent"), dict) else None
    tp = None
    if agent and isinstance(agent.get("tool_permissions"), dict):
        tp = agent["tool_permissions"]
    elif isinstance(data.get("tool_permissions"), dict):
        tp = data["tool_permissions"]
    if tp is not None:
        tools = tp.get("tools") if isinstance(tp.get("tools"), dict) else tp
        if isinstance(tools.get("terminal"), dict):
            return tools["terminal"], tp
    if isinstance(data.get("tools"), dict) and isinstance(data["tools"].get("terminal"), dict):
        return data["tools"]["terminal"], data
    if isinstance(data.get("terminal"), dict):
        return data["terminal"], data
    for key in ("always_allow", "always_confirm", "always_deny"):
        if key in data:
            candidates.append(key)
    if candidates:
        return data, data
    return {}, data


@dataclass
class Policy:
    deny: List[Rule] = field(default_factory=list)
    confirm: List[Rule] = field(default_factory=list)
    allow: List[Rule] = field(default_factory=list)
    default: str = DEFAULT_TIER
    match: str = MATCH_WORD
    source: str = "builtin"
    errors: List[str] = field(default_factory=list)
    mtime: float = 0.0

    @property
    def invalid_rules(self) -> List[Rule]:
        return [r for r in (self.deny + self.confirm + self.allow) if r.regex is None]

    def summary_lines(self) -> List[str]:
        def fmt(rules):
            return ", ".join(r.source for r in rules) if rules else "(none)"
        return [
            f"always denied: {fmt(self.deny)}",
            f"always need operator approval: {fmt(self.confirm)}",
            f"always allowed: {fmt(self.allow)}",
        ]


def parse_policy(data: Dict[str, Any], source: str = "inline") -> Policy:
    term, glob = _find_terminal_block(data)
    match_mode = str(term.get("match", glob.get("match", MATCH_WORD)) or MATCH_WORD).lower()
    if match_mode not in (MATCH_WORD, MATCH_ZED):
        match_mode = MATCH_WORD
    default = str(term.get("default", glob.get("default", DEFAULT_TIER)) or DEFAULT_TIER).lower()
    if default == "ask":
        default = CONFIRM
    if default not in (ALLOW, CONFIRM, DENY, DEFAULT_TIER):
        default = DEFAULT_TIER

    pol = Policy(default=default, match=match_mode, source=source)
    for key, kind, bucket in (("always_deny", DENY, pol.deny),
                              ("always_confirm", CONFIRM, pol.confirm),
                              ("always_ask", CONFIRM, pol.confirm),
                              ("always_allow", ALLOW, pol.allow)):
        entries = term.get(key, [])
        if isinstance(entries, (str, dict)):
            entries = [entries]
        for entry in entries or []:
            bucket.extend(_compile_rule(kind, entry, match_mode))
    for r in pol.invalid_rules:
        pol.errors.append(f"{r.kind} pattern {r.source!r}: {r.error}")
    return pol


BUILTIN_POLICY = {
    "terminal": {
        "default": "tier",
        "always_deny": ["rm", "rmdir", "unlink", "shred"],
        "always_confirm": ["sudo", "doas", "pkexec", r"su\s+(-c|root)"],
        "always_allow": [],
    }
}


def _policy_candidates() -> List[Path]:
    env = os.environ.get("OGNEE_SHELL_POLICY", "").strip()
    out = []
    if env:
        out.append(Path(os.path.expanduser(env)))
    here = Path(__file__).resolve().parent
    out.append(here / "shell-policy.json")
    out.append(here.parent / "shell-policy.json")
    return out


class PolicyStore:
    """Loads the policy file and hot-reloads it when its mtime changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._policy: Optional[Policy] = None
        self._path: Optional[Path] = None
        self._load_error: str = ""

    def _resolve(self) -> Optional[Path]:
        for p in _policy_candidates():
            if p.is_file():
                return p
        return None

    def get(self) -> Policy:
        with self._lock:
            path = self._resolve()
            if path is None:
                if self._policy is None or self._policy.source != "builtin":
                    self._policy = parse_policy(BUILTIN_POLICY, "builtin")
                    self._path = None
                return self._policy
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            if (self._policy is None or self._path != path or self._policy.mtime != mtime):
                self._path = path
                try:
                    data = json.loads(path.read_text())
                    pol = parse_policy(data, str(path))
                    self._load_error = ""
                except (OSError, ValueError) as e:
                    # Unreadable/invalid JSON: fail closed, loudly.
                    pol = parse_policy(BUILTIN_POLICY, str(path))
                    pol.errors.append(f"could not load {path}: {e}")
                    self._load_error = str(e)
                pol.mtime = mtime
                self._policy = pol
            return self._policy

    def reload(self) -> Policy:
        with self._lock:
            self._policy = None
        return self.get()


_store = PolicyStore()


def get_policy() -> Policy:
    return _store.get()


def reload_policy() -> Policy:
    return _store.reload()


# =============================================================================
# EVALUATION
# =============================================================================

_RECURSIVE_RM_TARGETS = {"/", "/*", "~", "~/", "~/*", "$HOME", "${HOME}", "$HOME/", "${HOME}/",
                         "$HOME/*", ".", "./", "./*", "..", "../", "*", "/.", "/.."}


def _builtin_hard_deny(seg: Segment) -> Optional[str]:
    argv = effective_argv(seg.words())
    if not argv or argv[0] not in ("rm",):
        return None
    flags = _short_flags(argv)
    recursive = "r" in flags.lower() or "--recursive" in argv
    if not recursive:
        return None
    home = os.path.expanduser("~")
    for a in argv[1:]:
        if a.startswith("-"):
            continue
        norm = a.rstrip("/") or "/"
        if a in _RECURSIVE_RM_TARGETS or norm in ("/", home, ".", ".."):
            return f"recursive delete of `{a}`"
    return None


def _match_any(rules: List[Rule], text: str) -> Optional[Rule]:
    for r in rules:
        if r.regex is not None and r.regex.search(text):
            return r
    return None


def _allow_text(seg: Segment) -> str:
    """Sub-command text used for allow matching (env/keywords stripped,
    program path reduced to its basename)."""
    words = seg.words()
    raw = seg.collapsed
    lead = _strip_lead(words)
    if not lead:
        return raw
    dropped = len(words) - len(lead)
    # rebuild text from the raw string by skipping the leading tokens
    text = raw
    for _ in range(dropped):
        parts = text.split(" ", 1)
        text = parts[1] if len(parts) > 1 else ""
    first, _, rest = text.partition(" ")
    if "/" in first and not first.startswith("$"):
        first = os.path.basename(first)
    return (first + (" " + rest if rest else "")).strip()


def evaluate(command: str, tier_level: int = 1, policy: Optional[Policy] = None) -> PolicyDecision:
    """Decide allow / confirm / deny for a shell command."""
    pol = policy or get_policy()
    segments, parse_ok = split_command(command)
    dec = PolicyDecision(result=ALLOW, reason="", segments=segments, parse_ok=parse_ok,
                         source=pol.source)

    # 1. built-in hard rules
    for seg in segments:
        why = _builtin_hard_deny(seg)
        if why:
            dec.result = DENY
            dec.reason = f"built-in rule: {why} is never allowed"
            dec.triggers.append(Trigger("builtin", why, seg.collapsed))
            return dec

    # 2. invalid policy => fail closed
    if pol.errors:
        dec.result = DENY
        dec.reason = ("shell policy has errors, so every shell command is blocked until it's "
                      "fixed: " + "; ".join(pol.errors[:3]))
        dec.triggers.append(Trigger("invalid", "; ".join(pol.errors), command))
        return dec

    if not segments:
        dec.result = ALLOW
        dec.reason = "empty command"
        return dec

    # 3/4/5. deny-any, confirm-any, allow-all
    confirm_hits: List[Tuple[Rule, Segment]] = []
    all_allowed = parse_ok and pol.allow != []
    for seg in segments:
        text = seg.collapsed
        hit = _match_any(pol.deny, text)
        if hit:
            dec.result = DENY
            dec.reason = f"blocked by shell policy: `{hit.source}` is in always_deny (matched `{_clip(text)}`)"
            dec.triggers.append(Trigger(DENY, hit.source, text, hit.note))
            return dec
        chit = _match_any(pol.confirm, text)
        if chit:
            confirm_hits.append((chit, seg))
        if all_allowed:
            if pol.match == MATCH_ZED:
                ahit = _match_any(pol.allow, text)
            else:
                ahit = _match_any(pol.allow, _allow_text(seg))
                if ahit and segment_write_reason(seg) and any(
                        r.op in (">", ">>", ">|", "&>", "&>>", "<>") and r.target not in _NON_FILE_TARGETS
                        and not r.target.startswith("&") for r in seg.redirects):
                    ahit = None     # `cat x > file` is not "just cat"
            if not ahit:
                all_allowed = False

    writes = []
    for seg in segments:
        why = segment_write_reason(seg)
        if why:
            writes.append((seg.collapsed, why))
    dec.writes = writes

    if confirm_hits:
        dec.result = CONFIRM
        seen = []
        for rule, seg in confirm_hits:
            if rule.source not in seen:
                seen.append(rule.source)
                dec.exception_keys.append((f"shell:rule:{rule.source}", f"rule `{rule.source}`"))
            dec.triggers.append(Trigger(CONFIRM, rule.source, seg.collapsed, rule.note))
        dec.reason = "needs approval: " + ", ".join(f"`{s}` is in always_confirm" for s in seen)
        return dec

    if all_allowed:
        dec.result = ALLOW
        dec.reason = "every sub-command is in always_allow"
        return dec

    # 6. default
    mode = pol.default
    if not parse_ok and mode == ALLOW:
        mode = CONFIRM          # can't parse it -> never auto-allow it
    if mode == DEFAULT_TIER:
        if tier_level <= 0:
            mode = CONFIRM
            reason = "tier 0 (probation): every command not in always_allow needs approval"
        elif tier_level == 1:
            if writes or not parse_ok:
                mode = CONFIRM
                if not parse_ok:
                    reason = "couldn't parse this command safely (tier 1 asks)"
                else:
                    reason = "tier 1: " + "; ".join(w for _, w in writes[:3])
            else:
                mode = ALLOW
                reason = "tier 1: read-only command"
        else:
            mode = ALLOW
            reason = f"tier {tier_level}: commands not covered by the policy are allowed"
    else:
        reason = f"not covered by the policy; default is `{mode}`"

    dec.result = mode
    dec.reason = reason
    if mode == CONFIRM:
        progs = []
        target_segs = [s for s in segments if segment_write_reason(s)] or segments
        for seg in target_segs:
            argv = effective_argv(seg.words())
            prog = argv[0] if argv else seg.collapsed.split(" ")[0]
            if prog and prog not in progs:
                progs.append(prog)
        for prog in progs:
            dec.exception_keys.append((f"shell:cmd:{prog}", f"`{prog}` commands"))
            dec.triggers.append(Trigger("default", prog, ""))
    elif mode == DENY:
        dec.triggers.append(Trigger("default", "deny", ""))
    return dec


def _clip(text: str, n: int = 80) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"
