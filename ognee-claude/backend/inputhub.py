#!/usr/bin/env python3
"""
inputhub.py - the operator's keyboard

One thread owns the terminal: the main thread. The agent runs in a worker
thread, so you can type while it works — /p, /pf, /tier 3, or a plain
message that gets delivered at the next turn boundary.

What lives here:

* Line editing. The old fallback called input() without ever importing
  readline, so arrow keys came through as raw escape codes (the ``^[[D``
  spam). This module imports readline (or uses prompt_toolkit when you opt in
  with OGNEE_INPUT=ptk), so left/right, Home/End, Ctrl-A/E/W/U, up-arrow
  history and Tab completion of slash commands all work.

* Output that doesn't trample your typing. While you're at the prompt, any
  output from the agent thread is printed ABOVE your half-typed line and the
  line is redrawn with the cursor where you left it.

* Questions. When a gate needs an answer, the agent thread posts a question
  here and waits; your next line answers it. Slash commands typed at a gate
  prompt are executed on the main thread and reported back to the gate.

Backends (OGNEE_INPUT env var):
    readline (default) | ptk (prompt_toolkit, opt-in) | basic (no editing)
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

BACKEND_READLINE = "readline"
BACKEND_PTK = "prompt_toolkit"
BACKEND_BASIC = "basic"

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
_SGR_RE = re.compile(r"(\x1b\[[0-9;]*m)")

# Short answers at a gate prompt that shouldn't clutter up-arrow history.
_GATE_ANSWER_RE = re.compile(r"^\s*([1-7]|[vVyYnNaAdDoOsS?hH]|yes|no|view|help)\s*$")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def _wrap_readline_escapes(prompt: str) -> str:
    """Wrap ANSI color codes in readline's \\001/\\002 'zero width' markers."""
    return _SGR_RE.sub(lambda m: "\x01" + m.group(1) + "\x02", prompt)


def _display_width(text: str) -> int:
    try:
        from wcwidth import wcswidth
        w = wcswidth(text)
        return w if w >= 0 else len(text)
    except Exception:
        return len(text)


@dataclass
class Answer:
    """What the operator typed in reply to a question."""
    kind: str                 # "line" | "command" | "abort" | "eof"
    text: str = ""
    command: str = ""         # canonical command name when kind == "command"


class _Question:
    __slots__ = ("prompt", "kind", "event", "answer")

    def __init__(self, prompt: str, kind: str):
        self.prompt = prompt
        self.kind = kind
        self.event = threading.Event()
        self.answer: Optional[Answer] = None


# =============================================================================
# OUTPUT PROXY (readline / basic backends)
# =============================================================================

class _ProxyStream:
    """sys.stdout / sys.stderr stand-in.

    Writes from the main thread, or when nobody is at the prompt, go straight
    through. Writes from other threads while the operator is typing are
    buffered per line and printed above the prompt, then the prompt and the
    half-typed line are redrawn.

    fileno()/isatty() delegate to the real stream so builtin input() still
    recognises the terminal and keeps using GNU readline.
    """

    def __init__(self, real, hub: "InputHub"):
        self._real = real
        self._hub = hub
        self._buf = ""

    # --- the parts input() and print() rely on ---
    def write(self, s):
        if not s:
            return 0
        hub = self._hub
        if not hub._reading or threading.get_ident() == hub._main_ident:
            with hub._out_lock:
                if self._buf:
                    pending, self._buf = self._buf, ""
                    self._real.write(pending)
                return self._real.write(s)
        with hub._out_lock:
            self._buf += s
            if "\n" not in self._buf:
                return len(s)
            head, _, tail = self._buf.rpartition("\n")
            self._buf = tail
            hub._emit_above_prompt(self._real, head + "\n")
        return len(s)

    def flush(self):
        hub = self._hub
        with hub._out_lock:
            if self._buf and hub._reading and threading.get_ident() != hub._main_ident:
                pending, self._buf = self._buf, ""
                hub._emit_above_prompt(self._real, pending + "\n")
            elif self._buf:
                pending, self._buf = self._buf, ""
                self._real.write(pending)
            try:
                self._real.flush()
            except Exception:
                pass

    def fileno(self):
        return self._real.fileno()

    def isatty(self):
        return self._real.isatty()

    def writable(self):
        return True

    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._real, "errors", "strict")

    def __getattr__(self, name):
        return getattr(self._real, name)


# =============================================================================
# THE HUB
# =============================================================================

class InputHub:
    def __init__(self):
        self.backend = BACKEND_BASIC
        self.threaded = False
        self.note = ""
        self._main_ident = threading.main_thread().ident
        self._question: Optional[_Question] = None
        self._qlock = threading.Lock()
        self._out_lock = threading.RLock()
        self._reading = False
        self._shown_prompt = ""          # prompt readline currently displays
        self._prompt_provider: Callable[[], str] = lambda: "› "
        self.commands_provider: Callable[[], List[str]] = lambda: []
        self.args_provider: Callable[[str], List[str]] = lambda cmd: []
        # Executes a slash command on the main thread and returns its
        # canonical name. Set by the interactive session.
        self.command_executor: Optional[Callable[[str], str]] = None
        self._rl = None
        self._rl_lib = None
        self._rl_fns: Dict[str, object] = {}
        self._ptk_session = None
        self._ptk_patch = None
        self._ptk_ansi = None
        self._orig_stdout = None
        self._orig_stderr = None
        self.history_path: Optional[Path] = None
        self._started = False
        self._last_resolved = 0.0

    # ------------------------------------------------------------------
    # setup / teardown
    # ------------------------------------------------------------------

    def select_backend(self) -> str:
        pref = os.environ.get("OGNEE_INPUT", "").strip().lower()
        try:
            tty = sys.stdin.isatty() and sys.stdout.isatty()
        except Exception:
            tty = False
        if not tty:
            self.note = "stdin/stdout is not a terminal — line editing off"
            return BACKEND_BASIC
        if pref in ("basic", "plain", "none"):
            self.note = "OGNEE_INPUT=basic — line editing off"
            return BACKEND_BASIC
        if pref in ("ptk", "prompt_toolkit", "prompt-toolkit"):
            try:
                import prompt_toolkit  # noqa: F401
                return BACKEND_PTK
            except ImportError:
                self.note = "OGNEE_INPUT=ptk but prompt_toolkit isn't installed — using readline"
        try:
            import readline  # noqa: F401
            return BACKEND_READLINE
        except ImportError:
            self.note = ("Python's readline module is unavailable — arrow keys won't edit. "
                         "Fix: install prompt_toolkit and set OGNEE_INPUT=ptk")
            return BACKEND_BASIC

    def start(self, history_path: Optional[Path] = None,
              prompt_provider: Optional[Callable[[], str]] = None,
              threaded: bool = True) -> str:
        """Pick a backend, load history, install the output proxy."""
        if self._started:
            return self.backend
        self._started = True
        self.history_path = Path(history_path) if history_path else None
        if prompt_provider is not None:
            self._prompt_provider = prompt_provider
        self.backend = self.select_backend()
        if self.backend == BACKEND_PTK:
            try:
                self._start_ptk()
            except Exception as e:     # pragma: no cover - depends on environment
                self.note = f"prompt_toolkit failed to start ({e}) — using readline"
                self.backend = BACKEND_READLINE
        if self.backend == BACKEND_READLINE:
            self._start_readline()
        try:
            out_tty = sys.stdout.isatty()
        except Exception:
            out_tty = False
        if self.backend in (BACKEND_READLINE, BACKEND_BASIC) and threaded and out_tty:
            self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
            sys.stdout = _ProxyStream(sys.stdout, self)
            sys.stderr = _ProxyStream(sys.stderr, self)
        self.threaded = threaded
        return self.backend

    def stop(self):
        if not self._started:
            return
        self._started = False
        self.threaded = False
        if self._ptk_patch is not None:
            try:
                self._ptk_patch.__exit__(None, None, None)
            except Exception:
                pass
            self._ptk_patch = None
        if self._orig_stdout is not None:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            sys.stdout, sys.stderr = self._orig_stdout, self._orig_stderr
            self._orig_stdout = self._orig_stderr = None

    def describe(self) -> str:
        if self.backend == BACKEND_READLINE:
            lib = "GNU readline" if self._rl_lib is not None else "readline"
            redraw = "full redraw" if "rl_forced_update_display" in self._rl_fns else "basic redraw"
            return f"{lib} ({redraw}) — arrows, history, Tab completion"
        if self.backend == BACKEND_PTK:
            return "prompt_toolkit — arrows, history, Tab completion"
        return "basic input() — no line editing" + (f" ({self.note})" if self.note else "")

    # --- readline backend ---
    def _start_readline(self):
        import readline
        self._rl = readline
        doc = getattr(readline, "__doc__", "") or ""
        if "libedit" in doc:
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
        readline.set_completer_delims(" \t\n")
        readline.set_completer(self._rl_complete)
        try:
            readline.set_history_length(5000)
        except Exception:
            pass
        if self.history_path and self.history_path.exists():
            try:
                readline.read_history_file(str(self.history_path))
                # rewrite once to apply the length cap (appends never truncate)
                readline.write_history_file(str(self.history_path))
            except Exception:
                pass
        # ctypes hooks for clean async redraw (GNU readline only)
        try:
            lib = ctypes.CDLL(readline.__file__)
            fns = {}
            for name in ("rl_forced_update_display", "rl_on_new_line"):
                fn = getattr(lib, name, None)
                if fn is not None:
                    fn.restype = ctypes.c_int
                    fn.argtypes = []
                    fns[name] = fn
            setp = getattr(lib, "rl_set_prompt", None)
            if setp is not None:
                setp.restype = ctypes.c_int
                setp.argtypes = [ctypes.c_char_p]
                fns["rl_set_prompt"] = setp
            try:
                fns["rl_point"] = ctypes.c_int.in_dll(lib, "rl_point")
            except ValueError:
                pass
            self._rl_lib = lib
            self._rl_fns = fns
        except Exception:
            self._rl_lib = None
            self._rl_fns = {}

    def _rl_complete(self, text, state):
        try:
            buf = self._rl.get_line_buffer()
        except Exception:
            buf = text
        options = self._completions(buf, text)
        return options[state] if state < len(options) else None

    def _completions(self, buf: str, text: str) -> List[str]:
        if not buf.lstrip().startswith("/"):
            return []
        head = buf.lstrip()
        if " " not in head:
            cands = sorted(set(self.commands_provider()))
            return [c + " " for c in cands if c.startswith(text)]
        cmd = head.split(" ", 1)[0][1:]
        try:
            args = self.args_provider(cmd) or []
        except Exception:
            args = []
        return [a + " " for a in sorted(set(args)) if a.startswith(text)]

    # --- prompt_toolkit backend (opt-in) ---
    def _start_ptk(self):  # pragma: no cover - not available in the build sandbox
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.formatted_text import ANSI
        from prompt_toolkit.history import FileHistory, InMemoryHistory
        from prompt_toolkit.patch_stdout import patch_stdout

        hub = self

        class _SlashCompleter(Completer):
            def get_completions(self, document, complete_event):
                buf = document.text_before_cursor
                word = document.get_word_before_cursor(WORD=True)
                for c in hub._completions(buf, word):
                    yield Completion(c.rstrip(), start_position=-len(word))

        history = InMemoryHistory()
        if self.history_path:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)

            class _Filtered(FileHistory):
                def append_string(self, string):
                    if _GATE_ANSWER_RE.match(string or ""):
                        return
                    super().append_string(string)

            history = _Filtered(str(self.history_path))
        self._ptk_ansi = ANSI
        self._ptk_session = PromptSession(history=history, completer=_SlashCompleter(),
                                          complete_while_typing=False)
        self._ptk_patch = patch_stdout(raw=True)
        self._ptk_patch.__enter__()

    # ------------------------------------------------------------------
    # prompt text
    # ------------------------------------------------------------------

    def set_prompt_provider(self, fn: Callable[[], str]):
        self._prompt_provider = fn

    def current_prompt(self) -> str:
        q = self._question
        if q is not None:
            return q.prompt
        try:
            return self._prompt_provider()
        except Exception:
            return "› "

    def refresh_prompt(self):
        """Redraw the prompt if it changed (callable from any thread)."""
        if not self._reading:
            return
        if self.backend == BACKEND_PTK:  # pragma: no cover
            try:
                app = self._ptk_session.app
                app.invalidate()
            except Exception:
                pass
            return
        if self.backend != BACKEND_READLINE:
            return
        new = self.current_prompt()
        if new == self._shown_prompt:
            return
        setp = self._rl_fns.get("rl_set_prompt")
        redraw = self._rl_fns.get("rl_forced_update_display")
        if setp is None or redraw is None:
            return
        with self._out_lock:
            try:
                real = self._orig_stdout or sys.__stdout__
                self._clear_input_area(real)
                real.flush()
                setp(_wrap_readline_escapes(new).encode("utf-8", "replace"))
                self._shown_prompt = new
                redraw()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # output above the prompt
    # ------------------------------------------------------------------

    def _clear_input_area(self, real):
        """Move to the first row of the prompt+input and clear to end of screen."""
        try:
            cols = max(1, shutil.get_terminal_size().columns)
        except Exception:
            cols = 80
        rows_up = 0
        try:
            buf = self._rl.get_line_buffer() if self._rl else ""
            point_obj = self._rl_fns.get("rl_point")
            point = point_obj.value if point_obj is not None else len(buf)
            prompt_w = _display_width(strip_ansi(self._shown_prompt))
            rows_up = (prompt_w + _display_width(buf[:point])) // cols
        except Exception:
            rows_up = 0
        seq = (f"\x1b[{rows_up}A" if rows_up > 0 else "") + "\r\x1b[J"
        real.write(seq)

    def _emit_above_prompt(self, real, text: str):
        """Print *text* (ending in newline) above the prompt, then redraw it.
        Caller holds _out_lock."""
        try:
            if self.backend == BACKEND_READLINE and "rl_forced_update_display" in self._rl_fns:
                self._clear_input_area(real)
                real.write(text)
                real.flush()
                self._rl_fns["rl_forced_update_display"]()
                return
            buf = ""
            if self.backend == BACKEND_READLINE and self._rl is not None:
                try:
                    buf = self._rl.get_line_buffer()
                except Exception:
                    buf = ""
            real.write("\r\x1b[2K" + text + self._shown_prompt + buf)
            real.flush()
        except Exception:
            try:
                real.write(text)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------

    def on_main_thread(self) -> bool:
        return threading.get_ident() == self._main_ident

    def read_line(self, prompt: Optional[str] = None) -> str:
        """Read one line on the main thread. Raises EOFError / KeyboardInterrupt."""
        text = prompt if prompt is not None else self.current_prompt()
        self._shown_prompt = text
        self._reading = True
        try:
            if self.backend == BACKEND_PTK:  # pragma: no cover
                ansi = self._ptk_ansi
                if prompt is not None:
                    return self._ptk_session.prompt(ansi(text))
                return self._ptk_session.prompt(lambda: ansi(self.current_prompt()))
            if self.backend == BACKEND_READLINE:
                before = self._history_len()
                line = input(_wrap_readline_escapes(text))
                if self._history_len() > before:   # readline added it (it skips repeats itself)
                    self._save_history_line(line)
                return line
            return input(text)
        finally:
            self._reading = False

    def _history_len(self) -> int:
        try:
            return self._rl.get_current_history_length()
        except Exception:
            return 0

    def _save_history_line(self, line: str):
        """Append each entered line right away, so a crash or kill doesn't
        lose the session's history. Short gate answers are skipped."""
        if not self.history_path or not line.strip() or _GATE_ANSWER_RE.match(line):
            return
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.history_path.exists():
                self.history_path.touch()
            self._rl.append_history_file(1, str(self.history_path))
        except Exception:
            pass

    def forget_last_history_entry(self, line: str):
        """Drop a short gate answer ('1', 'n', ...) from up-arrow history."""
        if self.backend != BACKEND_READLINE or self._rl is None:
            return
        if not _GATE_ANSWER_RE.match(line or ""):
            return
        try:
            n = self._rl.get_current_history_length()
            if n > 0 and self._rl.get_history_item(n) == line:
                self._rl.remove_history_item(n - 1)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # questions (gates and y/n prompts)
    # ------------------------------------------------------------------

    def pending(self) -> Optional[_Question]:
        return self._question

    def ask(self, prompt: str, kind: str = "gate") -> Answer:
        """Ask the operator something. Safe from any thread.

        Main thread (or non-threaded mode): reads the terminal directly.
        Worker thread: posts the question and blocks until the main thread
        answers it (or it's cancelled by /pf, Ctrl-C, or shutdown).
        """
        if not self.threaded or self.on_main_thread():
            try:
                line = self.read_line(prompt)
            except EOFError:
                return Answer("eof")
            except KeyboardInterrupt:
                print()
                return Answer("abort")
            self.forget_last_history_entry(line)
            stripped = line.strip()
            if stripped.startswith("/") and self.command_executor is not None:
                name = self.command_executor(stripped)
                return Answer("command", stripped, name or "")
            return Answer("line", line)

        q = _Question(prompt, kind)
        while True:
            with self._qlock:
                if self._question is None:
                    self._question = q
                    break
            if q.event.wait(0.05):
                break
        self.refresh_prompt()
        q.event.wait()
        with self._qlock:
            if self._question is q:
                self._question = None
        self.refresh_prompt()
        return q.answer or Answer("abort")

    def _resolve(self, answer: Answer) -> bool:
        with self._qlock:
            q = self._question
            if q is None or q.answer is not None:
                return False
            q.answer = answer
            self._question = None
            self._last_resolved = time.monotonic()
        q.event.set()
        return True

    def recently_resolved(self, seconds: float = 3.0) -> bool:
        """True if a question was resolved in the last *seconds* (used to drop
        a late '3' typed after /pf already cancelled the approval)."""
        return (time.monotonic() - self._last_resolved) < seconds

    @staticmethod
    def looks_like_gate_answer(line: str) -> bool:
        return bool(_GATE_ANSWER_RE.match(line or ""))

    def deliver(self, line: str) -> bool:
        """Main thread: hand a typed line to the pending question."""
        q = self._question
        if q is None:
            return False
        self.forget_last_history_entry(line)
        stripped = line.strip()
        if stripped.startswith("/") and self.command_executor is not None:
            name = self.command_executor(stripped)
            return self._resolve(Answer("command", stripped, name or ""))
        return self._resolve(Answer("line", line))

    def cancel_pending(self, kind: str = "abort", text: str = "") -> bool:
        """Resolve any pending question without operator input."""
        return self._resolve(Answer(kind, text))


_hub = InputHub()


def get_hub() -> InputHub:
    return _hub
