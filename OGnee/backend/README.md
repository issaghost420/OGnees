# OGnee

OGnee is a local-first autonomous agent framework that runs against an
OpenAI-compatible LLM endpoint (LM Studio, llama.cpp, vLLM, …) and drives
tools through the Model Context Protocol (MCP) — with a trust-tiered
permission gate system in front of every side effect.

## Install

Requires Python 3.10+.

```sh
# From a clone (the repo directory IS the "backend" package)
pip install -r requirements.txt
python ognee.py --help

# Or install as a package with a console entry point
pip install .
ognee --help
```

### Nix / NixOS notes

There is no flake yet; use a shell or venv:

```sh
nix-shell -p python312 python312Packages.pip
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # pinned versions
python ognee.py
```

`wcwidth` and `requests` are the only required runtime deps (`prompt_toolkit`
is optional, used with `OGNEE_INPUT=ptk`; `tiktoken` optional for token
counts) and are all available in nixpkgs if you prefer a `buildPythonApplication` wrapper.

## Configuration

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `LM_STUDIO_URL` | `http://localhost:1234/v1/chat/completions` | Chat-completions endpoint |
| `LM_STUDIO_TOKEN` | `lm-studio` | API token |
| `MCP_CONFIG` | `backend/mcp.json` (falls back to repo `mcp.json`) | MCP server config |
| `MCP_MODE` | `stdio` | `stdio` (spawn servers) or `proxy` (attach to an mcpo-style HTTP proxy) |
| `OGNEE_TOOL_TIMEOUT` | `15` (seconds) | Global tool timeout override |
| `OGNEE_TOOL_TIMEOUT_<TOOL>` | — | Per-tool/per-server timeout, e.g. `OGNEE_TOOL_TIMEOUT_DEEP_RESEARCH=300` |
| `OGNEE_SPEAK` | off | Opt-in: enable `local/speak` TTS |
| `OGNEE_SPEAK_URL` | `https://dojos.duckdns.org/chatterbox/api/tts` | TTS endpoint for `local/speak` |
| `OGNEE_INPUT` | `readline` | Line editor: `readline`, `ptk` (prompt_toolkit), or `basic` |
| `OGNEE_SHELL_POLICY` | `backend/shell-policy.json` | Shell policy file to use instead |
| `OGNEE_TEMPERATURE` | `0.7` | Sampling temperature (above 0 so `/regenerate` varies) |
| `OGNEE_MAX_TOKENS` | `8192` | Completion cap per LLM call |

`local/speak` POSTs assistant text to an external TTS endpoint. It is
**disabled by default**; enable it explicitly with `/speak-on` or
`OGNEE_SPEAK=1`, and it is gated as a network request like any web tool.

## Trust tiers

Every tool call passes through `gates.gate_tool_call`, which classifies tools
by capability (file read / file write / move / shell / network) and enforces:

- `EXCLUDED_PATHS` — never model-readable (Documents, Downloads, `~/.ssh`,
  `~/.gnupg`, `~/.aws`, `~/.azure`, `~/.kube`, `~/.config`, `~/.secrets`, …)
- `READ_ONLY_PATHS` / `WRITABLE_PATHS` ACLs; writes outside writable areas are
  redirected to a shadow staging tree for review.
- Trust tiers: **probation (0) → supervised (1) → trusted (2) → autonomous (3)**
  (`config.TIERS`). Higher tiers unlock direct writes, network egress, and SSH.
- Shell commands (`execute_command`) are decided by `shell-policy.json`, a
  Zed-style `always_deny` / `always_confirm` / `always_allow` list checked per
  sub-command (`&&`, `;`, `|`, `$(...)` are split). Deny wins over confirm,
  confirm over allow; anything unlisted falls back to the trust tier. Edit it
  live — it's hot-reloaded (`/policy` shows, tests and reloads it).
- Approval prompts are a **1-7 menu**: allow, allow + note, deny, deny +
  reason, say something instead, allow until the task ends, allow for the
  session. Whatever you type there reaches the model. They **fail closed**:
  Enter re-prompts, EOF or Ctrl-C denies, and text that isn't a choice is
  sent as a message — never as an allow.

## Interactive mode

The keyboard stays live while OGneey works:

- `/p` pauses at the next turn boundary, `/pf` (or Ctrl-C) stops right now —
  it cancels the LLM request and running MCP calls. `/r` resumes.
- Plain text typed mid-turn is delivered at the next turn boundary.
- `/tier N`, `/disable`, `/tools-off` and friends work mid-turn, even at an
  approval prompt (the pending call is re-checked under the new settings).
- **`/regenerate [N]`** (`/regen`, `/retry`, `/redo`) redoes a reply. Every
  OGneey reply is printed with its number (`#3 · ↻ /regen 3`), so any reply
  can be redone, not just the latest. With no number it redoes the latest
  reply instantly. In plan mode it redoes the whole latest plan turn (the
  lookups plus the summary) and discards the steps that turn queued, so
  `/execute` never runs the old plan next to the new one. Redoing an earlier
  reply rewinds the conversation to just before it — the replies and messages
  after it leave the model's context — so it asks first if that drops your
  messages or skips back over tool calls that really ran (those aren't
  undone). Everything dropped stays in the session log. At
  `OGNEE_TEMPERATURE=0` a redo will usually come out the same.

`/help` lists every command (generated from the command registry);
`/help <command>` shows details and aliases.

## MCP modes

- **stdio** (default): servers from `mcp.json` are spawned as subprocesses,
  with per-server reader/stderr threads, stdin write serialization, crash-loop
  budgets, and auto-restart.
- **proxy** (`MCP_MODE=proxy`): attach to an already-running mcpo-style HTTP
  proxy instead of spawning servers.

## Model profiles

OGnee auto-detects the tool-calling format per model: native OpenAI
`tool_calls`, Qwen XML (`<tool_call>`), or Gemma-style function-call tags, and
falls back to extracting JSON tool calls from text with a string-aware parser.
The registry of actually-available tools is enforced before dispatch —
hallucinated tool names are rejected with a format correction, never executed.

## Development

```sh
python3 -m compileall -q .          # syntax check
python3 -m unittest discover -s tests   # full test suite (stdlib unittest)
```

## License

MIT — see LICENSE.
