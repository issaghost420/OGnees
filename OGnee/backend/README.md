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

`prompt_toolkit`, `wcwidth`, and `requests` are the only runtime deps and are
all available in nixpkgs if you prefer a `buildPythonApplication` wrapper.

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
- Gate prompts **fail closed**: Enter, garbage input, EOF, or Ctrl-C at a
  trust gate means Deny.

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
