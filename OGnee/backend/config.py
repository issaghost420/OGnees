#!/usr/bin/env python3
"""
config.py - Central configuration for OGnee

All the knobs in one place. Paths, limits, model configs, tier definitions.
OGneeys are the model instances. OGnee is the system that manages them.
"""

import os
import re
import socket
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# =============================================================================
# HOST DETECTION
# =============================================================================

HOSTNAME = socket.gethostname().lower()

HOSTS = {
    "tower": {
        "ip": "192.168.50.140",
        "wireguard_ip": "10.66.66.2",
        "role": "primary",
        "has_gpu": True,
        "lm_studio_port": 1234,
    },
    "titan": {
        "ip": "192.168.50.197",
        "wireguard_ip": "10.66.66.4",
        "role": "secondary",
        "has_gpu": True,
        "lm_studio_port": 1234,
    },
    "dacrib": {
        "ip": "192.168.50.141",
        "wireguard_ip": "10.66.66.1",
        "role": "homelab",
        "has_gpu": False,
        "lm_studio_port": None,
    },
}

def get_current_host() -> dict:
    return HOSTS.get(HOSTNAME, HOSTS["tower"])

def get_host(name: str) -> dict:
    return HOSTS.get(name.lower(), None)

# =============================================================================
# PATHS
# =============================================================================

HOME = Path.home()
TOWERSTACK = HOME / "towerstack"
OGNEE_ROOT = TOWERSTACK / "OGnee"
BACKEND_DIR = OGNEE_ROOT / "backend"
OGNEEYS_DIR = OGNEE_ROOT / "OGneeys"
DATA_PLAIN = HOME / "archive-of-infinite-fold" / "tower-data-plain"

# OGneey working directories
SHADOW_ROOT = OGNEEYS_DIR / "shadow"
SESSION_LOGS = OGNEEYS_DIR / "sessions"
STATE_FILE = OGNEEYS_DIR / "state.md"          # The persistent brain
CONTINUATION_DIR = OGNEEYS_DIR / "continuations"
SCRATCH_DIR = OGNEEYS_DIR / "scratch"
TOOL_LOG = OGNEEYS_DIR / "tool-calls.log"

# MCP config
MCP_CONFIG = BACKEND_DIR / "mcp.json"
MCP_CONFIG_FALLBACK = DATA_PLAIN / "lm-studio" / "mcp.json"

# =============================================================================
# READ-ONLY PATHS
# =============================================================================

READ_ONLY_PATHS = [
    "/etc",
    str(HOME),
    "/data",
    "/hdd2",
    "/sata",
    "/990",
    "/run",
    "/var/log",
]

EXCLUDED_PATHS = [
    str(HOME / "Documents"),
    str(HOME / "Downloads"),
    str(HOME / ".local/share/Trash"),
    str(HOME / ".cache"),
    str(HOME / ".mozilla"),
    str(HOME / ".librewolf"),
    # Credential stores — never model-readable.
    str(HOME / ".ssh"),
    str(HOME / ".gnupg"),
    str(HOME / ".aws"),
    str(HOME / ".azure"),
    str(HOME / ".kube"),
    str(HOME / ".config"),
    str(HOME / ".secrets"),
    str(HOME / ".netrc"),
]

# =============================================================================
# WRITABLE PATHS
# =============================================================================

WRITABLE_PATHS = [
    str(SHADOW_ROOT),
    str(SESSION_LOGS),
    str(CONTINUATION_DIR),
    str(SCRATCH_DIR),
    str(OGNEEYS_DIR),
    "/tmp",
]

# =============================================================================
# SSH CONFIG
# =============================================================================

@dataclass
class SSHConfig:
    enabled: bool = True
    user: str = "tower"
    key_path: Path = HOME / ".ssh" / "id_ed25519"
    known_hosts: Path = HOME / ".ssh" / "known_hosts"
    timeout: int = 10
    allowed_targets: List[str] = field(default_factory=lambda: list(HOSTS))
    accept_remote_commands: bool = True
    remote_listen_port: int = 7777

SSH = SSHConfig()

def get_ssh_target(host: str) -> Optional[str]:
    host_config = get_host(host)
    if not host_config or host.lower() not in SSH.allowed_targets:
        return None
    ip = host_config.get("wireguard_ip") or host_config.get("ip")
    return f"{SSH.user}@{ip}"

def build_ssh_command(host: str, command: str) -> List[str]:
    target = get_ssh_target(host)
    if not target:
        raise ValueError(f"Unknown or disallowed host: {host}")
    return [
        "ssh",
        "-i", str(SSH.key_path),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"ConnectTimeout={SSH.timeout}",
        "-o", "BatchMode=yes",
        target,
        command,
    ]

# =============================================================================
# LM STUDIO MODEL DETECTION
# =============================================================================

def get_loaded_model_info() -> Dict[str, Any]:
    """Get info from currently loaded LM Studio model via `lms ps --json`."""
    try:
        # Build clean environment without nix library paths that cause segfaults
        # The lms CLI (Electron/Node) conflicts with nix store libraries
        clean_env = os.environ.copy()
        for var in ['LD_LIBRARY_PATH', 'LD_PRELOAD']:
            if var in clean_env:
                # Filter out nix store paths, keep system paths
                paths = clean_env[var].split(':')
                clean_paths = [p for p in paths if '/nix/store' not in p]
                if clean_paths:
                    clean_env[var] = ':'.join(clean_paths)
                else:
                    del clean_env[var]
        
        result = subprocess.run(
            ["/home/tower/.lmstudio/bin/lms", "ps", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
            env=clean_env
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data and len(data) > 0:
                model = data[0]
                return {
                    "model": model.get("modelKey", "unknown"),
                    "display_name": model.get("displayName", "Unknown Model"),
                    "max_context": model.get("maxContextLength", 32000),
                    "context": model.get("contextLength", 32000),
                    "vision": model.get("vision", False),
                    "tool_use": model.get("trainedForToolUse", False),
                    "parallel": model.get("parallel", 1),
                    "architecture": model.get("architecture", "unknown"),
                    "quantization": model.get("quantization", {}).get("name", "unknown"),
                    "size_bytes": model.get("sizeBytes", 0),
                    "status": model.get("status", "unknown"),
                }
    except FileNotFoundError:
        # lms command not found - not on a machine with LM Studio CLI
        pass
    except subprocess.TimeoutExpired:
        pass
    except json.JSONDecodeError:
        pass
    except Exception:
        pass
    
    # Fallback - no model info available
    return {
        "model": "unknown",
        "display_name": "Unknown Model",
        "max_context": 32000,
        "context": 32000,
        "vision": False,
        "tool_use": False,
        "parallel": 1,
        "architecture": "unknown",
        "quantization": "unknown",
        "size_bytes": 0,
        "status": "unknown",
    }

# =============================================================================
# MODEL CONFIGS
# =============================================================================

@dataclass
class ModelConfig:
    name: str
    context_limit: int
    hard_stop_threshold: int
    warning_threshold: int
    tokens_per_turn_estimate: int

MODELS = {
    "qwen3-32b": ModelConfig("qwen3-32b", 128_000, 30_000, 50_000, 2_000),
    "qwen3-14b": ModelConfig("qwen3-14b", 128_000, 30_000, 50_000, 1_500),
    "qwen3-8b": ModelConfig("qwen3-8b", 128_000, 30_000, 50_000, 1_200),
    "mistral-nemo": ModelConfig("mistral-nemo", 128_000, 30_000, 50_000, 1_200),
    "llama-70b": ModelConfig("llama-70b", 128_000, 30_000, 50_000, 2_500),
    "nemotron-super": ModelConfig("nemotron-super", 128_000, 30_000, 50_000, 2_000),
    "default": ModelConfig("unknown", 32_000, 8_000, 12_000, 1_000),
}

# =============================================================================
# TRUST TIERS
# =============================================================================

@dataclass
class TrustTier:
    name: str
    level: int
    can_read: bool = True
    can_write_shadow: bool = False
    can_write_direct: bool = False
    can_sudo: bool = False
    can_rm: bool = False                    # NEVER
    can_network: bool = False
    can_ssh: bool = False
    requires_tool_approval: bool = True
    requires_plan_approval: bool = True
    auto_fallback_tasks: bool = False
    check_in_interval: int = 5

TIERS = {
    0: TrustTier("probation", 0, can_write_shadow=False, can_ssh=False,                          # OGreenie
                 requires_tool_approval=True, requires_plan_approval=True, check_in_interval=3),
    1: TrustTier("supervised", 1, can_write_shadow=True, can_ssh=False,                          # OSeemie
                 requires_tool_approval=False, requires_plan_approval=True, check_in_interval=5),
    2: TrustTier("trusted", 2, can_write_shadow=True, can_write_direct=True,                     # OGneey
                 can_network=True, can_ssh=True, requires_tool_approval=False,
                 requires_plan_approval=False, check_in_interval=10),
    3: TrustTier("autonomous", 3, can_write_shadow=True, can_write_direct=True,                  # OFreeney
                 can_network=True, can_ssh=True, can_sudo=True, requires_tool_approval=False,
                 requires_plan_approval=False, auto_fallback_tasks=True, check_in_interval=0),
}

DEFAULT_TRUST_TIER = 1

# =============================================================================
# BANNED OPERATIONS
# =============================================================================

BANNED_COMMANDS = [
    "rm -rf /", "rm -rf ~", "rm -rf /*", ":(){:|:&};:",
    "mkfs", "dd if=/dev/zero", "dd if=/dev/random",
    "chmod -R 777 /", "> /dev/sda", "mv /* /dev/null",
]

SUDO_PATTERNS = ["sudo ", "doas ", "pkexec ", "su -c", "su root"]
RM_PATTERNS = ["rm ", "rmdir ", "unlink ", "shred "]

# =============================================================================
# DISPLAY
# =============================================================================

@dataclass
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Semantic
    INTERNAL = DIM + BRIGHT_BLACK
    CONVERSATION = BRIGHT_WHITE
    WARNING = YELLOW
    ERROR = BRIGHT_RED
    SYSTEM = CYAN
    SUCCESS = BRIGHT_GREEN
    COMMAND = MAGENTA
    TOOL_NAME = BLUE
    HOST = BRIGHT_CYAN
    OGNEE = BRIGHT_MAGENTA              # OGnee branding color

COLORS = Colors()
STATUS_REFRESH_INTERVAL = 1.0

# =============================================================================
# API SETTINGS
# =============================================================================

def get_lm_studio_url(host: str = None) -> str:
    if host:
        host_config = get_host(host)
        if host_config and host_config.get("lm_studio_port"):
            ip = host_config.get("ip")
            port = host_config.get("lm_studio_port")
            return f"http://{ip}:{port}/v1/chat/completions"
    return os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")

LM_STUDIO_URL = get_lm_studio_url()
LM_STUDIO_TOKEN = os.environ.get("LM_STUDIO_TOKEN", "lm-studio")

REQUEST_TIMEOUT = 120
TOOL_TIMEOUT = 15  # default per-tool timeout (seconds)
MAX_ITERATIONS = 25
TOOL_FAILURE_BACKOFF = 3

# Per-server/per-tool timeout overrides (seconds). Keys may be "tool",
# "server/tool", or "server". Environment overrides:
#   OGNEE_TOOL_TIMEOUT=60                  (global default)
#   OGNEE_TOOL_TIMEOUT_DEEP_RESEARCH=300   (per-tool; uppercase, - . -> _)
TOOL_TIMEOUTS: dict = {
    "deep_research": 300,
    "deep-research": 300,
    "deepresearch": 300,
    "browser": 60,
    "playwright": 60,
}


def get_tool_timeout(server: str = "", tool: str = "") -> int:
    """Resolve the timeout (seconds) for a tool call.

    Precedence: env per-key override > TOOL_TIMEOUTS["server/tool"] >
    TOOL_TIMEOUTS[tool] > TOOL_TIMEOUTS[server] > env OGNEE_TOOL_TIMEOUT >
    TOOL_TIMEOUT default.
    """
    def _env_key(name: str) -> str:
        return "OGNEE_TOOL_TIMEOUT_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()

    candidates = [f"{server}/{tool}", tool, server]
    for name in candidates:
        if not name or name.startswith("/"):
            continue
        val = os.environ.get(_env_key(name))
        if val:
            try:
                return max(1, int(val))
            except ValueError:
                pass
        if name in TOOL_TIMEOUTS:
            return TOOL_TIMEOUTS[name]
    val = os.environ.get("OGNEE_TOOL_TIMEOUT")
    if val:
        try:
            return max(1, int(val))
        except ValueError:
            pass
    return TOOL_TIMEOUT

# =============================================================================
# SESSION SETTINGS
# =============================================================================

STATE_INJECTION_INTERVAL = 3  # inject state every 3 turns (was 1 — too chatty for token budget)
LOG_TOOL_CALLS = True
LOG_FULL_RESPONSES = True
LOG_TOKEN_USAGE = True
CONTINUATION_SUMMARY_TOKENS = 2000

# =============================================================================
# HELPERS
# =============================================================================

def get_model_config(model_name: str) -> ModelConfig:
    if model_name in MODELS:
        return MODELS[model_name]
    for key in MODELS:
        if key in model_name.lower():
            return MODELS[key]
    return MODELS["default"]

def get_trust_tier(level: int = DEFAULT_TRUST_TIER) -> TrustTier:
    return TIERS.get(level, TIERS[0])

def ensure_directories():
    for path in [SHADOW_ROOT, SESSION_LOGS, CONTINUATION_DIR, SCRATCH_DIR]:
        path.mkdir(parents=True, exist_ok=True)

def _resolve_path(path) -> Path:
    """Expand ``~`` and resolve to an absolute path.

    ``Path.resolve()`` alone does NOT expand a leading tilde, so every
    path check in this module goes through here (canonical fix for the
    tilde bugs previously patched ad-hoc in gates.py/context.py).
    """
    return Path(os.path.expanduser(str(path))).resolve()

# Path lists resolved once at load time (with ~ expansion) so checks use
# Path.is_relative_to() against real Path objects instead of string
# startswith matching (which let "/tmp" match "/tmpevil").
_EXCLUDED_RESOLVED = [_resolve_path(p) for p in EXCLUDED_PATHS]
_READ_ONLY_RESOLVED = [_resolve_path(p) for p in READ_ONLY_PATHS]
_WRITABLE_RESOLVED = [_resolve_path(p) for p in WRITABLE_PATHS]

def is_path_excluded(path: str) -> bool:
    p = _resolve_path(path)
    return any(p.is_relative_to(ex) for ex in _EXCLUDED_RESOLVED)

def is_path_read_only(path: str) -> bool:
    p = _resolve_path(path)
    if is_path_excluded(path):
        return False
    return any(p.is_relative_to(ro) for ro in _READ_ONLY_RESOLVED)

def is_path_writable(path: str) -> bool:
    p = _resolve_path(path)
    return any(p.is_relative_to(w) for w in _WRITABLE_RESOLVED)

def can_access_path(path: str) -> bool:
    if is_path_excluded(path):
        return False
    return is_path_read_only(path) or is_path_writable(path)

def get_shadow_path(real_path) -> Path:
    """Map a real filesystem path to its staging path under SHADOW_ROOT.

    THE canonical shadow mapping (shadow.py imports this — there is no
    second implementation). Semantics:

    - Path already under SHADOW_ROOT  -> returned unchanged (idempotent;
      re-gating a staged file must never double the shadow prefix).
    - Path under OGNEE_ROOT           -> SHADOW_ROOT/<relative to root>.
    - Anything else                   -> SHADOW_ROOT/outside/<from />.
    """
    real = _resolve_path(real_path)
    if real.is_relative_to(SHADOW_ROOT):
        return real
    if real.is_relative_to(OGNEE_ROOT):
        return SHADOW_ROOT / real.relative_to(OGNEE_ROOT)
    return SHADOW_ROOT / "outside" / real.relative_to(real.anchor)

def get_real_path(shadow_path) -> Path:
    """Inverse of :func:`get_shadow_path` (used by /diff and /apply).

    A path that is not under SHADOW_ROOT is already a real path and is
    returned unchanged.
    """
    sp = _resolve_path(shadow_path)
    if not sp.is_relative_to(SHADOW_ROOT):
        return sp
    rel = sp.relative_to(SHADOW_ROOT)
    if rel.parts and rel.parts[0] == "outside":
        # Ambiguity guard: a real file at OGNEE_ROOT/outside/... maps to the
        # same shadow path as /... via the outside prefix. Prefer the
        # under-root interpretation when that real file actually exists.
        under_root = OGNEE_ROOT / rel
        if under_root.exists():
            return under_root
        return Path("/") / Path(*rel.parts[1:])
    return OGNEE_ROOT / rel

def is_remote_session() -> bool:
    return bool(os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"))

def get_caller_host() -> Optional[str]:
    ssh_client = os.environ.get("SSH_CLIENT", "")
    if ssh_client:
        caller_ip = ssh_client.split()[0]
        for name, config in HOSTS.items():
            if config["ip"] == caller_ip or config.get("wireguard_ip") == caller_ip:
                return name
    return None
