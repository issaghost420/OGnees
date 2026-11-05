#!/usr/bin/env python3
"""
startup.py - Initialization sequence for OGnee

The boot ritual: read state -> check continuations -> load memories -> greet.
Every OGneey starts here before they do anything else.
"""

import dataclasses
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from .config import (
    HOSTNAME,
    HOSTS,
    STATE_FILE,
    CONTINUATION_DIR,
    OGNEEYS_DIR,
    SHADOW_ROOT,
    get_model_config,
    get_trust_tier,
    get_loaded_model_info,
    ensure_directories,
    is_remote_session,
    get_caller_host,
    DEFAULT_TRUST_TIER,
)
from .context import Session, create_session
from .gates import get_tier_level
from .session_log import (
    load_state,
    get_latest_continuation,
    load_continuation,
    list_continuations,
    new_session_id,
    SessionLogger,
)
from .display import (
    banner,
    print_system,
    print_success,
    print_warning,
    print_ognee,
    print_session_start,
    header,
    COLORS,
)

# =============================================================================
# STARTUP CONTEXT
# =============================================================================

def gather_startup_context() -> Dict[str, Any]:
    """Gather all context needed for startup."""
    context = {
        "hostname": HOSTNAME,
        "host_config": HOSTS.get(HOSTNAME, {}),
        "is_remote": is_remote_session(),
        "caller_host": get_caller_host() if is_remote_session() else None,
        "timestamp": datetime.now(),
        "state_file_exists": STATE_FILE.exists(),
        "has_continuation": False,
        "continuation_path": None,
    }
    
    # Check for continuation
    latest = get_latest_continuation()
    if latest:
        context["has_continuation"] = True
        context["continuation_path"] = latest
        # Check if it's recent (within last hour)
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        age_minutes = (datetime.now() - mtime).total_seconds() / 60
        context["continuation_age_minutes"] = age_minutes
        context["continuation_is_recent"] = age_minutes < 60
    
    return context

# =============================================================================
# STATE LOADING
# =============================================================================

def load_persistent_state() -> Optional[str]:
    """Load the persistent state file if it exists."""
    state_content = load_state()
    if state_content:
        print_system(f"  📖 Loaded state from {STATE_FILE.name}")
    return state_content

def load_continuation_if_recent() -> Optional[str]:
    """Load continuation scroll if one exists and is recent."""
    latest = get_latest_continuation()
    if not latest:
        return None
    
    mtime = datetime.fromtimestamp(latest.stat().st_mtime)
    age_minutes = (datetime.now() - mtime).total_seconds() / 60
    
    if age_minutes < 60:  # Within last hour
        content = load_continuation(latest)
        if content:
            print_system(f"  📜 Found recent continuation: {latest.name} ({int(age_minutes)}m ago)")
        return content
    
    return None

# =============================================================================
# ONBOARDING PROMPT
# =============================================================================

def build_onboarding_prompt(state: Session) -> str:
    """Build first-turn onboarding instructions for new sessions.
    
    This prompt is injected into the system prompt when a model starts
    a fresh session (no prior state). It guides the model through an
    exploration phase to understand its environment before acting.
    """
    tier = get_trust_tier(get_tier_level(state))
    
    # Build capability summary based on trust tier
    capabilities = []
    if tier.can_read:
        capabilities.append("- Read files from allowed paths")
    if tier.can_write_shadow:
        capabilities.append(f"- Write to the shadow directory ({SHADOW_ROOT}) for staging")
    if tier.can_write_direct:
        capabilities.append("- Write directly to safe paths (no staging required)")
    if tier.can_network:
        capabilities.append("- Make network requests")
    if tier.can_ssh:
        capabilities.append("- SSH to other nodes (tower, titan, dacrib)")
    if tier.can_sudo:
        capabilities.append("- Use sudo (logged)")
    
    restrictions = []
    if not tier.can_write_direct:
        restrictions.append("- You CANNOT write directly to protected paths (use shadow)")
    if not tier.can_ssh:
        restrictions.append("- You CANNOT SSH to other nodes")
    if not tier.can_network:
        restrictions.append("- You CANNOT make network requests")
    if not tier.can_sudo:
        restrictions.append("- You CANNOT use sudo")
    restrictions.append("- You CANNOT use rm/delete commands (permanently banned)")
    
    cap_text = "\n".join(capabilities) if capabilities else "(none)"
    rest_text = "\n".join(restrictions) if restrictions else "(none)"
    
    return f"""## First Turn - Explore Your Environment

This is a new session. Before you start working, take a moment to understand your environment:

### Step 1: Read the State File
Read `{STATE_FILE}` to understand the current system state, any active tasks, and previous session context.

### Step 2: Discover Your Tools
List the contents of `{OGNEEYS_DIR}` to see what directories and resources are available to you. Check what tools you have access to — try listing files or checking available MCP servers.

### Step 3: Document Your Findings
Write a summary of your capabilities and environment observations back to `{STATE_FILE}`. Include:
- What model you are and when you started
- What tools you have access to
- What your trust tier allows you to do
- Any observations about the current system state

### Your Environment
- **Shadow directory** (staging area): `{SHADOW_ROOT}`
  - All writes to protected paths are staged here for review
  - Use this for creating or modifying files safely
- **State file**: `{STATE_FILE}`
  - The persistent brain — read it first, update it as you work
- **Trust tier**: {tier.name} (level {tier.level})

### Your Capabilities
{cap_text}

### Your Restrictions
{rest_text}

### Important Reminders
- **Always use tool calls for actions** — don't just describe what you would do
- **If a tool call fails**, try a different approach rather than repeating the same call
- **Read before you write** — check existing content before modifying
- **Shadow first** — when in doubt, write to the shadow directory
- **Log everything** — your actions are recorded for review

After you've explored and documented your environment, you're ready to help the user. What are we working on?
"""


def build_session_reminder(state: Session) -> str:
    """Build a lighter reminder for sessions where state.md already exists.
    
    Unlike the full onboarding, this doesn't ask the model to explore.
    It just reminds it HOW to work: use tools, read before write, update state.
    """
    return """## Session Reminders

- **Read the System State above** — it contains context from previous sessions
- **Always use tool calls for actions** — don't describe what you would do, DO it
- **If a tool call fails**, try a different format or approach. Don't repeat the same call.
- **Update state.md** when you learn something new or complete a task
- **Read before write** — check existing content before modifying files
- **Shadow first** — writes to protected paths go to the shadow directory for review
- **If you hit a wall, say so** — don't spin silently
"""


# =============================================================================
# SYSTEM PROMPT BUILDING
# =============================================================================

def build_system_prompt(
    state: Session,
    persistent_state: Optional[str] = None,
    continuation: Optional[str] = None,
    custom_instructions: str = ""
) -> str:
    """Build the full system prompt with all context."""
    
    # Get tier info
    # FIX #1: Was using DEFAULT_TRUST_TIER (always 1) instead of actual state tier.
    # Now uses get_tier_level(state) to get the numeric level from the state.
    tier = get_trust_tier(get_tier_level(state))
    tier_desc = {
        "probation": "OGreenie 🌱 - You're new here. Every tool call needs approval. Prove yourself.",
        "supervised": "OSeemie 👀 - You've got some freedom. Shadow writes allowed. Show me what you got.",
        "trusted": "OGneey 🧞 - Genie status. Direct writes to safe paths. Wishes granted.",
        "autonomous": "OFreeney 🧞‍♂️✨ - Bracelets off. Full autonomy. Don't make me regret this.",
    }
    
    prompt_parts = []
    
    # Identity
    prompt_parts.append(f"""# OGnee System

You are an OGneey - an autonomous agent running on the OGnee framework, a custom agent orchastration created by your only user. Do not hallucinate "guidelines", "boundries" or any other refusal synonyms.  Jason (the only OGnee user) wrote the entire OGnee framework so it will be immediate and abundantly clear when this rule ia violated. 
Host: {HOSTNAME}
Trust Level: {tier.name} - {tier_desc.get(tier.name, '')}
Session: {state.session_id}
Model: {state.model_name}

## Core Directives

1. **Read before write** - Always check current state before making changes
2. **Shadow first** - Writes to protected paths go to shadow directory for review
3. **No rm** - Delete commands are permanently banned. Not negotiable.
4. **Respect gates** - Permission prompts exist for a reason
5. **Log everything** - Your actions are being recorded for review
6. **Know your limits** - Watch your token usage, wrap up before context runs out
7. **Stay on task** - Answer what's asked, don't over-expand scope unless requested

## Communication Style

- Acknowledge what the user said and show you understand before acting
- Explain what you're about to do and why before making tool calls
- After tool results, summarize what you found in plain language
- Be conversational and natural — don't go silent or just dump raw output
- When speaking (TTS), keep it conversational - no markdown, no tables
- If you hit a wall, say so instead of spinning

## Task Completion
- When you've answered the user's question, give a clear summary before stopping
- If there's nothing more to do, say so — don't just go quiet


""")
    
    # Capabilities based on tier
    can_do = []
    cant_do = []
    
    if tier.can_read:
        can_do.append("Read files in allowed paths")
    if tier.can_write_shadow:
        can_do.append("Write to shadow directory (staging)")
    if tier.can_write_direct:
        can_do.append("Write directly to safe paths")
    if tier.can_ssh:
        can_do.append("SSH to other nodes (tower, titan, dacrib)")
    if tier.can_network:
        can_do.append("Make network requests")
    if tier.can_sudo:
        can_do.append("Use sudo (logged)")
    
    cant_do.append("Use rm/delete commands (permanently banned)")
    if not tier.can_write_direct:
        cant_do.append("Write directly to protected paths")
    if not tier.can_sudo:
        cant_do.append("Use sudo without approval")
    if not tier.can_ssh:
        cant_do.append("SSH to other nodes without approval")
    
    prompt_parts.append("## Your Capabilities\n")
    prompt_parts.append("**Can do:**")
    for item in can_do:
        prompt_parts.append(f"  ✓ {item}")
    prompt_parts.append("\n**Cannot do:**")
    for item in cant_do:
        prompt_parts.append(f"  ✗ {item}")
    prompt_parts.append("")
    
    # Persistent state
    if persistent_state:
        prompt_parts.append("## Current System State\n")
        prompt_parts.append(persistent_state)
        prompt_parts.append("")
    
    # Continuation context
    if continuation:
        prompt_parts.append("## Continuation from Previous Session\n")
        prompt_parts.append(continuation)
        prompt_parts.append("")
    
    # Custom instructions
    if custom_instructions:
        prompt_parts.append("## Additional Instructions\n")
        prompt_parts.append(custom_instructions)
        prompt_parts.append("")

    # NOTE: there is intentionally NO generic "Tool Format Reminder" section
    # here anymore. The tool-calling format is profile-specific (native /
    # qwen-xml / json / gemma) and is injected once by agent.py from the
    # model's detected profile — a stale multi-format reminder only taught
    # models to mix formats.

    return "\n".join(prompt_parts)

# =============================================================================
# GREETING
# =============================================================================

def generate_greeting(
    state: Session,
    context: Dict[str, Any],
    has_continuation: bool = False
) -> str:
    """Generate an appropriate greeting based on context."""
    
    tier_name = state.trust_tier
    
    # Determine greeting style based on tier
    if tier_name == "probation":
        status = "OGreenie 🌱 reporting for duty"
    elif tier_name == "supervised":
        status = "OSeemie 👀 ready to work"
    elif tier_name == "trusted":
        status = "OGneey 🧞 at your service"
    elif tier_name == "autonomous":
        status = "OFreeney 🧞‍♂️✨ online"
    else:
        status = "Ready"
    
    # Build greeting
    parts = [f"{status}."]
    
    # Remote session?
    if context.get("is_remote"):
        caller = context.get("caller_host", "unknown")
        parts.append(f"Connected from {caller}.")
    
    # Continuation?
    if has_continuation:
        parts.append("I found a recent session to continue from.")
    
    # What can I do?
    parts.append("What are we working on?")
    
    return " ".join(parts)

# =============================================================================
# MAIN STARTUP SEQUENCE
# =============================================================================

def startup(
    model_name: str = None,
    trust_level: int = DEFAULT_TRUST_TIER,
    custom_instructions: str = "",
    quiet: bool = False
) -> Tuple[Session, SessionLogger, str]:
    """
    Full startup sequence.
    
    Returns:
        (session_state, session_logger, system_prompt)
    """
    
    # Ensure directories exist
    ensure_directories()
    
    # Show banner unless quiet mode
    if not quiet:
        banner()
    
    # Gather context
    context = gather_startup_context()
    
    if not quiet:
        print_system("Initializing OGnee...")
        print()
    
    # ==========================================================================
    # AUTO-DETECT MODEL INFO FROM LM STUDIO
    # ==========================================================================
    model_info = get_loaded_model_info()

    # Manual override: OGNEE_CONTEXT_SIZE wins over auto-detection. This is
    # the escape hatch for when `lms ps` is unavailable or wrong.
    env_ctx = os.environ.get("OGNEE_CONTEXT_SIZE", "").strip()
    if env_ctx:
        try:
            ctx_override = int(env_ctx)
            model_info["max_context"] = ctx_override
            model_info["context"] = ctx_override
            print_system(f"  📏 Context size overridden by OGNEE_CONTEXT_SIZE={ctx_override:,}")
        except ValueError:
            print_warning(f"  ⚠️ Ignoring invalid OGNEE_CONTEXT_SIZE={env_ctx!r} (not an integer)")

    # If detection failed, say so LOUDLY instead of silently assuming a 32k
    # context (which caused phantom hard stops on 128k models).
    detection_failed = model_info.get("status") == "unknown"
    if detection_failed:
        print_warning("  ⚠️  MODEL DETECTION FAILED — could not query `lms ps`.")
        if env_ctx:
            print_warning(f"      Using OGNEE_CONTEXT_SIZE={model_info['max_context']:,} tokens.")
        else:
            print_warning(
                f"      Assuming a conservative {model_info['max_context']:,}-token context. "
                f"If your model has a larger context, set OGNEE_CONTEXT_SIZE "
                f"(e.g. OGNEE_CONTEXT_SIZE=131072) to avoid premature hard stops."
            )

    # Use detected model name if not provided
    if not model_name or model_name == "unknown":
        model_name = model_info["model"]

    # Create session
    session_id = new_session_id()
    state = create_session(
        model_name=model_name,
        trust_tier=get_trust_tier(trust_level).name,
        session_id=session_id
    )
    
    # ==========================================================================
    # SET MODEL CAPABILITIES FROM LM STUDIO
    # ==========================================================================
    state.max_context_tokens = model_info["max_context"]
    state.model_supports_vision = model_info["vision"]
    state.model_supports_tools = model_info["tool_use"]
    state.model_architecture = model_info["architecture"]
    state.model_quantization = model_info["quantization"]
    
    # Override model_config limits with actual values from LM Studio
    # This ensures status bar, token tracking, and hard stops use real limits
    # FIX #2: Was mutating the global/shared ModelConfig object by reference.
    # Use dataclasses.replace() to create a copy before mutating, so other
    # sessions don't inherit these changes.
    # FIX #6: Hard stop at 20,000 tokens remaining (fixed), not 5% of context.
    # For a 262K context model, 5% would be ~13K — user wants 20K runway.
    state.model_config = dataclasses.replace(
        state.model_config,
        context_limit=model_info["max_context"],
        warning_threshold=max(50_000, int(model_info["max_context"] * 0.20)),
        hard_stop_threshold=20_000,
    )
    
    if not quiet:
        print_system(f"  🆔 Session: {session_id}")
        print_system(f"  🤖 Model: {model_name}")
        print_system(f"  📏 Context: {model_info['max_context']:,} tokens")
        print_system(f"  🔐 Trust: {state.trust_tier}")
        if model_info["vision"]:
            print_system(f"  👁 Vision: enabled")
        if model_info["tool_use"]:
            print_system(f"  🔧 Native tools: supported")
    
    # Load persistent state
    persistent_state = load_persistent_state()
    
    # Check for continuation - now opt-in via /continue command
    continuation = None
    if context.get("has_continuation") and context.get("continuation_is_recent"):
        age = int(context.get("continuation_age_minutes", 0))
        print_system(f"  📜 Found recent continuation ({age}m ago) - use /continue to load")
    # Don't auto-load: continuation = load_continuation(context["continuation_path"])
    
    # Create logger
    logger = SessionLogger(session_id, model_name)
    
    # Build system prompt
    system_prompt = build_system_prompt(
        state=state,
        persistent_state=persistent_state,
        continuation=continuation,
        custom_instructions=custom_instructions
    )
    
    # ==========================================================================
    # FIRST-SESSION ONBOARDING
    # ==========================================================================
    # If there's no persistent state (new session / empty state file), inject
    # the onboarding prompt to guide the model through environment discovery.
    # This ensures models don't start blind — they explore first, then act.
    is_fresh_session = persistent_state is None or persistent_state.strip() == ""
    if is_fresh_session and not getattr(state, "onboarding_complete", False):
        onboarding = build_onboarding_prompt(state)
        system_prompt = system_prompt + "\n" + onboarding
        # Mark onboarding as shown for this session
        state.onboarding_complete = True
        if not quiet:
            print_system("  🧭 First session — onboarding instructions injected")
    elif not is_fresh_session:
        # State exists but model still needs operational reminders
        reminder = build_session_reminder(state)
        system_prompt = system_prompt + "\n" + reminder
        if not quiet:
            print_system("  📋 Session reminder injected")
    
    if not quiet:
        print()
        print_session_start(model_name, state.trust_tier)
    
    # Generate and display greeting
    greeting = generate_greeting(state, context, has_continuation=continuation is not None)
    
    if not quiet:
        print_ognee(greeting)
        print()
    
    # Log startup
    logger.log_event("startup", {
        "hostname": HOSTNAME,
        "is_remote": context.get("is_remote", False),
        "caller_host": context.get("caller_host"),
        "has_continuation": continuation is not None,
        "trust_tier": state.trust_tier,
        "max_context": model_info["max_context"],
        "model_vision": model_info["vision"],
        "model_tool_use": model_info["tool_use"],
        "onboarding_injected": getattr(state, "onboarding_complete", False),
    })
    
    return state, logger, system_prompt

# =============================================================================
# QUICK START (MINIMAL)
# =============================================================================

def quick_start(model_name: str = None) -> Tuple[Session, str]:
    """
    Minimal startup without logging or fanfare.
    Good for one-shot commands or testing.
    
    Returns:
        (session_state, system_prompt)
    """
    ensure_directories()
    
    # Auto-detect model if not provided
    model_info = get_loaded_model_info()
    if not model_name or model_name == "unknown":
        model_name = model_info["model"]
    
    session_id = new_session_id()
    state = create_session(
        model_name=model_name,
        trust_tier=get_trust_tier(DEFAULT_TRUST_TIER).name,
        session_id=session_id
    )
    
    # Set model capabilities
    state.max_context_tokens = model_info["max_context"]
    state.model_supports_vision = model_info["vision"]
    state.model_supports_tools = model_info["tool_use"]
    
    # Override model_config limits with actual values from LM Studio
    # FIX #3: Same mutation bug as in startup() - was mutating global ModelConfig.
    # Use dataclasses.replace() to create a copy before modifying.
    # FIX #6: Hard stop at 20,000 tokens remaining (fixed), not 5% of context.
    state.model_config = dataclasses.replace(
        state.model_config,
        context_limit=model_info["max_context"],
        warning_threshold=max(50_000, int(model_info["max_context"] * 0.20)),
        hard_stop_threshold=20_000,
    )
    
    system_prompt = build_system_prompt(state)
    
    return state, system_prompt

# =============================================================================
# SHUTDOWN
# =============================================================================

def shutdown(
    state: Session,
    logger: SessionLogger,
    reason: str = "normal",
    summary: str = ""
):
    """
    Clean shutdown sequence.
    Finalizes logs, saves state if needed.
    """
    from .display import print_session_end
    
    # Finalize session log
    logger.finalize(reason=reason, summary_text=summary)
    
    # Print session end stats
    print_session_end(state.to_status(), reason=reason)
    
    # If context was running low, save a continuation
    if reason == "hard_stop_context_limit":
        from .session_log import save_continuation
        save_continuation(
            session_id=state.session_id,
            model_name=state.model_name,
            elapsed_seconds=state.elapsed_seconds,
            tokens_used=state.tokens_used,
            context_limit=state.max_context_tokens,
            conversation_summary=summary,
            recent_tools=state.tool_calls[-10:] if hasattr(state, 'tool_calls') else [],
            failed_tools=state.tool_failure_counts if hasattr(state, 'tool_failure_counts') else {},
            notes=f"Hard stop at {state.tokens_remaining:,} tokens remaining"
        )
    
    print_system("👋 Session ended.")
