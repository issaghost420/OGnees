#!/usr/bin/env python3
"""
ognee.py - Entry point for OGnee

Usage:
    python ognee.py                    # Interactive mode
    python ognee.py "do something"     # One-shot task
    python ognee.py --help             # Show help

Environment:
    LM_STUDIO_URL     - API endpoint (default: http://192.168.50.140:1234/v1/chat/completions)
    ENVIRONMENT_FILE  - /home/tower/.secrets/lms.env
#    LM_STUDIO_TOKEN  - API token
    MCP_CONFIG        - Path to mcp.json (default: ~/towerstack/OGnee/backend/mcp.json)
"""

import sys
import types
import argparse
from pathlib import Path


def _bootstrap_backend_package() -> None:
    """Make this directory importable as the ``backend`` package.

    Works regardless of the directory's actual name: if the directory is
    literally named ``backend``, a normal sys.path import is used; otherwise
    the directory is registered as a module with ``__path__`` so the
    relative imports inside the package resolve (this is what broke
    ``python ognee.py`` whenever the checkout wasn't named ``backend/``).
    """
    pkg_dir = Path(__file__).resolve().parent
    if "backend" in sys.modules:
        return
    if pkg_dir.name == "backend":
        parent = str(pkg_dir.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
    else:
        pkg = types.ModuleType("backend")
        pkg.__path__ = [str(pkg_dir)]
        sys.modules["backend"] = pkg


_bootstrap_backend_package()

from backend.agent import run_interactive, run_one_shot, get_current_model
from backend.config import (
    DEFAULT_TRUST_TIER,
    TIERS,
    MCP_CONFIG,
    ensure_directories,
)
from backend.display import banner, print_system, print_error, COLORS

def main():
    parser = argparse.ArgumentParser(
        description="OGnee - Autonomous agent framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Trust Tiers:
  0 - OGreenie 🌱  Probation, every tool needs approval
  1 - OSeemie  👀  Supervised, shadow writes allowed
  2 - OGneey   🧞  Trusted, direct writes to safe paths  
  3 - OFreeney 🧞‍♂️✨ Autonomous, full permissions (except rm)

Examples:
  python ognee.py                          # Interactive mode
  python ognee.py "check nginx logs"       # One-shot task
  python ognee.py -t 0 "risky task"        # Run as OGreenie (probation)
  python ognee.py --model qwen3-32b        # Specify model
        """
    )
    
    parser.add_argument(
        "task",
        nargs="?",
        help="Task to execute (omit for interactive mode)"
    )
    
    parser.add_argument(
        "-t", "--tier",
        type=int,
        choices=[0, 1, 2, 3],
        default=DEFAULT_TRUST_TIER,
        help=f"Trust tier (default: {DEFAULT_TRUST_TIER})"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Model name (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help=f"Path to mcp.json (default: {MCP_CONFIG})"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output (for scripting)"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show MCP server and host status, then exit"
    )
    
    parser.add_argument(
        "--hosts",
        action="store_true",
        help="Show SSH host connectivity, then exit"
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    # Status check mode
    if args.status:
        from backend.mcp import MCPManager
        from backend.ssh import print_squad_status
        
        banner()
        
        mcp = MCPManager(args.config)
        if mcp.load_config():
            try:
                mcp.start_all()
                mcp.print_status()
            finally:
                # Clean up servers even if status printing crashes
                mcp.stop_all()
        
        print_squad_status()
        return 0
    
    # Host check mode
    if args.hosts:
        from backend.ssh import print_squad_status
        banner()
        print_squad_status()
        return 0
    
    # Detect model if not specified. --quiet silences the chatter, NOT the
    # detection itself — skipping detection left model_name=None and forced
    # a wasteful re-detection (a real completion call) deeper in the stack.
    model_name = args.model
    if not model_name:
        if not args.quiet:
            print_system("Detecting model...")
        model_name = get_current_model()
        if model_name == "unknown":
            print_error("Could not detect model. Is LM Studio running?")
            print_system(f"Trying to connect to: {COLORS.DIM}(check LM_STUDIO_URL env var){COLORS.RESET}")
            return 1
    
    # Run mode
    if args.task:
        # One-shot mode
        success = run_one_shot(
            task=args.task,
            model_name=model_name,
            trust_level=args.tier,
            config_path=args.config
        )
        return 0 if success else 1
    else:
        # Interactive mode
        run_interactive(
            model_name=model_name,
            trust_level=args.tier,
            config_path=args.config
        )
        return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n👋 Bye!")
        sys.exit(0)
