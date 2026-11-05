#!/usr/bin/env python3
"""
ssh.py - Remote execution and cross-node commands

Run commands on the hosts in config.HOSTS (e.g. tower, titan, dacrib).
SSH into the squad, execute tasks, bring back results.
"""

import subprocess
import asyncio
import shlex
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from .config import (
    HOSTNAME,
    HOSTS,
    SSH,
    get_ssh_target,
    build_ssh_command,
    get_host,
)
from .display import (
    print_system,
    print_success,
    print_warning,
    print_error,
    print_dim,
    header,
    COLORS,
)

# =============================================================================
# SSH RESULT
# =============================================================================

@dataclass
class SSHResult:
    """Result from an SSH command."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    host: str
    command: str
    duration_ms: int = 0

# =============================================================================
# CONNECTION TESTING
# =============================================================================

def test_connection(host: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Test SSH connectivity to a host.
    Returns (success, message).
    """
    target = get_ssh_target(host)
    if not target:
        return False, f"Unknown host: {host}"
    
    try:
        cmd = build_ssh_command(host, "echo ok")
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        
        if result.returncode == 0 and "ok" in result.stdout:
            return True, f"Connected to {host}"
        else:
            return False, f"Connection failed: {result.stderr.strip()}"
            
    except subprocess.TimeoutExpired:
        return False, f"Connection timed out after {timeout}s"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def test_all_hosts() -> Dict[str, Tuple[bool, str]]:
    """Test connectivity to all known hosts."""
    results = {}
    
    for host in SSH.allowed_targets:
        if host == HOSTNAME:
            results[host] = (True, "localhost (current)")
        else:
            results[host] = test_connection(host)
    
    return results

def print_host_status():
    """Print connectivity status for all hosts."""
    print()
    print_system("Host Connectivity")
    print_system("─" * 40)
    
    results = test_all_hosts()
    
    for host, (success, message) in results.items():
        host_config = get_host(host)
        ip = host_config.get("ip", "?") if host_config else "?"
        
        if success:
            status = f"{COLORS.SUCCESS}✓{COLORS.RESET}"
        else:
            status = f"{COLORS.ERROR}✗{COLORS.RESET}"
        
        current = " (current)" if host == HOSTNAME else ""
        print(f"  {status} {COLORS.HOST}{host:10}{COLORS.RESET} {ip:15} {message}{current}")
    
    print()

# =============================================================================
# COMMAND EXECUTION
# =============================================================================

def run_ssh_command(
    host: str,
    command: str,
    timeout: int = 30
) -> SSHResult:
    """
    Run a command on a remote host via SSH.
    Synchronous version.
    """
    import time
    start = time.time()
    
    # Handle localhost
    if host == HOSTNAME:
        try:
            # Explicit shell invocation (no shell=True): the argv form is
            # unambiguous about which interpreter runs the command string.
            result = subprocess.run(
                ["/bin/sh", "-c", command],
                capture_output=True,
                timeout=timeout,
                text=True
            )
            duration_ms = int((time.time() - start) * 1000)
            
            return SSHResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                host=host,
                command=command,
                duration_ms=duration_ms
            )
        except subprocess.TimeoutExpired:
            return SSHResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                exit_code=-1,
                host=host,
                command=command,
                duration_ms=timeout * 1000
            )
        except Exception as e:
            return SSHResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                host=host,
                command=command,
                duration_ms=int((time.time() - start) * 1000)
            )
    
    # Remote host
    target = get_ssh_target(host)
    if not target:
        return SSHResult(
            success=False,
            stdout="",
            stderr=f"Unknown or disallowed host: {host}",
            exit_code=-1,
            host=host,
            command=command,
            duration_ms=0
        )
    
    try:
        ssh_cmd = build_ssh_command(host, command)
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        duration_ms = int((time.time() - start) * 1000)
        
        return SSHResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            host=host,
            command=command,
            duration_ms=duration_ms
        )
        
    except subprocess.TimeoutExpired:
        return SSHResult(
            success=False,
            stdout="",
            stderr=f"SSH command timed out after {timeout}s",
            exit_code=-1,
            host=host,
            command=command,
            duration_ms=timeout * 1000
        )
    except Exception as e:
        return SSHResult(
            success=False,
            stdout="",
            stderr=str(e),
            exit_code=-1,
            host=host,
            command=command,
            duration_ms=int((time.time() - start) * 1000)
        )

async def run_ssh_command_async(
    host: str,
    command: str,
    timeout: int = 30
) -> SSHResult:
    """
    Run a command on a remote host via SSH.
    Async version.
    """
    import time
    start = time.time()
    
    # Handle localhost
    if host == HOSTNAME:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout
            )
            
            duration_ms = int((time.time() - start) * 1000)
            
            return SSHResult(
                success=proc.returncode == 0,
                stdout=stdout.decode() if stdout else "",
                stderr=stderr.decode() if stderr else "",
                exit_code=proc.returncode or 0,
                host=host,
                command=command,
                duration_ms=duration_ms
            )
            
        except asyncio.TimeoutError:
            return SSHResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                exit_code=-1,
                host=host,
                command=command,
                duration_ms=timeout * 1000
            )
        except Exception as e:
            return SSHResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                host=host,
                command=command,
                duration_ms=int((time.time() - start) * 1000)
            )
    
    # Remote host
    target = get_ssh_target(host)
    if not target:
        return SSHResult(
            success=False,
            stdout="",
            stderr=f"Unknown or disallowed host: {host}",
            exit_code=-1,
            host=host,
            command=command,
            duration_ms=0
        )
    
    try:
        ssh_cmd = build_ssh_command(host, command)
        
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout
        )
        
        duration_ms = int((time.time() - start) * 1000)
        
        return SSHResult(
            success=proc.returncode == 0,
            stdout=stdout.decode() if stdout else "",
            stderr=stderr.decode() if stderr else "",
            exit_code=proc.returncode or 0,
            host=host,
            command=command,
            duration_ms=duration_ms
        )
        
    except asyncio.TimeoutError:
        return SSHResult(
            success=False,
            stdout="",
            stderr=f"SSH command timed out after {timeout}s",
            exit_code=-1,
            host=host,
            command=command,
            duration_ms=timeout * 1000
        )
    except Exception as e:
        return SSHResult(
            success=False,
            stdout="",
            stderr=str(e),
            exit_code=-1,
            host=host,
            command=command,
            duration_ms=int((time.time() - start) * 1000)
        )

# =============================================================================
# MULTI-HOST EXECUTION
# =============================================================================

async def run_on_all_hosts(
    command: str,
    hosts: List[str] = None,
    timeout: int = 30
) -> Dict[str, SSHResult]:
    """Run a command on multiple hosts in parallel."""
    if hosts is None:
        hosts = SSH.allowed_targets
    
    tasks = []
    for host in hosts:
        tasks.append(run_ssh_command_async(host, command, timeout))
    
    results = await asyncio.gather(*tasks)
    
    return {host: result for host, result in zip(hosts, results)}

def run_on_all_hosts_sync(
    command: str,
    hosts: List[str] = None,
    timeout: int = 30
) -> Dict[str, SSHResult]:
    """Synchronous wrapper for run_on_all_hosts."""
    return asyncio.run(run_on_all_hosts(command, hosts, timeout))

# =============================================================================
# FILE TRANSFER
# =============================================================================

def scp_to_host(
    local_path: str,
    host: str,
    remote_path: str,
    timeout: int = 60
) -> Tuple[bool, str]:
    """
    Copy a file to a remote host.
    Returns (success, message).
    """
    target = get_ssh_target(host)
    if not target:
        return False, f"Unknown host: {host}"
    
    try:
        cmd = [
            "scp",
            "-i", str(SSH.key_path),
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={SSH.timeout}",
            local_path,
            f"{target}:{remote_path}"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        
        if result.returncode == 0:
            return True, f"Copied to {host}:{remote_path}"
        else:
            return False, f"SCP failed: {result.stderr.strip()}"
            
    except subprocess.TimeoutExpired:
        return False, f"Transfer timed out after {timeout}s"
    except Exception as e:
        return False, f"Transfer error: {str(e)}"

def scp_from_host(
    host: str,
    remote_path: str,
    local_path: str,
    timeout: int = 60
) -> Tuple[bool, str]:
    """
    Copy a file from a remote host.
    Returns (success, message).
    """
    target = get_ssh_target(host)
    if not target:
        return False, f"Unknown host: {host}"
    
    try:
        cmd = [
            "scp",
            "-i", str(SSH.key_path),
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={SSH.timeout}",
            f"{target}:{remote_path}",
            local_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        
        if result.returncode == 0:
            return True, f"Copied from {host}:{remote_path}"
        else:
            return False, f"SCP failed: {result.stderr.strip()}"
            
    except subprocess.TimeoutExpired:
        return False, f"Transfer timed out after {timeout}s"
    except Exception as e:
        return False, f"Transfer error: {str(e)}"

# =============================================================================
# REMOTE OGNEE INVOCATION
# =============================================================================

def invoke_remote_ognee(
    host: str,
    task: str,
    timeout: int = 300
) -> SSHResult:
    """
    Invoke OGnee on a remote host with a task.
    Useful for distributed workloads.
    """
    # Escape the task for shell
    escaped_task = shlex.quote(task)
    
    # Build the remote command
    remote_cmd = f"cd ~/towerstack/OGnee && python ognee.py {escaped_task}"
    
    return run_ssh_command(host, remote_cmd, timeout)

async def invoke_remote_ognee_async(
    host: str,
    task: str,
    timeout: int = 300
) -> SSHResult:
    """Async version of invoke_remote_ognee."""
    escaped_task = shlex.quote(task)
    remote_cmd = f"cd ~/towerstack/OGnee && python ognee.py {escaped_task}"
    
    return await run_ssh_command_async(host, remote_cmd, timeout)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_ssh(host: str, command: str) -> str:
    """
    Quick SSH command, returns stdout or error message.
    Good for simple one-liners.
    """
    result = run_ssh_command(host, command)
    
    if result.success:
        return result.stdout.strip()
    else:
        return f"[ERROR] {result.stderr.strip()}"

def get_remote_info(host: str) -> Dict[str, str]:
    """Get basic info about a remote host."""
    info = {}
    
    # Hostname
    result = run_ssh_command(host, "hostname", timeout=5)
    info["hostname"] = result.stdout.strip() if result.success else "unknown"
    
    # Uptime
    result = run_ssh_command(host, "uptime -p", timeout=5)
    info["uptime"] = result.stdout.strip() if result.success else "unknown"
    
    # Load
    result = run_ssh_command(host, "cat /proc/loadavg | cut -d' ' -f1-3", timeout=5)
    info["load"] = result.stdout.strip() if result.success else "unknown"
    
    # Memory
    result = run_ssh_command(host, "free -h | grep Mem | awk '{print $3\"/\"$2}'", timeout=5)
    info["memory"] = result.stdout.strip() if result.success else "unknown"
    
    # GPU (if nvidia-smi available)
    result = run_ssh_command(host, "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'no gpu'", timeout=5)
    info["gpu"] = result.stdout.strip() if result.success else "none"
    
    return info

def print_remote_info(host: str):
    """Print formatted info about a remote host."""
    print_system(f"\n📡 {host}")
    print_system("─" * 30)
    
    info = get_remote_info(host)
    
    print(f"  Hostname: {info.get('hostname', '?')}")
    print(f"  Uptime:   {info.get('uptime', '?')}")
    print(f"  Load:     {info.get('load', '?')}")
    print(f"  Memory:   {info.get('memory', '?')}")
    print(f"  GPU:      {info.get('gpu', '?')}")
    print()

# =============================================================================
# SQUAD STATUS
# =============================================================================

def get_squad_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all hosts in the squad."""
    status = {}
    
    for host in SSH.allowed_targets:
        host_status = {
            "reachable": False,
            "info": {}
        }
        
        if host == HOSTNAME:
            host_status["reachable"] = True
            host_status["info"] = {"note": "localhost"}
        else:
            success, message = test_connection(host)
            host_status["reachable"] = success
            
            if success:
                host_status["info"] = get_remote_info(host)
            else:
                host_status["info"] = {"error": message}
        
        status[host] = host_status
    
    return status

def print_squad_status():
    """Print status of all hosts in the squad."""
    print()
    header("Squad Status")
    
    for host in SSH.allowed_targets:
        host_config = get_host(host)
        ip = host_config.get("ip", "?") if host_config else "?"
        role = host_config.get("role", "?") if host_config else "?"
        
        if host == HOSTNAME:
            print(f"\n{COLORS.HOST}{host}{COLORS.RESET} ({ip}) - {role} [current]")
            print_system("  Status: localhost")
        else:
            success, message = test_connection(host)
            
            if success:
                print(f"\n{COLORS.SUCCESS}●{COLORS.RESET} {COLORS.HOST}{host}{COLORS.RESET} ({ip}) - {role}")
                info = get_remote_info(host)
                print(f"  Uptime: {info.get('uptime', '?')}")
                print(f"  Load:   {info.get('load', '?')}")
                print(f"  Memory: {info.get('memory', '?')}")
                if info.get('gpu') and info['gpu'] != 'no gpu':
                    print(f"  GPU:    {info.get('gpu', '?')}")
            else:
                print(f"\n{COLORS.ERROR}●{COLORS.RESET} {COLORS.HOST}{host}{COLORS.RESET} ({ip}) - {role}")
                print(f"  {COLORS.ERROR}{message}{COLORS.RESET}")
    
    print()
