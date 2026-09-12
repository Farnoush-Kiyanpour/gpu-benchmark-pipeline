#!/usr/bin/env python3
"""
Phase 00: Create conda environments for all 36 tools.
Each tool gets its own conda env with the specified Python/PyTorch/CUDA versions.

Usage:
    python 00_setup_environments.py                    # Create all envs
    python 00_setup_environments.py --tools TargetDiff  # Create specific tool env
    python 00_setup_environments.py --dry-run           # Print commands without executing
"""
import os
import sys
import json
import argparse
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load tools configuration."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'tools_config.json')
    with open(config_path) as f:
        return json.load(f)


def build_conda_create_cmd(env_name: str, spec: dict) -> str:
    """Build conda create command from spec dict."""
    parts = [f"conda create -n {env_name} -y"]

    python_ver = spec.get('python', '3.8')
    parts.append(f"python={python_ver}")

    # PyTorch
    pytorch_ver = spec.get('pytorch')
    cuda_ver = spec.get('cuda')
    if pytorch_ver and cuda_ver:
        parts.append(f"pytorch={pytorch_ver} pytorch-cuda={cuda_ver} -c pytorch -c nvidia")
    elif pytorch_ver:
        parts.append(f"pytorch={pytorch_ver} -c pytorch")

    # Conda packages
    conda_pkgs = spec.get('conda', [])
    if conda_pkgs:
        parts.append(f"conda install -y {' '.join(conda_pkgs)} -c conda-forge")

    return ' && '.join(parts)


def build_pip_install_cmd(env_name: str, spec: dict) -> str:
    """Build pip install command for pip-only packages."""
    pip_pkgs = spec.get('pip', [])
    if not pip_pkgs:
        return ""

    # Map common package names to their install commands
    pkg_cmds = []
    for pkg in pip_pkgs:
        if pkg == 'torch-geometric':
            pkg_cmds.append("pip install torch-geometric -c pyg 2>/dev/null || pip install torch-geometric")
        elif pkg == 'pyg':
            pkg_cmds.append("pip install torch-geometric -c pyg 2>/dev/null || pip install torch-geometric")
        elif pkg == 'rdkit':
            pkg_cmds.append("pip install rdkit 2>/dev/null || conda install -y rdkit -c conda-forge")
        elif pkg == 'openbabel':
            pkg_cmds.append("pip install openbabel-wheel 2>/dev/null || conda install -y openbabel -c conda-forge")
        elif pkg == 'biopython':
            pkg_cmds.append("pip install biopython")
        elif pkg == 'transformers':
            pkg_cmds.append("pip install transformers")
        elif pkg == 'pytorch-lightning':
            pkg_cmds.append("pip install pytorch-lightning")
        elif pkg == 'posecheck':
            pkg_cmds.append("pip install posecheck 2>/dev/null || echo 'posecheck not available, skipping'")
        elif pkg == 'fair-esm':
            pkg_cmds.append("pip install fair-esm")
        elif pkg == 'fairseq':
            pkg_cmds.append("pip install fairseq 2>/dev/null || echo 'fairseq install failed, may need manual setup'")
        elif pkg == 'e3nn':
            pkg_cmds.append("pip install e3nn")
        elif pkg == 'wandb':
            pkg_cmds.append("pip install wandb")
        elif pkg == 'datasets':
            pkg_cmds.append("pip install datasets")
        else:
            pkg_cmds.append(f"pip install {pkg}")

    return f"conda run -n {env_name} " + ' && '.join(pkg_cmds)


def setup_tool_env(tool_name: str, tool_config: dict, dry_run: bool = False) -> bool:
    """Create conda environment for a single tool.

    Returns True if env was created successfully (or already exists).
    """
    env_name = tool_config.get('conda_env', tool_name.lower())
    spec = tool_config.get('conda_spec', {})

    # Check if env already exists
    check = subprocess.run(f"conda env list | grep '{env_name} '", shell=True, capture_output=True, text=True)
    if check.returncode == 0:
        logger.info(f"[{tool_name}] Conda env '{env_name}' already exists, skipping")
        return True

    # Build and run conda create command
    create_cmd = build_conda_create_cmd(env_name, spec)
    pip_cmd = build_pip_install_cmd(env_name, spec)

    full_cmd = create_cmd
    if pip_cmd:
        full_cmd += f" && {pip_cmd}"

    # Add extra setup commands
    extra_setup = tool_config.get('extra_setup', [])
    for setup_cmd in extra_setup:
        full_cmd += f" && {setup_cmd}"

    logger.info(f"[{tool_name}] Creating conda env '{env_name}'...")
    logger.debug(f"  Command: {full_cmd[:200]}...")

    if dry_run:
        print(f"[DRY RUN] {tool_name}: {full_cmd}")
        return True

    try:
        # conda create with pytorch+cuda routinely takes 20-60+ min with the
        # classic solver (especially under concurrent env creation contending
        # for the same package cache), so a 10-min timeout was killing many
        # legitimately-in-progress installs. 1 hour, still bounded so a truly
        # stuck solve doesn't block the whole phase forever.
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            logger.info(f"[{tool_name}] Environment created successfully")
            return True
        else:
            logger.error(f"[{tool_name}] Environment creation failed: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"[{tool_name}] Environment creation timed out (60 min limit)")
        return False
    except Exception as e:
        logger.error(f"[{tool_name}] Environment creation error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Create conda environments for all benchmark tools")
    parser.add_argument('--config', type=str, default=None, help='Path to tools_config.json')
    parser.add_argument('--tools', type=str, nargs='+', default=None, help='Specific tools to set up')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--status-file', type=str, default='benchmark_status.json', help='Status file path')
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of conda envs to build concurrently (default: 6). '
                             'The 36 tool envs are fully independent, so building them serially '
                             'wastes wall time; keep this modest to avoid saturating the login '
                             'node / package cache locks on a shared cluster.')
    args = parser.parse_args()

    config = load_config(args.config)
    from utils.error_handler import ErrorHandler, ToolStatus
    handler = ErrorHandler(args.status_file)

    # Filter out _comment key
    tools = {k: v for k, v in config.items() if not k.startswith('_')}
    if args.tools:
        tools = {k: v for k, v in tools.items() if k in args.tools}

    logger.info(f"Setting up conda environments for {len(tools)} tools "
                f"({args.workers} concurrent)...")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_tool = {
            executor.submit(setup_tool_env, tool_name, tool_config, args.dry_run): tool_name
            for tool_name, tool_config in tools.items()
        }
        for future in as_completed(future_to_tool):
            tool_name = future_to_tool[future]
            try:
                ok = future.result()
            except Exception as e:
                logger.error(f"[{tool_name}] Environment creation raised an exception: {e}")
                ok = False
            if ok:
                handler.set_tool_status(tool_name, "env", ToolStatus.ENV_OK)
                success_count += 1
            else:
                handler.set_tool_status(tool_name, "env", ToolStatus.ENV_FAILED, "conda create failed")
                fail_count += 1

    logger.info(f"Environment setup complete: {success_count} succeeded, {fail_count} failed")
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
