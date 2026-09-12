#!/usr/bin/env python3
"""
Phase 01: Download pre-trained weights for all tools.
Handles HuggingFace, Google Drive, and direct URL downloads.

Usage:
    python 01_download_weights.py                    # Download all weights
    python 01_download_weights.py --tools TargetDiff  # Download specific tool weights
    python 01_download_weights.py --dry-run           # Print commands without executing
"""
import os
import sys
import json
import argparse
import subprocess
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'tools_config.json')
    with open(config_path) as f:
        return json.load(f)


def download_huggingface(url: str, dest_path: str, repo_path: str, max_retries: int = 3) -> bool:
    """Download model from HuggingFace Hub."""
    # Extract repo ID from URL (e.g., https://huggingface.co/GenSI/MolCRAFT -> GenSI/MolCRAFT)
    repo_id = url.split("huggingface.co/")[-1] if "huggingface.co/" in url else url

    for attempt in range(max_retries):
        try:
            # Use huggingface_hub CLI or Python API
            py_script = (
                f"from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id='{repo_id}', "
                f"local_dir='{dest_path}', local_dir_use_symlinks=False)"
            )
            cmd = f"cd {repo_path} && python -c \"{py_script}\""
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                return True
            logger.warning(f"HuggingFace download attempt {attempt+1} failed: {result.stderr[:300]}")
        except Exception as e:
            logger.warning(f"HuggingFace download attempt {attempt+1} error: {e}")
        time.sleep(5 * (attempt + 1))
    return False


def download_gdrive(url: str, dest_path: str, repo_path: str, max_retries: int = 3) -> bool:
    """Download from Google Drive using gdown."""
    # Extract file/folder ID from URL
    file_id = None
    if '/file/d/' in url:
        file_id = url.split('/file/d/')[1].split('/')[0]
    elif '/folders/' in url:
        file_id = url.split('/folders/')[1].split('?')[0]
    elif 'id=' in url:
        file_id = url.split('id=')[1].split('&')[0]

    if not file_id:
        logger.error(f"Could not extract Google Drive ID from {url}")
        return False

    for attempt in range(max_retries):
        try:
            is_folder = '/folders/' in url
            if is_folder:
                cmd = f"cd {repo_path} && mkdir -p {dest_path} && gdown --folder {url} -O {dest_path}"
            else:
                cmd = f"cd {repo_path} && mkdir -p {os.path.dirname(dest_path)} && gdown {url} -O {dest_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                return True
            logger.warning(f"gdown attempt {attempt+1} failed: {result.stderr[:300]}")
        except Exception as e:
            logger.warning(f"gdown attempt {attempt+1} error: {e}")
        time.sleep(5 * (attempt + 1))
    return False


def download_wget(url: str, dest_path: str, repo_path: str, max_retries: int = 3) -> bool:
    """Download from direct URL using wget."""
    for attempt in range(max_retries):
        try:
            cmd = f"cd {repo_path} && mkdir -p {os.path.dirname(dest_path)} && wget -q --show-progress -O {dest_path} '{url}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                return True
            logger.warning(f"wget attempt {attempt+1} failed")
        except Exception as e:
            logger.warning(f"wget attempt {attempt+1} error: {e}")
        time.sleep(5 * (attempt + 1))
    return False


def download_tool_weights(tool_name: str, tool_config: dict, bench_root: str, dry_run: bool = False) -> bool:
    """Download weights for a single tool.

    Returns True if weights are available (downloaded or already present).
    """
    weights = tool_config.get('weights', {})
    url = weights.get('url')
    dest_path = weights.get('path')
    method = weights.get('method', 'wget')

    # No weights needed
    if url is None or dest_path is None:
        logger.info(f"[{tool_name}] No weights required, skipping")
        return True

    repo_path = os.path.join(bench_root, tool_config.get('repo_path', ''))
    full_dest = os.path.join(repo_path, dest_path)

    # Check if weights already exist
    if os.path.exists(full_dest) and os.path.getsize(full_dest) > 0:
        logger.info(f"[{tool_name}] Weights already present at {dest_path}")
        return True

    # Check if it's a directory (might have partial download)
    if os.path.isdir(full_dest) and any(os.scandir(full_dest)):
        logger.info(f"[{tool_name}] Weights directory already has files at {dest_path}")
        return True

    logger.info(f"[{tool_name}] Downloading weights from {url}...")

    if dry_run:
        print(f"[DRY RUN] {tool_name}: {method} {url} -> {dest_path}")
        return True

    if method == 'huggingface':
        return download_huggingface(url, full_dest, repo_path)
    elif method == 'gdown':
        return download_gdrive(url, full_dest, repo_path)
    elif method == 'wget':
        return download_wget(url, full_dest, repo_path)
    else:
        logger.error(f"[{tool_name}] Unknown download method: {method}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained weights for all tools")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--tools', type=str, nargs='+', default=None)
    parser.add_argument('--bench-root', type=str, default='.', help='Benchmark root directory')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--status-file', type=str, default='benchmark_status.json')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of weight downloads to run concurrently (default: 8). '
                             'Downloads are network I/O-bound and fully independent per tool, '
                             'so running them serially wastes wall time waiting on the network.')
    args = parser.parse_args()

    config = load_config(args.config)
    from utils.error_handler import ErrorHandler, ToolStatus
    handler = ErrorHandler(args.status_file)

    tools = {k: v for k, v in config.items() if not k.startswith('_')}
    if args.tools:
        tools = {k: v for k, v in tools.items() if k in args.tools}

    # Filter out tools whose env setup already failed before dispatching any downloads
    runnable_tools = {}
    for tool_name, tool_config in tools.items():
        env_status = handler.get_tool_status(tool_name, "env")
        if env_status == ToolStatus.ENV_FAILED:
            logger.warning(f"[{tool_name}] Skipping weights — env setup failed")
            handler.set_tool_status(tool_name, "weights", ToolStatus.SKIPPED, "env failed")
        else:
            runnable_tools[tool_name] = tool_config

    logger.info(f"Downloading weights for {len(runnable_tools)} tools "
                f"({args.workers} concurrent)...")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_tool = {
            executor.submit(download_tool_weights, tool_name, tool_config,
                             args.bench_root, args.dry_run): tool_name
            for tool_name, tool_config in runnable_tools.items()
        }
        for future in as_completed(future_to_tool):
            tool_name = future_to_tool[future]
            tool_config = runnable_tools[tool_name]
            try:
                ok = future.result()
            except Exception as e:
                logger.error(f"[{tool_name}] Weight download raised an exception: {e}")
                ok = False
            if ok:
                handler.set_tool_status(tool_name, "weights", ToolStatus.WEIGHTS_OK)
                success_count += 1
            else:
                handler.set_tool_status(tool_name, "weights", ToolStatus.WEIGHTS_FAILED,
                                         f"download failed from {tool_config.get('weights', {}).get('url', '?')}")
                fail_count += 1

    logger.info(f"Weight download complete: {success_count} succeeded, {fail_count} failed")
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
