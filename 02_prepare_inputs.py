#!/usr/bin/env python3
"""
Phase 02: Prepare tool-specific input files for all tool×target combinations.
Uses the tool wrappers to convert target data into each tool's expected format.

Usage:
    python 02_prepare_inputs.py                           # Prepare all inputs
    python 02_prepare_inputs.py --tools TargetDiff DrugGPT # Prepare specific tools
    python 02_prepare_inputs.py --targets EGFR BRAF        # Prepare for specific targets
"""
import os
import sys
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'tools_config.json')
    with open(config_path) as f:
        return json.load(f)


def load_targets(targets_path: str = None) -> list:
    if targets_path is None:
        targets_path = os.path.join(os.path.dirname(__file__), 'config', 'targets.json')
    with open(targets_path) as f:
        data = json.load(f)
    return data.get('targets', data)


def main():
    parser = argparse.ArgumentParser(description="Prepare tool-specific inputs for all targets")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--targets-file', type=str, default=None)
    parser.add_argument('--tools', type=str, nargs='+', default=None)
    parser.add_argument('--targets', type=str, nargs='+', default=None)
    parser.add_argument('--bench-root', type=str, default='.', help='Benchmark root directory')
    parser.add_argument('--output-dir', type=str, default='prepared_inputs', help='Output directory for prepared inputs')
    parser.add_argument('--status-file', type=str, default='benchmark_status.json')
    args = parser.parse_args()

    config = load_config(args.config)
    targets = load_targets(args.targets_file)
    from utils.error_handler import ErrorHandler, ToolStatus
    handler = ErrorHandler(args.status_file)

    # Import wrapper registry
    sys.path.insert(0, os.path.dirname(__file__))
    from tool_wrappers import get_wrapper

    tools = {k: v for k, v in config.items() if not k.startswith('_')}
    if args.tools:
        tools = {k: v for k, v in tools.items() if k in args.tools}

    if args.targets:
        targets = [t for t in targets if t['name'] in args.targets]

    logger.info(f"Preparing inputs for {len(tools)} tools × {len(targets)} targets = {len(tools)*len(targets)} combinations...")

    success_count = 0
    fail_count = 0

    for tool_name, tool_config in tools.items():
        # Skip tools that failed env or weights
        if not handler.is_tool_ready(tool_name):
            env_status = handler.get_tool_status(tool_name, "env")
            weights_status = handler.get_tool_status(tool_name, "weights")
            logger.warning(f"[{tool_name}] Skipping input prep — env={env_status}, weights={weights_status}")
            handler.set_tool_status(tool_name, "input", ToolStatus.SKIPPED, "env or weights failed")
            continue

        wrapper = get_wrapper(tool_name, tool_config, args.bench_root)
        if wrapper is None:
            logger.error(f"[{tool_name}] No wrapper available, skipping")
            handler.set_tool_status(tool_name, "input", ToolStatus.INPUT_FAILED, "no wrapper")
            fail_count += 1
            continue

        tool_success = True
        for target in targets:
            target_name = target['name']
            input_dir = os.path.join(args.output_dir, tool_name, target_name)

            try:
                if os.path.exists(input_dir) and os.listdir(input_dir):
                    logger.info(f"[{tool_name}/{target_name}] Inputs already prepared")
                    success_count += 1
                    continue

                os.makedirs(input_dir, exist_ok=True)
                inputs = wrapper.prepare_input(target, input_dir)

                # Save input manifest
                manifest_path = os.path.join(input_dir, 'input_manifest.json')
                with open(manifest_path, 'w') as f:
                    json.dump({
                        'tool': tool_name,
                        'target': target_name,
                        'input_files': inputs,
                        'input_format': wrapper.input_format,
                    }, f, indent=2)

                logger.info(f"[{tool_name}/{target_name}] Inputs prepared: {list(inputs.keys())}")
                success_count += 1

            except Exception as e:
                logger.error(f"[{tool_name}/{target_name}] Input prep failed: {e}")
                handler.set_job_status(tool_name, target_name, ToolStatus.INPUT_FAILED, str(e))
                tool_success = False
                fail_count += 1

        if tool_success:
            handler.set_tool_status(tool_name, "input", ToolStatus.INPUT_OK)

    logger.info(f"Input preparation complete: {success_count} succeeded, {fail_count} failed")
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
