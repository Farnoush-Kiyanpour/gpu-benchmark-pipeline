#!/usr/bin/env python3
"""
Phase 04: Collect and standardize results from all tool×target combinations.
Parses tool outputs (SDF, SMI, etc.), canonicalizes SMILES, and writes
standardized {tool}_{target}_final.smi files.

Usage:
    python 04_collect_results.py                           # Collect all results
    python 04_collect_results.py --tools TargetDiff         # Collect specific tool
    python 04_collect_results.py --targets EGFR             # Collect for specific target
"""
import os
import sys
import json
import glob
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


def collect_tool_target(tool_name: str, tool_config: dict, target: dict,
                         results_dir: str, output_dir: str, bench_root: str) -> int:
    """Collect molecules from a single tool×target combination.

    Returns number of unique molecules collected.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from tool_wrappers import get_wrapper
    from utils.smiles_standardizer import standardize_molecules, write_standard_smi
    from utils.error_handler import ErrorHandler, ToolStatus

    target_name = target['name']
    tool_output_dir = os.path.join(results_dir, tool_name, target_name)

    if not os.path.isdir(tool_output_dir):
        logger.warning(f"[{tool_name}/{target_name}] Output directory not found")
        return 0

    # Get wrapper for output parsing
    wrapper = get_wrapper(tool_name, tool_config, bench_root)

    # Parse output
    if wrapper:
        try:
            raw_molecules = wrapper.parse_output(tool_output_dir)
        except Exception as e:
            logger.error(f"[{tool_name}/{target_name}] Output parsing failed: {e}")
            # Fallback: try generic parsing
            from utils.smiles_standardizer import parse_output as parse_out
            output_pattern = tool_config.get('output_path', '{output_dir}/*.sdf').format(output_dir=tool_output_dir)
            raw_molecules = parse_out(output_pattern, tool_config.get('output_format', 'sdf'))
    else:
        # Generic parsing
        from utils.smiles_standardizer import parse_output as parse_out
        output_pattern = tool_config.get('output_path', '{output_dir}/*.sdf').format(output_dir=tool_output_dir)
        raw_molecules = parse_out(output_pattern, tool_config.get('output_format', 'sdf'))

    if not raw_molecules:
        logger.warning(f"[{tool_name}/{target_name}] No molecules found in output")
        return 0

    logger.info(f"[{tool_name}/{target_name}] Parsed {len(raw_molecules)} raw molecules")

    # Standardize: canonicalize, remove salts, filter by size, deduplicate
    standardized = standardize_molecules(raw_molecules, remove_salt=True, min_atoms=3, max_atoms=100)

    # Extract Vina scores if available (from AutoGrow4 ranked files)
    vina_scores = None
    if tool_name == 'AutoGrow4':
        vina_scores = []
        run_dir = os.path.join(tool_output_dir, 'Run_0')
        if not os.path.isdir(run_dir):
            run_dir = tool_output_dir

        score_map = {}
        for ranked_file in sorted(glob.glob(os.path.join(run_dir, 'generation_*/generation_*_ranked.smi'))):
            with open(ranked_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        try:
                            score_map[parts[0]] = float(parts[2])
                        except (ValueError, IndexError):
                            pass

        vina_scores = [score_map.get(smiles, None) for smiles, _ in standardized]

    # Write standardized output
    os.makedirs(output_dir, exist_ok=True)
    final_smi_path = os.path.join(output_dir, f"{tool_name}_{target_name}_final.smi")
    write_standard_smi(final_smi_path, standardized, vina_scores)

    logger.info(f"[{tool_name}/{target_name}] Collected {len(standardized)} unique molecules -> {final_smi_path}")
    return len(standardized)


def main():
    parser = argparse.ArgumentParser(description="Collect and standardize results from all tools")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--targets-file', type=str, default=None)
    parser.add_argument('--tools', type=str, nargs='+', default=None)
    parser.add_argument('--targets', type=str, nargs='+', default=None)
    parser.add_argument('--bench-root', type=str, default='.')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory with raw tool outputs')
    parser.add_argument('--output-dir', type=str, default='collected_results', help='Output directory for standardized SMILES')
    parser.add_argument('--status-file', type=str, default='benchmark_status.json')
    args = parser.parse_args()

    config = load_config(args.config)
    targets = load_targets(args.targets_file)
    from utils.error_handler import ErrorHandler, ToolStatus
    handler = ErrorHandler(args.status_file)

    tools = {k: v for k, v in config.items() if not k.startswith('_')}
    if args.tools:
        tools = {k: v for k, v in tools.items() if k in args.tools}

    if args.targets:
        targets = [t for t in targets if t['name'] in args.targets]

    total = len(tools) * len(targets)
    logger.info(f"Collecting results for {len(tools)} tools × {len(targets)} targets = {total} combinations...")

    results_summary = {}
    success_count = 0
    fail_count = 0
    total_molecules = 0

    for tool_name, tool_config in tools.items():
        for target in targets:
            target_name = target['name']
            job_key = f"{tool_name}_{target_name}"

            # Check job status
            job_status = handler.get_job_status(tool_name, target_name)
            if job_status not in [ToolStatus.INFERENCE_OK, ToolStatus.NO_OUTPUT, None]:
                logger.info(f"[{job_key}] Job status={job_status}, skipping collection")
                continue

            count = collect_tool_target(tool_name, tool_config, target,
                                         args.results_dir, args.output_dir, args.bench_root)

            results_summary[job_key] = count
            total_molecules += count

            if count > 0:
                success_count += 1
            else:
                fail_count += 1

    # Save summary
    summary_path = os.path.join(args.output_dir, 'collection_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"\nCollection complete: {success_count} succeeded, {fail_count} failed")
    logger.info(f"Total unique molecules collected: {total_molecules}")
    logger.info(f"Summary saved to {summary_path}")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
