#!/usr/bin/env python3
"""
================================================================================
MAIN ORCHESTRATOR: Complete 36-Tool GPU Benchmark Pipeline
================================================================================

Runs the full benchmark pipeline in order:
  Phase 0: Setup conda environments for all 36 tools
  Phase 1: Download pre-trained weights
  Phase 2: Prepare tool-specific inputs for all targets
  Phase 3: Submit and monitor Slurm inference jobs (36 tools × 10 targets)
  Phase 4: Collect and standardize results
  Phase 5: Evaluate all molecules with full metric suite
  Phase 6: Generate comparison tables, charts, and report

Each phase checks the status file and skips already-completed work,
making the pipeline resumable. Errors in individual tools or targets
are logged and skipped without stopping the entire pipeline.

Usage:
  python run_benchmark.py                              # Full pipeline
  python run_benchmark.py --smoke-test                  # Quick test (5 mols, 1 target)
  python run_benchmark.py --phases 3 4 5 6              # Run specific phases only
  python run_benchmark.py --tools TargetDiff DrugGPT    # Run specific tools only
  python run_benchmark.py --targets EGFR BRAF           # Run specific targets only
  python run_benchmark.py --num-samples 100             # Molecules per target (default: 100)
  python run_benchmark.py --max-concurrent 4            # Max concurrent Slurm jobs
  python run_benchmark.py --skip-docking                # Skip Vina docking in evaluation
  python run_benchmark.py --dry-run                     # Print commands without executing

Prerequisites:
  - Slurm workload manager
  - Conda/Miniconda installed
  - All 36 tool repos cloned under bench_repos/
  - Target data prepared under targets/ (PDB, sequences, reference ligands)
  - AutoDock Vina installed
  - MGLTools/ADFRSuite installed (for PDBQT conversion)

Environment variables (optional):
  SLURM_PARTITION    - Slurm partition for GPU jobs (default: 'gpu')
  CONDA_INIT         - Path to conda.sh (default: '~/miniconda3/etc/profile.d/conda.sh')
  BENCH_ROOT         - Benchmark root directory (default: current directory)
================================================================================
"""
import os
import sys
import time
import json
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Phase definitions
# =============================================================================

PHASES = {
    0: ("Setup conda environments", "00_setup_environments.py"),
    1: ("Download pre-trained weights", "01_download_weights.py"),
    2: ("Prepare tool-specific inputs", "02_prepare_inputs.py"),
    3: ("Submit and monitor inference jobs", "03_submit_inference.py"),
    4: ("Collect and standardize results", "04_collect_results.py"),
    5: ("Evaluate all molecules", "05_evaluate.py"),
    6: ("Generate report and charts", "06_generate_report.py"),
}


def run_phase(phase_num: int, extra_args: list, bench_root: str, dry_run: bool = False) -> bool:
    """Run a single phase of the pipeline.

    Args:
        phase_num: Phase number (0-6)
        extra_args: Extra arguments to pass to the phase script
        bench_root: Benchmark root directory
        dry_run: If True, print the command without executing

    Returns:
        True if phase succeeded, False otherwise
    """
    description, script = PHASES[phase_num]
    script_path = os.path.join(os.path.dirname(__file__), script)

    if not os.path.exists(script_path):
        logger.error(f"Phase {phase_num}: Script not found: {script_path}")
        return False

    cmd = [sys.executable, script_path, '--bench-root', bench_root] + extra_args

    logger.info(f"\n{'='*70}")
    logger.info(f"PHASE {phase_num}: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*70}\n")

    if dry_run:
        print(f"[DRY RUN] Phase {phase_num}: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            logger.info(f"Phase {phase_num} completed successfully")
            return True
        else:
            logger.warning(f"Phase {phase_num} completed with errors (exit code {result.returncode})")
            # Phases 0-3 may have partial failures (some tools fail) — continue
            # Phases 4-6 should not fail entirely
            if phase_num <= 3:
                return True  # Continue despite partial failures
            return False
    except Exception as e:
        logger.error(f"Phase {phase_num} failed with exception: {e}")
        return False


def check_prerequisites(bench_root: str) -> list:
    """Check that required directories and files exist.

    Returns list of missing items (empty if all OK).
    """
    missing = []

    # Check config files
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    for fname in ['tools_config.json', 'targets.json', 'slurm_template.sh']:
        if not os.path.exists(os.path.join(config_dir, fname)):
            missing.append(f"config/{fname}")

    # Check tool wrappers
    wrapper_dir = os.path.join(os.path.dirname(__file__), 'tool_wrappers')
    if not os.path.exists(os.path.join(wrapper_dir, '__init__.py')):
        missing.append("tool_wrappers/__init__.py")

    # Check utils
    utils_dir = os.path.join(os.path.dirname(__file__), 'utils')
    if not os.path.exists(os.path.join(utils_dir, '__init__.py')):
        missing.append("utils/__init__.py")

    # Check bench repos (if bench_root is set)
    repos_dir = os.path.join(bench_root, 'bench_repos')
    if not os.path.isdir(repos_dir):
        missing.append(f"{repos_dir} (tool repositories)")

    # Check targets
    targets_dir = os.path.join(bench_root, 'targets')
    if not os.path.isdir(targets_dir):
        missing.append(f"{targets_dir} (target data)")

    # Check Slurm
    result = subprocess.run('which sbatch', shell=True, capture_output=True)
    if result.returncode != 0:
        missing.append("sbatch (Slurm not found in PATH)")

    # Check conda
    result = subprocess.run('which conda', shell=True, capture_output=True)
    if result.returncode != 0:
        missing.append("conda (Conda not found in PATH)")

    return missing


def print_status_summary(status_file: str):
    """Print a summary of tool/job statuses from the status file."""
    if not os.path.exists(status_file):
        logger.info("No status file found yet")
        return

    with open(status_file) as f:
        status = json.load(f)

    tools = status.get('tools', {})
    jobs = status.get('jobs', {})

    # Tool status summary
    phase_counts = {}
    for tool, phases in tools.items():
        for phase, info in phases.items():
            s = info.get('status', 'unknown')
            phase_counts.setdefault(phase, {})
            phase_counts[phase][s] = phase_counts[phase].get(s, 0) + 1

    logger.info("\nTool Status Summary:")
    for phase in sorted(phase_counts.keys()):
        counts = phase_counts[phase]
        total = sum(counts.values())
        parts = [f"{status}={count}" for status, count in sorted(counts.items())]
        logger.info(f"  {phase}: {total} tools ({', '.join(parts)})")

    # Job status summary
    job_counts = {}
    for job_key, info in jobs.items():
        s = info.get('status', 'unknown')
        job_counts[s] = job_counts.get(s, 0) + 1

    logger.info(f"\nJob Status Summary ({len(jobs)} total):")
    for status, count in sorted(job_counts.items()):
        logger.info(f"  {status}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete 36-Tool GPU Benchmark Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--phases', type=int, nargs='+', default=None,
                        help='Run specific phases only (0-6). Default: all phases')
    parser.add_argument('--tools', type=str, nargs='+', default=None,
                        help='Run specific tools only')
    parser.add_argument('--targets', type=str, nargs='+', default=None,
                        help='Run specific targets only')
    parser.add_argument('--bench-root', type=str, default='.',
                        help='Benchmark root directory')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of molecules per target (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for generation (default: 32)')
    parser.add_argument('--max-concurrent', type=int, default=4,
                        help='Max concurrent Slurm jobs (default: 4)')
    parser.add_argument('--skip-docking', action='store_true',
                        help='Skip Vina docking in evaluation phase')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Quick test: 5 molecules, 1 target, all phases')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--status-file', type=str, default='benchmark_status.json',
                        help='Status file path')
    parser.add_argument('--skip-checks', action='store_true',
                        help='Skip prerequisite checks')
    args = parser.parse_args()

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("COMPLETE 36-TOOL GPU BENCHMARK PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Bench root: {args.bench_root}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Max concurrent: {args.max_concurrent}")
    if args.tools:
        logger.info(f"Tools: {args.tools}")
    if args.targets:
        logger.info(f"Targets: {args.targets}")
    if args.smoke_test:
        logger.info("SMOKE TEST MODE: 5 molecules, 1 target")

    # Prerequisite checks
    if not args.skip_checks and not args.dry_run:
        missing = check_prerequisites(args.bench_root)
        if missing:
            logger.error("Missing prerequisites:")
            for m in missing:
                logger.error(f"  - {m}")
            logger.error("Fix these before running the pipeline, or use --skip-checks")
            return 1

    # Determine which phases to run
    if args.phases:
        phases_to_run = sorted(args.phases)
    else:
        phases_to_run = list(PHASES.keys())

    # Build common args for all phases
    common_args = []
    if args.tools:
        common_args.extend(['--tools'] + args.tools)
    if args.targets:
        common_args.extend(['--targets'] + args.targets)
    common_args.extend(['--status-file', args.status_file])

    if args.dry_run:
        common_args.append('--dry-run')

    # Run each phase
    phase_results = {}
    for phase_num in phases_to_run:
        phase_args = list(common_args)

        # Phase-specific args
        if phase_num == 3:  # Submit inference
            phase_args.extend([
                '--num-samples', str(args.num_samples),
                '--batch-size', str(args.batch_size),
                '--max-concurrent', str(args.max_concurrent),
            ])
            if args.smoke_test:
                phase_args.append('--smoke-test')
        elif phase_num == 5:  # Evaluate
            if args.skip_docking:
                phase_args.append('--skip-docking')

        success = run_phase(phase_num, phase_args, args.bench_root, args.dry_run)
        phase_results[phase_num] = success

        if not success and phase_num <= 3:
            logger.warning(f"Phase {phase_num} had errors, but continuing to next phase")
        elif not success:
            logger.error(f"Phase {phase_num} failed, stopping pipeline")
            break

        # Print status after each phase
        if not args.dry_run:
            print_status_summary(args.status_file)

    # Final summary
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total time: {elapsed/3600:.1f} hours ({elapsed:.0f}s)")

    for phase_num in phases_to_run:
        desc = PHASES[phase_num][0]
        status = "SUCCESS" if phase_results.get(phase_num, False) else "FAILED/SKIPPED"
        logger.info(f"  Phase {phase_num}: {desc} — {status}")

    if not args.dry_run:
        print_status_summary(args.status_file)
        logger.info(f"\nStatus file: {args.status_file}")
        logger.info(f"Pipeline log: benchmark_pipeline.log")
        logger.info(f"Results directory: results/")
        logger.info(f"Evaluations directory: evaluations/")
        logger.info(f"Report: report_benchmark.md")

    return 0 if all(phase_results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
