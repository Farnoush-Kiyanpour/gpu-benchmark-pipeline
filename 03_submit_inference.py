#!/usr/bin/env python3
"""
Phase 03: Submit Slurm jobs for all tool×target combinations.
Generates Slurm job scripts from the template and submits them.
Monitors job status and handles retries for OOM/crash/timeout.

Usage:
    python 03_submit_inference.py                           # Submit all jobs
    python 03_submit_inference.py --tools TargetDiff         # Submit specific tool
    python 03_submit_inference.py --targets EGFR             # Submit for specific target
    python 03_submit_inference.py --num-samples 100          # Molecules per target
    python 03_submit_inference.py --max-concurrent 4         # Max concurrent jobs
    python 03_submit_inference.py --smoke-test               # Quick test: 5 mols, 1 target
"""
import os
import sys
import json
import time
import argparse
import subprocess
import logging
from pathlib import Path

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


def load_slurm_template(template_path: str = None) -> str:
    if template_path is None:
        template_path = os.path.join(os.path.dirname(__file__), 'config', 'slurm_template.sh')
    with open(template_path) as f:
        return f.read()


def generate_job_script(template: str, tool_name: str, tool_config: dict,
                         target: dict, inputs: dict, output_dir: str,
                         num_samples: int, batch_size: int, bench_root: str,
                         log_dir: str, checkpoint_dir: str) -> str:
    """Generate a Slurm job script for a single tool×target combination."""
    job_name = f"{tool_name}_{target['name']}"

    # Get wrapper for command building
    sys.path.insert(0, os.path.dirname(__file__))
    from tool_wrappers import get_wrapper
    wrapper = get_wrapper(tool_name, tool_config, bench_root)

    if wrapper is None:
        # Fallback: use config template
        cmd_template = tool_config.get('inference_cmd', 'echo "No command"')
        cmd = cmd_template.format(
            pdb_path=inputs.get('pdb_path', ''),
            fasta_path=inputs.get('fasta_path', ''),
            ref_sdf=inputs.get('ref_sdf', ''),
            config_path=inputs.get('config_path', ''),
            target_name=target['name'],
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=output_dir,
            checkpoint=tool_config.get('weights', {}).get('path', ''),
        )
    else:
        cmd = wrapper.build_command(inputs, output_dir, num_samples, batch_size)

    # Fill in template
    script = template.format(
        job_name=job_name,
        partition=os.environ.get('SLURM_PARTITION', 'gpu'),
        num_gpus=1 if tool_config.get('gpu_required', True) else 0,
        cpus=4,
        memory=tool_config.get('gpu_memory_gb', 8) * 4,  # 4x GPU memory for system RAM
        time_limit=f"{tool_config.get('timeout_hours', 2)}:00:00",
        log_dir=log_dir,
        conda_init_path=os.environ.get('CONDA_INIT', '~/miniconda3/etc/profile.d/conda.sh'),
        conda_env=tool_config.get('conda_env', tool_name.lower()),
        repo_path=os.path.join(bench_root, tool_config.get('repo_path', '')),
        extra_setup='\n'.join(tool_config.get('extra_setup', [])),
        inference_cmd=cmd,
        checkpoint_dir=checkpoint_dir,
    )

    return script


def submit_job(script_path: str) -> str:
    """Submit a Slurm job and return the job ID."""
    result = subprocess.run(f'sbatch {script_path}', shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        # Parse job ID from "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        return job_id
    logger.error(f"Job submission failed: {result.stderr}")
    return None


def check_job_completed(job_id: str) -> str:
    """Check Slurm job status. Returns 'completed', 'failed', 'running', or 'unknown'."""
    if job_id is None:
        return 'unknown'
    result = subprocess.run(f'sacct -j {job_id} --format=State --noheader', shell=True, capture_output=True, text=True)
    state = result.stdout.strip().split('\n')[0].strip() if result.stdout.strip() else ''
    if 'COMPLETED' in state:
        return 'completed'
    elif 'FAILED' in state or 'CANCELLED' in state or 'TIMEOUT' in state:
        return 'failed'
    elif 'RUNNING' in state or 'PENDING' in state:
        return 'running'
    return 'unknown'


def count_concurrent_jobs() -> int:
    """Count currently running/pending Slurm jobs for this user."""
    result = subprocess.run('squeue --me --states=RUNNING,PENDING --noheader | wc -l', shell=True, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Submit Slurm jobs for all tool×target combinations")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--targets-file', type=str, default=None)
    parser.add_argument('--tools', type=str, nargs='+', default=None)
    parser.add_argument('--targets', type=str, nargs='+', default=None)
    parser.add_argument('--bench-root', type=str, default='.')
    parser.add_argument('--num-samples', type=int, default=100, help='Molecules per target')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--max-concurrent', type=int, default=4, help='Max concurrent Slurm jobs')
    parser.add_argument('--smoke-test', action='store_true', help='Quick test: 5 mols, 1 target only')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--job-dir', type=str, default='slurm_jobs', help='Directory for job scripts')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--status-file', type=str, default='benchmark_status.json')
    parser.add_argument('--poll-interval', type=int, default=60, help='Seconds between status checks')
    parser.add_argument('--no-wait', action='store_true', help='Submit jobs and exit without waiting')
    args = parser.parse_args()

    config = load_config(args.config)
    targets = load_targets(args.targets_file)
    template = load_slurm_template()
    from utils.error_handler import ErrorHandler, ToolStatus
    handler = ErrorHandler(args.status_file)

    sys.path.insert(0, os.path.dirname(__file__))
    from tool_wrappers import get_wrapper

    tools = {k: v for k, v in config.items() if not k.startswith('_')}
    if args.tools:
        tools = {k: v for k, v in tools.items() if k in args.tools}

    if args.smoke_test:
        args.num_samples = 5
        targets = targets[:1]
        args.no_wait = False

    if args.targets:
        targets = [t for t in targets if t['name'] in args.targets]

    # Create directories
    for d in [args.output_dir, args.job_dir, args.log_dir, args.checkpoint_dir]:
        os.makedirs(d, exist_ok=True)

    total_combos = len(tools) * len(targets)
    logger.info(f"Submitting {total_combos} jobs ({len(tools)} tools × {len(targets)} targets)...")
    logger.info(f"  Molecules per target: {args.num_samples}")
    logger.info(f"  Max concurrent jobs: {args.max_concurrent}")

    submitted_jobs = {}  # job_id -> (tool, target, attempt)
    job_scripts = {}     # (tool, target) -> script_path

    # Submit jobs
    for tool_name, tool_config in tools.items():
        # Skip tools that failed setup
        if not handler.is_tool_ready(tool_name):
            input_status = handler.get_tool_status(tool_name, "input")
            if input_status != ToolStatus.INPUT_OK:
                logger.warning(f"[{tool_name}] Skipping — setup not complete")
                continue

        wrapper = get_wrapper(tool_name, tool_config, args.bench_root)

        for target in targets:
            target_name = target['name']
            job_key = f"{tool_name}_{target_name}"

            # Check if already completed
            existing_status = handler.get_job_status(tool_name, target_name)
            if existing_status == ToolStatus.INFERENCE_OK:
                logger.info(f"[{job_key}] Already completed, skipping")
                continue

            # Load input manifest
            input_dir = os.path.join('prepared_inputs', tool_name, target_name)
            manifest_path = os.path.join(input_dir, 'input_manifest.json')
            if not os.path.exists(manifest_path):
                logger.warning(f"[{job_key}] No input manifest found, skipping")
                handler.set_job_status(tool_name, target_name, ToolStatus.INPUT_FAILED, "no manifest")
                continue

            with open(manifest_path) as f:
                manifest = json.load(f)
            inputs = manifest.get('input_files', {})

            # Output directory
            output_dir = os.path.join(args.output_dir, tool_name, target_name)
            os.makedirs(output_dir, exist_ok=True)

            # Generate job script
            script = generate_job_script(
                template, tool_name, tool_config, target, inputs,
                output_dir, args.num_samples, args.batch_size,
                args.bench_root, args.log_dir, args.checkpoint_dir
            )

            script_path = os.path.join(args.job_dir, f"{job_key}.sh")
            with open(script_path, 'w') as f:
                f.write(script)
            os.chmod(script_path, 0o755)

            job_scripts[job_key] = script_path

            # Wait for slot if at max concurrent
            while count_concurrent_jobs() >= args.max_concurrent:
                logger.info(f"  Waiting for job slot ({count_concurrent_jobs()}/{args.max_concurrent} running)...")
                time.sleep(args.poll_interval)

            # Submit job
            job_id = submit_job(script_path)
            if job_id:
                submitted_jobs[job_id] = (tool_name, target_name, 1)
                handler.set_job_status(tool_name, target_name, ToolStatus.INFERENCE_RUNNING, f"job_id={job_id}")
                logger.info(f"[{job_key}] Submitted as job {job_id}")
            else:
                handler.set_job_status(tool_name, target_name, ToolStatus.INFERENCE_FAILED, "submission failed")
                logger.error(f"[{job_key}] Submission failed")

            time.sleep(2)  # Small delay between submissions

    logger.info(f"Submitted {len(submitted_jobs)} jobs")

    if args.no_wait:
        logger.info("Jobs submitted. Use --no-wait=false to monitor until completion.")
        return 0

    # Monitor jobs
    logger.info("Monitoring jobs until completion...")
    completed = set()
    failed = {}

    while len(completed) < len(submitted_jobs):
        time.sleep(args.poll_interval)

        for job_id, (tool_name, target_name, attempt) in list(submitted_jobs.items()):
            if job_id in completed:
                continue

            status = check_job_completed(job_id)
            job_key = f"{tool_name}_{target_name}"

            if status == 'completed':
                # Check if output exists
                output_dir = os.path.join(args.output_dir, tool_name, target_name)
                wrapper = get_wrapper(tool_name, config[tool_name], args.bench_root)
                if wrapper:
                    output_path = config[tool_name].get('output_path', '{output_dir}/*.sdf').format(output_dir=output_dir)
                    exists, files = ErrorHandler.check_output_exists(output_path, config[tool_name].get('output_format', 'sdf'))
                    if exists:
                        handler.set_job_status(tool_name, target_name, ToolStatus.INFERENCE_OK, f"{len(files)} output files")
                        logger.info(f"[{job_key}] COMPLETED — {len(files)} output files")
                    else:
                        handler.set_job_status(tool_name, target_name, ToolStatus.NO_OUTPUT, "no output files found")
                        logger.warning(f"[{job_key}] Completed but no output files found")
                completed.add(job_id)

            elif status == 'failed':
                # Check error type from log
                log_file = os.path.join(args.log_dir, f"{job_key}.err")
                stderr = ""
                if os.path.exists(log_file):
                    with open(log_file) as f:
                        stderr = f.read()

                error_type = ErrorHandler.detect_error_type(stderr, 1)
                retry = ErrorHandler.get_retry_action(error_type, attempt - 1)

                if retry and attempt <= 3:
                    logger.warning(f"[{job_key}] Failed ({error_type}), retrying (attempt {attempt + 1})...")

                    # Re-derive this job's own target/inputs/output_dir/script_path —
                    # do NOT reuse whatever the outer submission loop last left behind,
                    # since that belongs to a different tool/target pair.
                    retry_target = next((t for t in targets if t['name'] == target_name), None)
                    if retry_target is None:
                        logger.error(f"[{job_key}] Could not find target '{target_name}' for retry, giving up")
                        handler.set_job_status(tool_name, target_name, error_type, f"stderr: {stderr[:200]}")
                        failed[job_key] = error_type
                        completed.add(job_id)
                        continue

                    retry_input_dir = os.path.join('prepared_inputs', tool_name, target_name)
                    retry_manifest_path = os.path.join(retry_input_dir, 'input_manifest.json')
                    with open(retry_manifest_path) as f:
                        retry_manifest = json.load(f)
                    retry_inputs = retry_manifest.get('input_files', {})

                    retry_output_dir = os.path.join(args.output_dir, tool_name, target_name)
                    retry_script_path = job_scripts.get(job_key, os.path.join(args.job_dir, f"{job_key}.sh"))

                    # Modify parameters for retry
                    new_batch = args.batch_size
                    new_samples = args.num_samples
                    if retry.get('action') == 'reduce_batch':
                        new_batch = max(1, int(args.batch_size * retry['factor']))
                    elif retry.get('action') == 'reduce_samples':
                        new_samples = retry.get('num_samples', 50)

                    # Regenerate and resubmit
                    script = generate_job_script(
                        template, tool_name, config[tool_name], retry_target, retry_inputs,
                        retry_output_dir, new_samples, new_batch,
                        args.bench_root, args.log_dir, args.checkpoint_dir
                    )
                    with open(retry_script_path, 'w') as f:
                        f.write(script)

                    new_job_id = submit_job(retry_script_path)
                    if new_job_id:
                        submitted_jobs[new_job_id] = (tool_name, target_name, attempt + 1)
                        handler.set_job_status(tool_name, target_name, ToolStatus.INFERENCE_RUNNING, f"retry job_id={new_job_id}")
                else:
                    handler.set_job_status(tool_name, target_name, error_type, f"stderr: {stderr[:200]}")
                    logger.error(f"[{job_key}] FAILED ({error_type}), no more retries")
                    failed[job_key] = error_type

                completed.add(job_id)

        running = len(submitted_jobs) - len(completed)
        logger.info(f"  Progress: {len(completed)}/{len(submitted_jobs)} done, {running} running, {len(failed)} failed")

    logger.info(f"\nAll jobs complete: {len(submitted_jobs) - len(failed)} succeeded, {len(failed)} failed")
    if failed:
        logger.info("Failed jobs:")
        for job_key, error_type in failed.items():
            logger.info(f"  {job_key}: {error_type}")

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
