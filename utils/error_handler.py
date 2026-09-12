#!/usr/bin/env python3
"""
Error handling and retry logic for the benchmark pipeline.
Handles OOM, crashes, timeouts, and missing outputs with configurable retry strategies.
"""
import os
import re
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ToolStatus:
    """Track status of each tool across pipeline phases."""
    PENDING = "pending"
    ENV_OK = "env_ok"
    ENV_FAILED = "env_failed"
    WEIGHTS_OK = "weights_ok"
    WEIGHTS_FAILED = "weights_failed"
    INPUT_OK = "input_ok"
    INPUT_FAILED = "input_failed"
    INFERENCE_RUNNING = "inference_running"
    INFERENCE_OK = "inference_ok"
    INFERENCE_FAILED = "inference_failed"
    OOM = "oom"
    TIMEOUT = "timeout"
    NO_OUTPUT = "no_output"
    SKIPPED = "skipped"


class ErrorHandler:
    """Centralized error handling with retry logic and status tracking."""

    def __init__(self, status_file: str = "benchmark_status.json"):
        self.status_file = status_file
        self.status = self._load_status()

    def _load_status(self) -> Dict:
        if os.path.exists(self.status_file):
            with open(self.status_file) as f:
                return json.load(f)
        return {"tools": {}, "jobs": {}}

    def _save_status(self):
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def set_tool_status(self, tool: str, phase: str, status: str, detail: str = ""):
        if tool not in self.status["tools"]:
            self.status["tools"][tool] = {}
        self.status["tools"][tool][phase] = {"status": status, "detail": detail, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        self._save_status()
        logger.info(f"[{tool}] {phase}: {status}" + (f" — {detail}" if detail else ""))

    def get_tool_status(self, tool: str, phase: str) -> Optional[str]:
        if tool in self.status["tools"] and phase in self.status["tools"][tool]:
            return self.status["tools"][tool][phase]["status"]
        return None

    def is_tool_ready(self, tool: str) -> bool:
        """Check if tool passed env and weights phases."""
        env = self.get_tool_status(tool, "env")
        weights = self.get_tool_status(tool, "weights")
        return env == ToolStatus.ENV_OK and weights == ToolStatus.WEIGHTS_OK

    def set_job_status(self, tool: str, target: str, status: str, detail: str = ""):
        key = f"{tool}_{target}"
        self.status["jobs"][key] = {"status": status, "detail": detail, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        self._save_status()

    def get_job_status(self, tool: str, target: str) -> Optional[str]:
        key = f"{tool}_{target}"
        if key in self.status["jobs"]:
            return self.status["jobs"][key]["status"]
        return None

    @staticmethod
    def detect_error_type(stderr: str, returncode: int) -> str:
        """Classify error from stderr and return code."""
        if stderr is None:
            stderr = ""
        stderr_lower = stderr.lower()

        # CUDA OOM
        if "cuda out of memory" in stderr_lower or "out of memory" in stderr_lower:
            return ToolStatus.OOM

        # Timeout (killed by signal)
        if returncode in (-9, 137, 143) or "killed" in stderr_lower or "timeout" in stderr_lower:
            return ToolStatus.TIMEOUT

        # Missing weights / checkpoint
        if "no such file or directory" in stderr_lower and ("checkpoint" in stderr_lower or "model" in stderr_lower or "weight" in stderr_lower):
            return ToolStatus.WEIGHTS_FAILED

        # Missing input
        if "no such file or directory" in stderr_lower and ("pdb" in stderr_lower or "input" in stderr_lower or "fasta" in stderr_lower):
            return ToolStatus.INPUT_FAILED

        # Import / dependency error
        if "modulenotfounderror" in stderr_lower or "importerror" in stderr_lower or "no module named" in stderr_lower:
            return ToolStatus.ENV_FAILED

        # Generic inference failure
        if returncode != 0:
            return ToolStatus.INFERENCE_FAILED

        return "ok"

    @staticmethod
    def get_retry_action(error_type: str, attempt: int) -> Optional[Dict[str, Any]]:
        """Determine retry action based on error type and attempt number.

        Returns dict with modified parameters, or None if no retry should be attempted.
        """
        max_attempts = {
            ToolStatus.OOM: 3,
            ToolStatus.INFERENCE_FAILED: 2,
            ToolStatus.TIMEOUT: 2,
        }

        if error_type not in max_attempts:
            return None

        if attempt >= max_attempts[error_type]:
            return None

        if error_type == ToolStatus.OOM:
            # Halve batch size each retry
            return {"action": "reduce_batch", "factor": 0.5, "attempt": attempt + 1}

        if error_type == ToolStatus.TIMEOUT:
            # Reduce num_samples to 50 on first retry, 25 on second
            return {"action": "reduce_samples", "num_samples": max(25, 100 // (2 ** attempt)), "attempt": attempt + 1}

        if error_type == ToolStatus.INFERENCE_FAILED:
            # Simple retry
            return {"action": "retry", "attempt": attempt + 1}

        return None

    @staticmethod
    def check_output_exists(output_path: str, output_format: str) -> Tuple[bool, list]:
        """Check if output files exist and are non-empty.

        Returns (exists, list_of_files).
        """
        # Handle glob patterns
        if '*' in output_path:
            import glob
            files = glob.glob(output_path)
            valid = [f for f in files if os.path.getsize(f) > 0]
            return len(valid) > 0, valid

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True, [output_path]

        return False, []

    @staticmethod
    def run_with_retry(cmd: str, cwd: str, env: dict = None,
                       max_retries: int = 2, timeout: int = 7200) -> Tuple[int, str, str]:
        """Run a command with simple retry logic.

        Returns (returncode, stdout, stderr).
        """
        for attempt in range(max_retries + 1):
            try:
                result = subprocess.run(
                    cmd, shell=True, cwd=cwd, env=env,
                    capture_output=True, text=True, timeout=timeout
                )
                if result.returncode == 0:
                    return result.returncode, result.stdout, result.stderr

                error_type = ErrorHandler.detect_error_type(result.stderr, result.returncode)
                logger.warning(f"Attempt {attempt + 1} failed ({error_type}): {result.stderr[:500]}")

                if attempt < max_retries:
                    retry = ErrorHandler.get_retry_action(error_type, attempt)
                    if retry is None:
                        break
                    logger.info(f"Retrying with action: {retry}")
                    time.sleep(5 * (attempt + 1))  # backoff
                else:
                    return result.returncode, result.stdout, result.stderr

            except subprocess.TimeoutExpired:
                logger.warning(f"Attempt {attempt + 1} timed out after {timeout}s")
                if attempt < max_retries:
                    time.sleep(10)
                else:
                    return -1, "", "TIMEOUT: process killed after exceeding time limit"

        return result.returncode, result.stdout, result.stderr
