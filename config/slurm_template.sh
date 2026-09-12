#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}G
#SBATCH --time={time_limit}
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --error={log_dir}/{job_name}.err

# Activate conda environment
source {conda_init_path}
conda activate {conda_env}

# Set working directory
cd {repo_path}

# Run extra setup commands
{extra_setup}

# Run inference
{inference_cmd}

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    touch {checkpoint_dir}/{job_name}.done
    echo "SUCCESS: {job_name} completed"
else
    touch {checkpoint_dir}/{job_name}.failed
    echo "FAILED: {job_name} exited with code $EXIT_CODE"
fi

exit $EXIT_CODE
