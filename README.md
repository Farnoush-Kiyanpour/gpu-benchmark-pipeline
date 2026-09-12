# GPU Benchmark Pipeline: 36-Tool De Novo Molecule Generation

Complete Python pipeline for benchmarking 36 target-conditioned de novo molecule generation tools across 10 protein targets on a GPU supercomputer with Slurm.

## Changelog

- **Fixed:** `03_submit_inference.py` retry logic referenced an undefined `batch_size`
  name and reused a stale `target`/`inputs`/`script_path` from the last tool in the
  initial submission loop instead of the job actually being retried. Any OOM/timeout
  retry across the 36 tools would previously crash or resubmit the wrong job.
- **Fixed:** `00_setup_environments.py` used a 10-minute timeout per conda env, which
  is too short for `conda create` with PyTorch+CUDA (often 20–60+ min); raised to 60 min.
- **Optimized:** Phases 0 (env setup) and 1 (weight downloads) now run concurrently
  across tools (`--workers`, default 6 and 8 respectively) instead of one tool at a
  time — these 36 tools are independent, so this is the main lever for cutting wall
  time before any GPU inference starts. Tune `--workers` down if it saturates the
  login node or hits shared package-cache lock contention.

## Quick Start

```bash
# 1. Copy this directory to your HPC account
scp -r gpu_benchmark/ user@hpc:/path/to/workdir/

# 2. Ensure tool repos are cloned
cd /path/to/workdir/
git clone https://github.com/guanjq/targetdiff.git bench_repos/TargetDiff
# ... (repeat for all 36 tools, or use the repo list in config/tools_config.json)

# 3. Ensure target data is prepared
# PDB files in targets/pdb/, sequences in targets/sequences/, etc.

# 4. Run the full pipeline
python run_benchmark.py --bench-root /path/to/workdir --num-samples 100 --max-concurrent 4

# 5. Or run a smoke test first (5 molecules, 1 target)
python run_benchmark.py --smoke-test --bench-root /path/to/workdir
```

## Execution Order

The pipeline runs 7 phases in order. Each phase is resumable — completed work is skipped based on the status file.

| Phase | Script | Description |
|-------|--------|-------------|
| 0 | `00_setup_environments.py` | Create conda env per tool |
| 1 | `01_download_weights.py` | Download pre-trained weights |
| 2 | `02_prepare_inputs.py` | Convert targets to tool-specific formats |
| 3 | `03_submit_inference.py` | Submit + monitor Slurm jobs (36×10=360 jobs) |
| 4 | `04_collect_results.py` | Standardize outputs to common SMILES format |
| 5 | `05_evaluate.py` | Compute all metrics (Vina, QED, SA, diversity, etc.) |
| 6 | `06_generate_report.py` | Generate tables, charts, markdown report |

Run specific phases only:
```bash
python run_benchmark.py --phases 3 4 5 6 --bench-root /path/to/workdir
```

## Configuration

### config/tools_config.json
Defines all 36 tools with:
- Conda environment specification (Python, PyTorch, CUDA versions)
- Pre-trained weight download URL and method (HuggingFace, Google Drive, wget)
- Input format (protein_pdb, protein_fasta, pharmacophore, etc.)
- Inference command template
- Output format and path pattern
- GPU memory requirement and timeout

### config/targets.json
Defines 10 protein targets with PDB IDs, UniProt accessions, FDA-approved reference drugs, and binding pocket coordinates.

### config/slurm_template.sh
Slurm job script template with placeholders for job name, GPU count, memory, time limit, conda env, and inference command.

## Error Handling

| Error | Detection | Action |
|-------|-----------|--------|
| Conda env creation fails | Non-zero exit code | Skip tool, log error |
| Weight download fails | HTTP error / missing file | Retry 3× with backoff, then skip |
| Inference OOM | "CUDA out of memory" in stderr | Halve batch size, retry (max 3) |
| Inference crash | Non-zero exit code | Retry once, then skip |
| Inference timeout | Slurm kills job | Reduce samples to 50, retry once |
| Output missing | No output files found | Mark as "no_output", skip |
| Output parsing fails | Invalid SDF/SMILES | Try RDKit, OpenBabel, regex extraction |
| Vina docking fails | Vina error | Set score to None, continue |

## Tool Categories

| Category | Count | Input | Examples |
|----------|-------|-------|----------|
| 3D pocket | 25 | Protein PDB | TargetDiff, Pocket2Mol, DiffSBDD, MolCRAFT |
| Sequence | 5 | Protein FASTA | DrugGPT, DrugGen, AlphaDrug |
| Interaction | 3 | PDB + ref ligand | DrugGEN, DeepICL, Assay2Mol |
| Pharmacophore | 2 | Pharmacophore features | TransPharmer, SQUID |
| Classical | 1 | Protein PDBQT | AutoGrow4 |

## Output Files

After completion, the following are generated:
- `benchmark_summary.csv` — All metrics in tabular format
- `benchmark_comparison.png/svg` — Bar charts (Vina, QED, SA, Tanimoto, diversity, Lipinski)
- `benchmark_radar.png/svg` — Aggregate normalized performance
- `benchmark_heatmap.png/svg` — Vina scores by tool × target
- `benchmark_boxplots.png/svg` — Metric distributions per tool
- `report_benchmark.md` — Full markdown report
- `benchmark_status.json` — Pipeline status tracking
- `collected_results/` — Standardized SMILES files per tool×target
- `evaluations/` — Individual evaluation JSONs per tool×target

## Compute Requirements

- ~200-300 GPU-hours for full benchmark (360 jobs)
- ~24-48h wall time with 4 concurrent GPU jobs
- ~180GB disk for conda environments
- ~50GB disk for pre-trained weights
- Each job: 1 GPU (6-16GB VRAM depending on tool), 32-64GB RAM
