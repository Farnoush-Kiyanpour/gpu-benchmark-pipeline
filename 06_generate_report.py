#!/usr/bin/env python3
"""
Phase 06: Generate comparison tables, charts, and final benchmark report.
Reads evaluation results from Phase 05 and produces:
- Summary CSV table (all tools × all metrics)
- Bar charts comparing key metrics across tools
- Radar chart (aggregate normalized performance)
- Heatmap (Vina scores by tool × target)
- Box plots (metric distributions)
- Markdown report with methods, results, and discussion

Usage:
    python 06_generate_report.py                           # Generate full report
    python 06_generate_report.py --output-dir /path/to/out  # Custom output directory
"""
import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(eval_dir: str) -> pd.DataFrame:
    """Load all evaluation results into a DataFrame."""
    rows = []
    for fpath in sorted(os.listdir(eval_dir)):
        if not fpath.endswith('_eval.json'):
            continue
        with open(os.path.join(eval_dir, fpath)) as f:
            data = json.load(f)
        rows.append(data)

    if not rows:
        logger.error("No evaluation results found")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(['tool', 'target']).reset_index(drop=True)
    return df


def generate_summary_table(df: pd.DataFrame, output_dir: str):
    """Save summary CSV and print formatted table."""
    csv_path = os.path.join(output_dir, 'benchmark_summary.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Summary table saved to {csv_path}")

    # Print formatted summary
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    for tool in sorted(df['tool'].unique()):
        tool_df = df[df['tool'] == tool]
        print(f"\n{tool} ({len(tool_df)} targets):")
        for _, r in tool_df.iterrows():
            vina_s = f"{r['vina_score_mean']:.2f}/{r['vina_score_best']:.2f}" if pd.notna(r.get('vina_score_mean')) else "NA"
            tan_s = f"{r['tanimoto_ref_mean']:.3f}" if pd.notna(r.get('tanimoto_ref_mean')) else "NA"
            div_s = f"{r['diversity']:.3f}" if pd.notna(r.get('diversity')) else "NA"
            sa_s = f"{r['sa_mean']:.3f}" if pd.notna(r.get('sa_mean')) else "NA"
            print(f"  {r['target']:12s} n={r['n_total']:4.0f}  Val={r['validity']:.3f}  Div={div_s}  "
                  f"QED={r['qed_mean']:.3f}  SA={sa_s}  Vina={vina_s}  Tan={tan_s}")


def generate_bar_charts(df: pd.DataFrame, output_dir: str):
    """Generate bar charts comparing key metrics across tools."""
    targets = sorted(df['target'].unique())
    tools = sorted(df['tool'].unique())
    n_targets = len(targets)
    x = np.arange(n_targets)
    w = 0.8 / max(len(tools), 1)
    colors = plt.cm.tab20(np.linspace(0, 1, len(tools)))

    metrics = [
        ('vina_score_mean', 'Vina Score (kcal/mol)', 'Mean Vina Docking Score', True),
        ('qed_mean', 'QED Score', 'Mean QED (Drug-likeness)', False),
        ('sa_mean', 'SA Score', 'Mean Synthetic Accessibility', True),
        ('tanimoto_ref_mean', 'Tanimoto Similarity', 'Mean Tanimoto to Reference Drug', False),
        ('diversity', 'Diversity', 'Internal Diversity', False),
        ('lipinski_pass_rate', 'Lipinski Pass Rate', 'Lipinski Rule of Five Pass Rate', False),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Benchmark Comparison: All Tools Across All Targets', fontsize=14, fontweight='bold')

    for idx, (metric, ylabel, title, lower_is_better) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        for i, tool in enumerate(tools):
            tool_df = df[df['tool'] == tool]
            vals = []
            for t in targets:
                row = tool_df[tool_df['target'] == t]
                val = row[metric].values[0] if len(row) > 0 and pd.notna(row[metric].values[0]) else 0
                vals.append(val)
            ax.bar(x + i * w, vals, w, label=tool, color=colors[i], alpha=0.85)

        ax.set_xticks(x + w * (len(tools) - 1) / 2)
        ax.set_xticklabels(targets, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        if len(tools) <= 10:
            ax.legend(fontsize=7, loc='best')
        if lower_is_better:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'benchmark_comparison.svg'), bbox_inches='tight')
    plt.close()
    logger.info("Bar charts saved")


def generate_radar_chart(df: pd.DataFrame, output_dir: str):
    """Generate radar chart showing aggregate normalized performance per tool."""
    metrics = ['Vina (neg)', 'QED', 'SA (inv)', 'Diversity', 'Tanimoto', 'Lipinski']
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab20(np.linspace(0, 1, len(df['tool'].unique())))

    for i, tool in enumerate(sorted(df['tool'].unique())):
        tool_df = df[df['tool'] == tool]

        # Normalize each metric to 0-1
        vina = tool_df['vina_score_mean'].mean()
        vina_norm = min(abs(vina) / 15, 1.0) if pd.notna(vina) else 0

        qed_norm = tool_df['qed_mean'].mean() if pd.notna(tool_df['qed_mean'].mean()) else 0

        sa = tool_df['sa_mean'].mean()
        sa_norm = 1 - min(sa / 6, 1.0) if pd.notna(sa) else 0

        div = tool_df['diversity'].mean()
        div_norm = div if pd.notna(div) else 0

        tan = tool_df['tanimoto_ref_mean'].mean()
        tan_norm = min(tan / 0.3, 1.0) if pd.notna(tan) else 0

        lip = tool_df['lipinski_pass_rate'].mean()
        lip_norm = lip if pd.notna(lip) else 0

        values = [vina_norm, qed_norm, sa_norm, div_norm, tan_norm, lip_norm]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=tool, color=colors[i], markersize=4)
        ax.fill(angles, values, alpha=0.08, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Aggregate Performance (Normalized)', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_radar.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'benchmark_radar.svg'), bbox_inches='tight')
    plt.close()
    logger.info("Radar chart saved")


def generate_heatmap(df: pd.DataFrame, output_dir: str):
    """Generate heatmap of Vina scores by tool × target."""
    pivot = df.pivot_table(index='tool', columns='target', values='vina_score_mean', aggfunc='first')

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.2), max(6, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=-6,
                ax=ax, linewidths=0.5, cbar_kws={'label': 'Vina Score (kcal/mol)'},
                annot_kws={'size': 7})
    ax.set_title('Vina Docking Scores by Tool and Target', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'benchmark_heatmap.svg'), bbox_inches='tight')
    plt.close()
    logger.info("Heatmap saved")


def generate_box_plots(df: pd.DataFrame, output_dir: str):
    """Generate box plots showing metric distributions per tool."""
    metrics = ['qed_mean', 'sa_mean', 'vina_score_mean', 'tanimoto_ref_mean', 'diversity']
    metric_labels = ['QED', 'SA Score', 'Vina Score', 'Tanimoto to Ref', 'Diversity']

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
    fig.suptitle('Metric Distributions by Tool', fontsize=14, fontweight='bold')

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        plot_df = df[df[metric].notna()]
        if len(plot_df) == 0:
            ax.set_title(f'{label} (no data)')
            continue
        sns.boxplot(data=plot_df, x='tool', y=metric, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_title(label)
        ax.set_xlabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'benchmark_boxplots.svg'), bbox_inches='tight')
    plt.close()
    logger.info("Box plots saved")


def generate_markdown_report(df: pd.DataFrame, output_dir: str):
    """Generate a comprehensive markdown report."""
    report_path = os.path.join(output_dir, 'report_benchmark.md')

    tools = sorted(df['tool'].unique())
    targets = sorted(df['target'].unique())

    # Aggregate stats per tool
    agg = df.groupby('tool').agg({
        'n_total': 'sum',
        'validity': 'mean',
        'diversity': 'mean',
        'qed_mean': 'mean',
        'sa_mean': 'mean',
        'lipinski_pass_rate': 'mean',
        'vina_score_mean': 'mean',
        'vina_score_best': 'min',
        'tanimoto_ref_mean': 'mean',
    }).round(3)

    report = f"""# Benchmark Report: Target-Conditioned De Novo Molecule Generation

## Summary

Benchmarked **{len(tools)} tools** across **{len(targets)} protein targets**, generating a total of **{int(df['n_total'].sum())} molecules**.

## Aggregate Results by Tool

| Tool | Total Mols | Validity | Diversity | QED | SA | Lipinski | Vina (mean) | Vina (best) | Tanimoto |
|------|-----------|----------|-----------|-----|-----|----------|-------------|-------------|----------|
"""
    for tool in tools:
        row = agg.loc[tool]
        vina_m = f"{row['vina_score_mean']:.2f}" if pd.notna(row['vina_score_mean']) else "NA"
        vina_b = f"{row['vina_score_best']:.2f}" if pd.notna(row['vina_score_best']) else "NA"
        tan = f"{row['tanimoto_ref_mean']:.3f}" if pd.notna(row['tanimoto_ref_mean']) else "NA"
        div = f"{row['diversity']:.3f}" if pd.notna(row['diversity']) else "NA"
        sa = f"{row['sa_mean']:.3f}" if pd.notna(row['sa_mean']) else "NA"
        report += f"| {tool} | {int(row['n_total'])} | {row['validity']:.3f} | {div} | {row['qed_mean']:.3f} | {sa} | {row['lipinski_pass_rate']:.3f} | {vina_m} | {vina_b} | {tan} |\n"

    report += f"""
## Per-Target Results

"""
    for target in targets:
        target_df = df[df['target'] == target]
        report += f"### {target}\n\n"
        report += "| Tool | N | Validity | Diversity | QED | SA | Vina (mean) | Vina (best) | Tanimoto |\n"
        report += "|------|---|----------|-----------|-----|-----|-------------|-------------|----------|\n"
        for _, row in target_df.iterrows():
            vina_m = f"{row['vina_score_mean']:.2f}" if pd.notna(row.get('vina_score_mean')) else "NA"
            vina_b = f"{row['vina_score_best']:.2f}" if pd.notna(row.get('vina_score_best')) else "NA"
            tan = f"{row['tanimoto_ref_mean']:.3f}" if pd.notna(row.get('tanimoto_ref_mean')) else "NA"
            div = f"{row['diversity']:.3f}" if pd.notna(row.get('diversity')) else "NA"
            sa = f"{row['sa_mean']:.3f}" if pd.notna(row.get('sa_mean')) else "NA"
            report += f"| {row['tool']} | {int(row['n_total'])} | {row['validity']:.3f} | {div} | {row['qed_mean']:.3f} | {sa} | {vina_m} | {vina_b} | {tan} |\n"
        report += "\n"

    report += """
## Figures

- `benchmark_comparison.png/svg` — Bar charts comparing all metrics across tools and targets
- `benchmark_radar.png/svg` — Aggregate normalized performance radar chart
- `benchmark_heatmap.png/svg` — Vina score heatmap by tool × target
- `benchmark_boxplots.png/svg` — Metric distribution box plots per tool
- `benchmark_summary.csv` — All metrics in tabular format

## Methods

### Target Selection
10 protein targets with well-defined 3D structures, co-crystallized ligands, and FDA-approved reference drugs.

### Evaluation Metrics
- **Validity**: Fraction of SMILES parseable by RDKit
- **Uniqueness**: Fraction of unique canonical SMILES
- **Diversity**: Mean pairwise Tanimoto distance (Morgan fingerprints, r=2, 2048 bits)
- **QED**: Quantitative Estimate of Drug-likeness (0-1, higher = better)
- **SA Score**: Synthetic Accessibility score (1-10, lower = easier to synthesize)
- **Lipinski**: Fraction passing Lipinski's Rule of Five
- **Vina Score**: AutoDock Vina docking score (kcal/mol, more negative = better)
- **Tanimoto to Ref**: Tanimoto similarity to FDA-approved reference drug

### Inference
Each tool generated 100 molecules per target using pre-trained weights. Tools that output SMILES without 3D coordinates were docked post-generation using OpenBabel for conformer generation and AutoDock Vina for scoring.
"""

    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Markdown report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison tables, charts, and report")
    parser.add_argument('--eval-dir', type=str, default='evaluations', help='Directory with evaluation JSONs')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for report and charts')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_results(args.eval_dir)
    if df.empty:
        return 1

    logger.info(f"Loaded {len(df)} evaluation results ({df['tool'].nunique()} tools × {df['target'].nunique()} targets)")

    generate_summary_table(df, args.output_dir)
    generate_bar_charts(df, args.output_dir)
    generate_radar_chart(df, args.output_dir)
    generate_heatmap(df, args.output_dir)
    generate_box_plots(df, args.output_dir)
    generate_markdown_report(df, args.output_dir)

    logger.info(f"\nAll outputs saved to {args.output_dir}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
