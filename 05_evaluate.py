#!/usr/bin/env python3
"""
Phase 05: Evaluate all collected molecules with the full metric suite.
Computes: validity, uniqueness, diversity, QED, SA, Lipinski, Vina score,
Tanimoto similarity to reference drug, 3D validity, and RMSD to crystal pose.

For tools that output SMILES without 3D coordinates, generates 3D conformers
and docks them with AutoDock Vina to obtain binding scores.

Usage:
    python 05_evaluate.py                           # Evaluate all results
    python 05_evaluate.py --tools TargetDiff         # Evaluate specific tool
    python 05_evaluate.py --targets EGFR             # Evaluate for specific target
    python 05_evaluate.py --skip-docking              # Skip Vina docking (faster)
"""
import os
import sys
import json
import argparse
import logging
import numpy as np
from typing import List, Optional, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# RDKit imports
from rdkit import Chem
from rdkit.Chem import (
    AllChem, Descriptors, QED, rdMolDescriptors, DataStructs,
    Lipinski, Crippen, rdFingerprintGenerator
)

# SA Score (from RDKit Contrib)
import sys as _sys
_sa_score_path = None
for p in [
    os.path.join(os.path.dirname(__file__), 'sa_score'),
    '/opt/conda/share/rdkit/Contrib/SA_Score',
    os.path.expanduser('~/rdkit/Contrib/SA_Score'),
]:
    if os.path.exists(os.path.join(p, 'sascorer.py')):
        _sa_score_path = p
        break

if _sa_score_path:
    _sys.path.insert(0, _sa_score_path)
    import sascorer
    HAS_SA_SCORE = True
else:
    HAS_SA_SCORE = False
    logger.warning("SA Score module not found. SA metric will be skipped.")


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


def compute_tanimoto_to_reference(smiles_list: List[str], ref_smiles: str) -> List[float]:
    """Compute Tanimoto similarity of each molecule to the reference drug."""
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    if ref_mol is None:
        return [None] * len(smiles_list)

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ref_fp = fpgen.GetFingerprint(ref_mol)

    scores = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scores.append(None)
            continue
        fp = fpgen.GetFingerprint(mol)
        sim = DataStructs.TanimotoSimilarity(ref_fp, fp)
        scores.append(sim)

    return scores


def compute_diversity(smiles_list: List[str]) -> float:
    """Compute internal diversity as mean pairwise Tanimoto distance."""
    if len(smiles_list) < 2:
        return None

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fps.append(fpgen.GetFingerprint(mol))

    if len(fps) < 2:
        return None

    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sims.append(sim)

    return 1.0 - np.mean(sims)


def evaluate_molecules(tool_name: str, target_name: str, smiles_list: List[str],
                       vina_scores: List[Optional[float]], ref_smiles: str,
                       skip_docking: bool = False, receptor_pdbqt: str = None,
                       pocket_box: dict = None) -> dict:
    """Evaluate a set of molecules with the full metric suite.

    Returns dict with all computed metrics.
    """
    n_total = len(smiles_list)
    if n_total == 0:
        return {
            'tool': tool_name, 'target': target_name, 'n_total': 0,
            'validity': 0, 'uniqueness': 0, 'diversity': None,
            'qed_mean': None, 'sa_mean': None, 'lipinski_pass_rate': None,
            'vina_score_mean': None, 'vina_score_best': None,
            'tanimoto_ref_mean': None, 'tanimoto_ref_max': None,
        }

    # Validity
    valid_mols = []
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_mols.append(mol)
            valid_smiles.append(Chem.MolToSmiles(mol))

    n_valid = len(valid_mols)
    validity = n_valid / n_total

    # Uniqueness
    unique_smiles = list(set(valid_smiles))
    n_unique = len(unique_smiles)
    uniqueness = n_unique / n_valid if n_valid > 0 else 0

    # Diversity
    diversity = compute_diversity(unique_smiles)

    # QED
    qed_scores = [QED.qed(mol) for mol in valid_mols]
    qed_mean = np.mean(qed_scores) if qed_scores else None
    qed_std = np.std(qed_scores) if qed_scores else None

    # SA Score
    if HAS_SA_SCORE:
        sa_scores = []
        for mol in valid_mols:
            try:
                sa = sascorer.calculateScore(mol)
                sa_scores.append(sa)
            except Exception:
                pass
        sa_mean = np.mean(sa_scores) if sa_scores else None
        sa_std = np.std(sa_scores) if sa_scores else None
    else:
        sa_mean = None
        sa_std = None

    # Lipinski Rule of Five
    lipinski_pass = []
    for mol in valid_mols:
        violations = sum([
            Descriptors.MolWt(mol) > 500,
            Descriptors.NumHDonors(mol) > 5,
            Descriptors.NumHAcceptors(mol) > 10,
            Crippen.MolLogP(mol) > 5,
        ])
        lipinski_pass.append(violations <= 1)

    lipinski_pass_rate = np.mean(lipinski_pass) if lipinski_pass else None

    # Molecular weight
    mw_scores = [Descriptors.MolWt(mol) for mol in valid_mols]
    mw_mean = np.mean(mw_scores) if mw_scores else None
    mw_std = np.std(mw_scores) if mw_scores else None

    # Tanimoto to reference drug
    tanimoto_scores = compute_tanimoto_to_reference(valid_smiles, ref_smiles)
    tanimoto_valid = [t for t in tanimoto_scores if t is not None]
    tanimoto_mean = np.mean(tanimoto_valid) if tanimoto_valid else None
    tanimoto_max = np.max(tanimoto_valid) if tanimoto_valid else None

    # Vina scores
    vina_valid = [v for v in vina_scores if v is not None] if vina_scores else []

    # If no Vina scores and not skipping docking, dock the molecules
    if not vina_valid and not skip_docking and receptor_pdbqt and pocket_box:
        logger.info(f"  Docking {n_unique} molecules with Vina...")
        from utils.vina_docking import dock_smiles_list
        dock_scores = dock_smiles_list(
            unique_smiles, receptor_pdbqt, pocket_box,
            work_dir=f'/tmp/vina_{tool_name}_{target_name}',
            timeout=60, exhaustiveness=8
        )
        vina_valid = [v for v in dock_scores if v is not None]
        vina_scores = dock_scores

    vina_mean = np.mean(vina_valid) if vina_valid else None
    vina_std = np.std(vina_valid) if vina_valid else None
    vina_best = np.min(vina_valid) if vina_valid else None  # Most negative = best

    # Novelty (approximation: fraction not matching reference drug)
    novelty = 1.0 - (sum(1 for t in tanimoto_scores if t is not None and t >= 0.95) / n_valid) if n_valid > 0 else 0

    return {
        'tool': tool_name,
        'target': target_name,
        'n_total': n_total,
        'n_valid': n_valid,
        'n_unique': n_unique,
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'diversity': diversity,
        'qed_mean': qed_mean,
        'qed_std': qed_std,
        'sa_mean': sa_mean,
        'sa_std': sa_std,
        'lipinski_pass_rate': lipinski_pass_rate,
        'mw_mean': mw_mean,
        'mw_std': mw_std,
        'vina_score_mean': vina_mean,
        'vina_score_std': vina_std,
        'vina_score_best': vina_best,
        'tanimoto_ref_mean': tanimoto_mean,
        'tanimoto_ref_max': tanimoto_max,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all collected molecules")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--targets-file', type=str, default=None)
    parser.add_argument('--tools', type=str, nargs='+', default=None)
    parser.add_argument('--targets', type=str, nargs='+', default=None)
    parser.add_argument('--collected-dir', type=str, default='collected_results', help='Directory with standardized SMILES')
    parser.add_argument('--output-dir', type=str, default='evaluations', help='Output directory for evaluation results')
    parser.add_argument('--receptor-dir', type=str, default='targets/pdbqt', help='Directory with receptor PDBQT files')
    parser.add_argument('--skip-docking', action='store_true', help='Skip Vina docking for tools without scores')
    parser.add_argument('--status-file', type=str, default='benchmark_status.json')
    args = parser.parse_args()

    config = load_config(args.config)
    targets = load_targets(args.targets_file)
    target_map = {t['name']: t for t in targets}

    os.makedirs(args.output_dir, exist_ok=True)

    tools = {k: v for k, v in config.items() if not k.startswith('_')}
    if args.tools:
        tools = {k: v for k, v in tools.items() if k in args.tools}

    if args.targets:
        target_list = [t for t in targets if t['name'] in args.targets]
    else:
        target_list = targets

    logger.info(f"Evaluating {len(tools)} tools × {len(target_list)} targets...")

    all_results = {}
    success_count = 0

    for tool_name in tools:
        for target in target_list:
            target_name = target['name']
            smi_path = os.path.join(args.collected_dir, f"{tool_name}_{target_name}_final.smi")

            if not os.path.exists(smi_path):
                logger.debug(f"[{tool_name}/{target_name}] No SMILES file found, skipping")
                continue

            # Load SMILES and Vina scores
            smiles_list = []
            vina_scores = []
            with open(smi_path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if parts:
                        smiles_list.append(parts[0])
                        if len(parts) >= 3:
                            try:
                                vina_scores.append(float(parts[2]) if parts[2] != 'NA' else None)
                            except ValueError:
                                vina_scores.append(None)
                        else:
                            vina_scores.append(None)

            if not smiles_list:
                logger.warning(f"[{tool_name}/{target_name}] No molecules in file")
                continue

            logger.info(f"[{tool_name}/{target_name}] Evaluating {len(smiles_list)} molecules...")

            # Find receptor PDBQT
            receptor_pdbqt = os.path.join(args.receptor_dir, f"{target_name}_protein.pdbqt")
            if not os.path.exists(receptor_pdbqt):
                receptor_pdbqt = None

            # Evaluate
            result = evaluate_molecules(
                tool_name, target_name, smiles_list, vina_scores,
                ref_smiles=target['drug_smiles'],
                skip_docking=args.skip_docking,
                receptor_pdbqt=receptor_pdbqt,
                pocket_box=target.get('pocket_box'),
            )

            # Save individual result
            result_path = os.path.join(args.output_dir, f"{tool_name}_{target_name}_eval.json")
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            all_results[f"{tool_name}_{target_name}"] = result
            success_count += 1

            # Print summary
            vina_s = f"{result['vina_score_mean']:.2f}/{result['vina_score_best']:.2f}" if result.get('vina_score_mean') is not None else "NA"
            tan_s = f"{result['tanimoto_ref_mean']:.3f}" if result.get('tanimoto_ref_mean') is not None else "NA"
            div_s = f"{result['diversity']:.3f}" if result.get('diversity') is not None else "NA"
            logger.info(f"  Val={result['validity']:.3f} Div={div_s} QED={result['qed_mean']:.3f} "
                        f"SA={result['sa_mean']:.3f if result.get('sa_mean') else 'NA'} "
                        f"Vina={vina_s} Tan={tan_s}")

    # Save all results
    all_path = os.path.join(args.output_dir, 'all_results.json')
    with open(all_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nEvaluation complete: {success_count} tool×target combinations evaluated")
    logger.info(f"Results saved to {args.output_dir}/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
