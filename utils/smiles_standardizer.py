#!/usr/bin/env python3
"""
SMILES standardization and output parsing utilities.
Handles SDF, MOL, SMI, and other output formats from different tools.
"""
import os
import re
import glob
from typing import List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def remove_salts(smiles: str) -> Optional[str]:
    """Remove salt fragments, keep largest organic fragment."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    remover = SaltRemover.SaltRemover()
    mol = remover.StripMol(mol)
    if mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def parse_sdf_file(sdf_path: str) -> List[Tuple[str, str]]:
    """Parse an SDF file and return list of (smiles, name) tuples."""
    results = []
    supplier = Chem.SDMolSupplier(sdf_path)
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        try:
            smiles = Chem.MolToSmiles(mol)
            name = mol.GetProp('_Name') if mol.HasProp('_Name') else f'mol_{i}'
            results.append((smiles, name))
        except Exception:
            continue
    return results


def parse_smi_file(smi_path: str) -> List[Tuple[str, str]]:
    """Parse a SMILES file (tab or whitespace separated)."""
    results = []
    with open(smi_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) >= 1:
                smiles = parts[0]
                name = parts[1] if len(parts) > 1 else f'mol_{i}'
                results.append((smiles, name))
    return results


def parse_output(output_path: str, output_format: str) -> List[Tuple[str, str]]:
    """Parse tool output in various formats. Returns list of (smiles, name).

    Tries multiple parsing strategies based on format and file extension.
    """
    # Handle glob patterns
    if '*' in output_path:
        files = sorted(glob.glob(output_path))
    else:
        files = [output_path] if os.path.exists(output_path) else []

    if not files:
        return []

    all_results = []

    for fpath in files:
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            continue

        ext = os.path.splitext(fpath)[1].lower()

        # Strategy 1: Try format-specific parser
        try:
            if ext in ('.sdf', '.mol'):
                all_results.extend(parse_sdf_file(fpath))
            elif ext in ('.smi', '.smiles', '.csv', '.tsv', '.txt'):
                all_results.extend(parse_smi_file(fpath))
            else:
                # Try SDF first, then SMI
                try:
                    all_results.extend(parse_sdf_file(fpath))
                except Exception:
                    all_results.extend(parse_smi_file(fpath))
        except Exception:
            # Strategy 2: Fallback regex extraction
            all_results.extend(_regex_extract_smiles(fpath))

    return all_results


def _regex_extract_smiles(filepath: str) -> List[Tuple[str, str]]:
    """Last-resort: extract SMILES-like strings from any text file using regex."""
    results = []
    # Basic SMILES pattern: starts with C/c/N/n/O/o/S/s/[/(/B/r and contains ring/branch chars
    smiles_pattern = re.compile(
        r'(?<![\w])([CNOSBPFHIKbrclnpso0-9\[\]\(\)=#@/\\\-+.:!])'
        r'([CNOSBPFHIKbrclnpso0-9\[\]\(\)=#@/\\\-+.:!]{5,80})'
        r'(?![\w])'
    )

    with open(filepath, errors='ignore') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Try each match
            for match in smiles_pattern.finditer(line):
                candidate = match.group(0)
                mol = Chem.MolFromSmiles(candidate)
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    results.append((smiles, f'regex_mol_{i}'))
                    break  # one per line

    return results


def standardize_molecules(smiles_list: List[Tuple[str, str]],
                          remove_salt: bool = True,
                          min_atoms: int = 3,
                          max_atoms: int = 100) -> List[Tuple[str, str]]:
    """Standardize a list of molecules: canonicalize, remove salts, filter by size.

    Args:
        smiles_list: List of (smiles, name) tuples
        remove_salt: Whether to remove salt fragments
        min_atoms: Minimum heavy atom count
        max_atoms: Maximum heavy atom count

    Returns:
        List of (canonical_smiles, name) tuples, deduplicated
    """
    seen = set()
    results = []

    for smiles, name in smiles_list:
        try:
            if remove_salt:
                canon = remove_salts(smiles)
            else:
                canon = canonicalize_smiles(smiles)

            if canon is None:
                continue

            mol = Chem.MolFromSmiles(canon)
            if mol is None:
                continue

            num_heavy = mol.GetNumHeavyAtoms()
            if num_heavy < min_atoms or num_heavy > max_atoms:
                continue

            if canon not in seen:
                seen.add(canon)
                results.append((canon, name))
        except Exception:
            continue

    return results


def write_standard_smi(output_path: str, smiles_list: List[Tuple[str, str]],
                       vina_scores: List[float] = None):
    """Write a standardized SMILES file.

    Format: SMILES\\tname\\tvina_score\\ttool_target
    """
    with open(output_path, 'w') as f:
        for i, (smiles, name) in enumerate(smiles_list):
            score = f"{vina_scores[i]:.3f}" if vina_scores and i < len(vina_scores) and vina_scores[i] is not None else "NA"
            f.write(f"{smiles}\t{name}\t{score}\n")


def generate_3d_conformer(smiles: str, method: str = 'etkdg') -> Optional[Chem.Mol]:
    """Generate a 3D conformer for a SMILES string.

    Args:
        smiles: Input SMILES
        method: 'etkdg' (RDKit) or 'openbabel' (OpenBabel)

    Returns:
        RDKit Mol with 3D coordinates, or None on failure
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if method == 'openbabel':
        # Use OpenBabel for 3D generation
        import tempfile
        import subprocess
        smi_path = tempfile.mktemp(suffix='.smi')
        sdf_path = tempfile.mktemp(suffix='.sdf')
        with open(smi_path, 'w') as f:
            f.write(f"{smiles}\tmol\n")
        subprocess.run(f'obabel {smi_path} -O {sdf_path} --gen3d -h 2>/dev/null', shell=True)
        if os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 0:
            mol3d = Chem.SDMolSupplier(sdf_path)[0]
            if mol3d is not None:
                return mol3d
        # Fallback to ETKDG
        method = 'etkdg'

    if method == 'etkdg':
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            return mol
        except Exception:
            return None

    return None
