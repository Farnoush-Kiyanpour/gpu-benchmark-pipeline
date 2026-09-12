#!/usr/bin/env python3
"""
Post-generation Vina docking for tools that output SMILES without 3D coordinates.
Generates 3D conformers and docks them against the target protein.
"""
import os
import re
import json
import subprocess
import tempfile
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem


def prepare_receptor_pdbqt(pdb_path: str, pdbqt_path: str,
                           prepare_receptor_script: str = None) -> bool:
    """Convert a protein PDB to PDBQT format for Vina docking.

    Args:
        pdb_path: Path to input PDB file
        pdbqt_path: Path to output PDBQT file
        prepare_receptor_script: Path to prepare_receptor4.py (MGLTools)

    Returns:
        True if successful, False otherwise
    """
    if prepare_receptor_script is None:
        # Try common locations
        candidates = [
            '/opt/ADFRsuite/CCSBpckgs/AutoDockTools/Utilities24/prepare_receptor4.py',
            '/usr/local/bin/prepare_receptor4.py',
        ]
        for c in candidates:
            if os.path.exists(c):
                prepare_receptor_script = c
                break

    if prepare_receptor_script is None:
        # Fallback: use OpenBabel
        cmd = f'obabel {pdb_path} -O {pdbqt_path} -xr 2>/dev/null'
        result = os.system(cmd)
        return os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0

    # Use MGLTools
    pythonsh = '/opt/ADFRsuite/bin/pythonsh'
    if not os.path.exists(pythonsh):
        pythonsh = 'python2'

    cmd = f'{pythonsh} {prepare_receptor_script} -r {pdb_path} -o {pdbqt_path} -A hydrogens_nonpolar'
    os.system(cmd)
    return os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0


def prepare_ligand_pdbqt(smiles: str, pdbqt_path: str,
                         prepare_ligand_script: str = None) -> bool:
    """Convert a SMILES string to PDBQT format for Vina docking.

    Uses OpenBabel for 3D generation, then MGLTools for PDBQT conversion.

    Args:
        smiles: SMILES string
        pdbqt_path: Output PDBQT path
        prepare_ligand_script: Path to prepare_ligand4.py

    Returns:
        True if successful
    """
    # Step 1: SMILES -> 3D SDF via OpenBabel
    smi_path = pdbqt_path.replace('.pdbqt', '.smi')
    sdf_path = pdbqt_path.replace('.pdbqt', '.sdf')

    with open(smi_path, 'w') as f:
        f.write(f"{smiles}\tligand\n")

    os.system(f'obabel {smi_path} -O {sdf_path} --gen3d -h 2>/dev/null')

    if not os.path.exists(sdf_path) or os.path.getsize(sdf_path) == 0:
        # Fallback: RDKit ETKDG
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            return False
        mol = Chem.RemoveHs(mol)
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()

    # Step 2: SDF -> PDBQT via MGLTools or OpenBabel
    if prepare_ligand_script is None:
        candidates = [
            '/opt/ADFRsuite/CCSBpckgs/AutoDockTools/Utilities24/prepare_ligand4.py',
            '/usr/local/bin/prepare_ligand4.py',
        ]
        for c in candidates:
            if os.path.exists(c):
                prepare_ligand_script = c
                break

    if prepare_ligand_script and os.path.exists(prepare_ligand_script):
        pythonsh = '/opt/ADFRsuite/bin/pythonsh'
        if not os.path.exists(pythonsh):
            pythonsh = 'python2'
        # Fix known basename bug in prepare_ligand4.py
        os.system(f"sed -i 's/ligand_filename = os.path.basename(a)/ligand_filename = a/' {prepare_ligand_script} 2>/dev/null || true")
        cmd = f'{pythonsh} {prepare_ligand_script} -l {sdf_path} -o {pdbqt_path} -A hydrogens_nonpolar'
        os.system(cmd)
    else:
        # OpenBabel fallback
        os.system(f'obabel {sdf_path} -O {pdbqt_path} 2>/dev/null')

    return os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0


def dock_molecule(ligand_pdbqt: str, receptor_pdbqt: str,
                  pocket_box: dict, timeout: int = 60,
                  exhaustiveness: int = 8, num_modes: int = 1) -> Optional[float]:
    """Dock a ligand PDBQT against a receptor PDBQT using AutoDock Vina.

    Args:
        ligand_pdbqt: Path to ligand PDBQT
        receptor_pdbqt: Path to receptor PDBQT
        pocket_box: Dict with center_x/y/z and size_x/y/z
        timeout: Timeout in seconds
        exhaustiveness: Vina exhaustiveness parameter
        num_modes: Number of output modes

    Returns:
        Best Vina score (float) or None on failure
    """
    out_pdbqt = ligand_pdbqt.replace('.pdbqt', '_vina.pdbqt')
    log_file = ligand_pdbqt.replace('.pdbqt', '_vina.log')

    cmd = (
        f'timeout {timeout} vina '
        f'--center_x {pocket_box["center_x"]} '
        f'--center_y {pocket_box["center_y"]} '
        f'--center_z {pocket_box["center_z"]} '
        f'--size_x {pocket_box["size_x"]} '
        f'--size_y {pocket_box["size_y"]} '
        f'--size_z {pocket_box["size_z"]} '
        f'--receptor {receptor_pdbqt} '
        f'--ligand {ligand_pdbqt} '
        f'--out {out_pdbqt} '
        f'--cpu 1 '
        f'--exhaustiveness {exhaustiveness} '
        f'--num_modes {num_modes}'
    )

    os.system(cmd + f' > {log_file} 2>&1')

    # Parse score from log
    if os.path.exists(log_file):
        with open(log_file) as f:
            content = f.read()
        # Match: "   1       -8.500          0          0"
        match = re.search(r'^\s+1\s+(-?\d+\.\d+)', content, re.MULTILINE)
        if match:
            return float(match.group(1))

    return None


def dock_smiles_list(smiles_list: List[str], receptor_pdbqt: str,
                     pocket_box: dict, work_dir: str = '/tmp/vina_docking',
                     timeout: int = 60, exhaustiveness: int = 8) -> List[Optional[float]]:
    """Dock a list of SMILES against a receptor.

    Args:
        smiles_list: List of SMILES strings
        receptor_pdbqt: Path to prepared receptor PDBQT
        pocket_box: Pocket box dictionary
        work_dir: Temporary directory for intermediate files
        timeout: Per-molecule docking timeout
        exhaustiveness: Vina exhaustiveness

    Returns:
        List of Vina scores (None for failed molecules)
    """
    os.makedirs(work_dir, exist_ok=True)
    scores = []

    for i, smiles in enumerate(smiles_list):
        ligand_pdbqt = os.path.join(work_dir, f'ligand_{i}.pdbqt')

        if not prepare_ligand_pdbqt(smiles, ligand_pdbqt):
            scores.append(None)
            continue

        score = dock_molecule(ligand_pdbqt, receptor_pdbqt, pocket_box,
                              timeout=timeout, exhaustiveness=exhaustiveness)
        scores.append(score)

        # Clean up intermediate files
        for ext in ['.smi', '.sdf', '_vina.pdbqt', '_vina.log']:
            fpath = ligand_pdbqt.replace('.pdbqt', ext)
            if os.path.exists(fpath):
                os.remove(fpath)

    return scores


def compute_rmsd_to_crystal(docked_pdbqt: str, crystal_sdf: str) -> Optional[float]:
    """Compute RMSD between a docked pose and the crystal ligand pose.

    Args:
        docked_pdbqt: Path to docked ligand PDBQT
        crystal_sdf: Path to crystal ligand SDF

    Returns:
        RMSD in Angstroms, or None on failure
    """
    try:
        from rdkit.Chem import rdMolAlign
        docked_mol = Chem.MolFromPDBFile(docked_pdbqt, removeHs=True)
        crystal_mol = Chem.SDMolSupplier(crystal_sdf)[0]

        if docked_mol is None or crystal_mol is None:
            return None

        # Need same atom ordering for RMSD
        if docked_mol.GetNumHeavyAtoms() != crystal_mol.GetNumHeavyAtoms():
            return None

        rmsd = rdMolAlign.CalcRMS(docked_mol, crystal_mol)
        return float(rmsd)
    except Exception:
        return None
