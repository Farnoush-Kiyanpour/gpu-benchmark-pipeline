#!/usr/bin/env python3
"""
Base wrapper class for all tool inference wrappers.
Each tool category inherits from this and implements tool-specific logic.
"""
import os
import json
import copy
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod


class BaseWrapper(ABC):
    """Abstract base class for tool inference wrappers.

    Each subclass implements:
    - prepare_input(): Convert target data to tool-specific input format
    - build_command(): Construct the inference command string
    - parse_output(): Extract SMILES from tool output
    """

    def __init__(self, tool_name: str, tool_config: dict, bench_root: str):
        self.tool_name = tool_name
        self.config = tool_config
        self.bench_root = bench_root
        self.repo_path = os.path.join(bench_root, tool_config.get('repo_path', ''))
        self.output_format = tool_config.get('output_format', 'sdf')
        self.input_format = tool_config.get('input_format', 'protein_pdb')

    @abstractmethod
    def prepare_input(self, target: dict, input_dir: str) -> Dict[str, str]:
        """Prepare tool-specific input files for a target.

        Args:
            target: Target metadata dict (name, pdb_id, pocket_box, etc.)
            input_dir: Directory to write prepared input files

        Returns:
            Dict mapping placeholder names to file paths, e.g.:
            {'pdb_path': '/path/to/protein.pdb', 'fasta_path': '/path/to/seq.fasta'}
        """
        pass

    @abstractmethod
    def build_command(self, inputs: Dict[str, str], output_dir: str,
                      num_samples: int = 100, batch_size: int = 32) -> str:
        """Build the inference command string.

        Args:
            inputs: Dict of input file paths from prepare_input()
            output_dir: Directory for output files
            num_samples: Number of molecules to generate
            batch_size: Batch size for generation

        Returns:
            Command string to execute
        """
        pass

    @abstractmethod
    def parse_output(self, output_dir: str) -> List[Tuple[str, str]]:
        """Parse tool output and extract SMILES.

        Args:
            output_dir: Directory containing tool output

        Returns:
            List of (smiles, name) tuples
        """
        pass

    def get_config_overrides(self, target: dict) -> Dict[str, Any]:
        """Return per-target config overrides (e.g., pocket box in config file).

        Override in subclass if the tool needs target-specific config files.
        """
        return {}

    def post_process(self, smiles_list: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Post-process molecules (e.g., filter, canonicalize).

        Default: no post-processing. Override in subclass if needed.
        """
        return smiles_list

    def get_gpu_memory_gb(self) -> int:
        """Return required GPU memory in GB."""
        return self.config.get('gpu_memory_gb', 8)

    def get_timeout_hours(self) -> int:
        """Return job timeout in hours."""
        return self.config.get('timeout_hours', 2)

    def is_gpu_required(self) -> bool:
        """Whether this tool requires a GPU."""
        return self.config.get('gpu_required', True)


class Pocket3DWrapper(BaseWrapper):
    """Base wrapper for 3D pocket-conditioned tools.

    Common pattern: takes a protein PDB file, generates molecules in the binding pocket.
    Output is typically SDF files.
    """

    def prepare_input(self, target: dict, input_dir: str) -> Dict[str, str]:
        """Prepare protein PDB for pocket-based tools.

        Most 3D pocket tools just need the protein PDB file.
        Some need a pocket-only PDB (subtracted around the ligand).
        """
        os.makedirs(input_dir, exist_ok=True)

        # Copy/symlink the protein PDB
        pdb_source = os.path.join(self.bench_root, 'targets', 'pdb', f"{target['pdb_id']}.pdb")
        pdb_dest = os.path.join(input_dir, f"{target['name']}_protein.pdb")

        if os.path.exists(pdb_source):
            if not os.path.exists(pdb_dest):
                os.system(f'cp {pdb_source} {pdb_dest}')
        else:
            # Try alternate location
            alt_source = os.path.join(self.bench_root, 'targets', 'pdb', f"{target['name']}.pdb")
            if os.path.exists(alt_source):
                os.system(f'cp {alt_source} {pdb_dest}')

        inputs = {'pdb_path': pdb_dest}

        # Some tools need a reference ligand SDF for pocket extraction
        if 'protein_pdb_sdf' in self.input_format or 'protein_pdb_interaction' in self.input_format:
            ref_sdf = os.path.join(self.bench_root, 'targets', 'reference_ligands',
                                   f"{target['name']}_{target['pdb_id']}_reference.sdf")
            if os.path.exists(ref_sdf):
                inputs['ref_sdf'] = ref_sdf
            else:
                # Generate from SMILES
                from utils.smiles_standardizer import generate_3d_conformer
                mol = generate_3d_conformer(target['drug_smiles'])
                if mol is not None:
                    ref_sdf = os.path.join(input_dir, f"{target['name']}_ref.sdf")
                    writer = __import__('rdkit').Chem.SDWriter(ref_sdf)
                    writer.write(mol)
                    writer.close()
                    inputs['ref_sdf'] = ref_sdf

        # Some tools need pharmacophore features
        if 'pharmacophore' in self.input_format:
            pharm_path = os.path.join(self.bench_root, 'targets', 'reference_ligands',
                                      f"{target['name']}_{target['pdb_id']}_pharmacophore.json")
            if os.path.exists(pharm_path):
                inputs['pharm_path'] = pharm_path

        return inputs

    def build_command(self, inputs: Dict[str, str], output_dir: str,
                      num_samples: int = 100, batch_size: int = 32) -> str:
        """Build command from template in config."""
        cmd_template = self.config.get('inference_cmd', '')
        cmd = cmd_template.format(
            pdb_path=inputs.get('pdb_path', ''),
            ref_sdf=inputs.get('ref_sdf', ''),
            fasta_path=inputs.get('fasta_path', ''),
            pharm_path=inputs.get('pharm_path', ''),
            config_path=inputs.get('config_path', ''),
            target_name=inputs.get('target_name', ''),
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=output_dir,
            checkpoint=self.config.get('weights', {}).get('path', ''),
        )
        return cmd

    def parse_output(self, output_dir: str) -> List[Tuple[str, str]]:
        """Parse SDF output from 3D pocket tools."""
        from utils.smiles_standardizer import parse_output as parse_out

        output_path = self.config.get('output_path', '{output_dir}/*.sdf')
        full_path = output_path.format(output_dir=output_dir)

        return parse_out(full_path, self.output_format)


class SequenceWrapper(BaseWrapper):
    """Base wrapper for protein sequence-conditioned tools.

    Common pattern: takes a protein FASTA sequence, generates molecules.
    Output is typically SMILES files.
    """

    def prepare_input(self, target: dict, input_dir: str) -> Dict[str, str]:
        """Prepare protein FASTA for sequence-based tools."""
        os.makedirs(input_dir, exist_ok=True)

        # Find or create FASTA file
        fasta_source = os.path.join(self.bench_root, 'targets', 'sequences',
                                    f"{target['name']}_uniprot.fasta")
        fasta_dest = os.path.join(input_dir, f"{target['name']}.fasta")

        if os.path.exists(fasta_source):
            os.system(f'cp {fasta_source} {fasta_dest}')
        else:
            # Create from UniProt ID
            with open(fasta_dest, 'w') as f:
                f.write(f">{target['uniprot']}|{target['name']}\n")
                f.write(target.get('sequence', '') + '\n')

        return {'fasta_path': fasta_dest}

    def build_command(self, inputs: Dict[str, str], output_dir: str,
                      num_samples: int = 100, batch_size: int = 32) -> str:
        cmd_template = self.config.get('inference_cmd', '')
        cmd = cmd_template.format(
            fasta_path=inputs.get('fasta_path', ''),
            pdb_path=inputs.get('pdb_path', ''),
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=output_dir,
        )
        return cmd

    def parse_output(self, output_dir: str) -> List[Tuple[str, str]]:
        from utils.smiles_standardizer import parse_output as parse_out
        output_path = self.config.get('output_path', '{output_dir}/*.smi')
        full_path = output_path.format(output_dir=output_dir)
        return parse_out(full_path, self.output_format)


class InteractionWrapper(BaseWrapper):
    """Base wrapper for protein-ligand interaction-conditioned tools.

    Common pattern: takes protein PDB + reference ligand, generates molecules
    guided by interaction patterns.
    """

    def prepare_input(self, target: dict, input_dir: str) -> Dict[str, str]:
        """Prepare protein PDB + reference ligand for interaction-based tools."""
        os.makedirs(input_dir, exist_ok=True)

        # Protein PDB
        pdb_source = os.path.join(self.bench_root, 'targets', 'pdb', f"{target['pdb_id']}.pdb")
        pdb_dest = os.path.join(input_dir, f"{target['name']}_protein.pdb")
        if os.path.exists(pdb_source):
            os.system(f'cp {pdb_source} {pdb_dest}')

        # Reference ligand SDF
        ref_sdf = os.path.join(self.bench_root, 'targets', 'reference_ligands',
                               f"{target['name']}_{target['pdb_id']}_reference.sdf")
        if not os.path.exists(ref_sdf):
            from utils.smiles_standardizer import generate_3d_conformer
            mol = generate_3d_conformer(target['drug_smiles'])
            if mol is not None:
                ref_sdf = os.path.join(input_dir, f"{target['name']}_ref.sdf")
                writer = __import__('rdkit').Chem.SDWriter(ref_sdf)
                writer.write(mol)
                writer.close()

        return {
            'pdb_path': pdb_dest,
            'ref_sdf': ref_sdf,
            'target_name': target['name'],
        }

    def build_command(self, inputs: Dict[str, str], output_dir: str,
                      num_samples: int = 100, batch_size: int = 32) -> str:
        cmd_template = self.config.get('inference_cmd', '')
        cmd = cmd_template.format(
            pdb_path=inputs.get('pdb_path', ''),
            ref_sdf=inputs.get('ref_sdf', ''),
            target_name=inputs.get('target_name', ''),
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=output_dir,
        )
        return cmd

    def parse_output(self, output_dir: str) -> List[Tuple[str, str]]:
        from utils.smiles_standardizer import parse_output as parse_out
        output_path = self.config.get('output_path', '{output_dir}/*.sdf')
        full_path = output_path.format(output_dir=output_dir)
        return parse_out(full_path, self.output_format)


class PharmacophoreWrapper(BaseWrapper):
    """Base wrapper for pharmacophore/shape-conditioned tools."""

    def prepare_input(self, target: dict, input_dir: str) -> Dict[str, str]:
        """Prepare pharmacophore features and protein PDB."""
        os.makedirs(input_dir, exist_ok=True)

        # Protein PDB
        pdb_source = os.path.join(self.bench_root, 'targets', 'pdb', f"{target['pdb_id']}.pdb")
        pdb_dest = os.path.join(input_dir, f"{target['name']}_protein.pdb")
        if os.path.exists(pdb_source):
            os.system(f'cp {pdb_source} {pdb_dest}')

        # Reference ligand SDF for shape
        ref_sdf = os.path.join(self.bench_root, 'targets', 'reference_ligands',
                               f"{target['name']}_{target['pdb_id']}_reference.sdf")
        if not os.path.exists(ref_sdf):
            from utils.smiles_standardizer import generate_3d_conformer
            mol = generate_3d_conformer(target['drug_smiles'])
            if mol is not None:
                ref_sdf = os.path.join(input_dir, f"{target['name']}_ref.sdf")
                writer = __import__('rdkit').Chem.SDWriter(ref_sdf)
                writer.write(mol)
                writer.close()

        # Pharmacophore features
        pharm_path = os.path.join(self.bench_root, 'targets', 'reference_ligands',
                                  f"{target['name']}_{target['pdb_id']}_pharmacophore.json")
        if not os.path.exists(pharm_path):
            # Generate basic pharmacophore from reference ligand
            pharm_path = os.path.join(input_dir, f"{target['name']}_pharm.json")
            _generate_pharmacophore(target['drug_smiles'], pharm_path)

        return {
            'pdb_path': pdb_dest,
            'ref_sdf': ref_sdf,
            'pharm_path': pharm_path,
        }

    def build_command(self, inputs: Dict[str, str], output_dir: str,
                      num_samples: int = 100, batch_size: int = 32) -> str:
        cmd_template = self.config.get('inference_cmd', '')
        cmd = cmd_template.format(
            pdb_path=inputs.get('pdb_path', ''),
            ref_sdf=inputs.get('ref_sdf', ''),
            pharm_path=inputs.get('pharm_path', ''),
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=output_dir,
        )
        return cmd

    def parse_output(self, output_dir: str) -> List[Tuple[str, str]]:
        from utils.smiles_standardizer import parse_output as parse_out
        output_path = self.config.get('output_path', '{output_dir}/*.smi')
        full_path = output_path.format(output_dir=output_dir)
        return parse_out(full_path, self.output_format)


def _generate_pharmacophore(smiles: str, output_path: str):
    """Generate basic pharmacophore features from a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return

    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    except Exception:
        return

    # Extract pharmacophore-like features
    features = {
        'smiles': smiles,
        'num_atoms': mol.GetNumHeavyAtoms(),
        'hbd': rdMolDescriptors.CalcNumHBD(mol),
        'hba': rdMolDescriptors.CalcNumHBA(mol),
        'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
    }

    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2)
