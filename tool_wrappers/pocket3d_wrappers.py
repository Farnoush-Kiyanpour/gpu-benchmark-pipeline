#!/usr/bin/env python3
"""
Wrappers for all 25 3D-pocket-conditioned tools.
Each tool inherits from Pocket3DWrapper and overrides tool-specific logic as needed.
"""
import os
import json
import copy
from typing import Dict, List, Tuple
from .base_wrapper import Pocket3DWrapper


class TargetDiffWrapper(Pocket3DWrapper):
    """TargetDiff: 3D Equivariant Diffusion (ICLR 2023)."""

    def get_config_overrides(self, target: dict) -> dict:
        return {
            'config_path': os.path.join(self.repo_path, 'configs/sampling.yml'),
            'center_pos_mode': 'protein',
        }

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/sampling.yml')
        return (f"python scripts/sample_for_pocket.py {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--batch_size {batch_size} "
                f"--result_path {output_dir}")


class DecompDiffWrapper(Pocket3DWrapper):
    """DecompDiff: Decomposed Diffusion (ICLR 2024)."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/sampling_drift.yml')
        return (f"python scripts/sample_for_pocket.py {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--batch_size {batch_size} "
                f"--result_path {output_dir}")


class DiffSBDDWrapper(Pocket3DWrapper):
    """DiffSBDD: Diffusion-based SBDD."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        ckpt = os.path.join(self.repo_path, self.config.get('weights', {}).get('path', 'checkpoints/crossdock_ca_cond.ckpt'))
        return (f"python generate_ligands.py {ckpt} "
                f"--pdbfile {inputs['pdb_path']} "
                f"--n_samples {num_samples} "
                f"--batch_size {min(batch_size, num_samples)} "
                f"--outfile {output_dir}/molecules.sdf --sanitize")


class Pocket2MolWrapper(Pocket3DWrapper):
    """Pocket2Mol: Efficient molecular sampling based on 3D protein pockets."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/sample_for_pdb.yml')
        return (f"python sample_for_pdb.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--batch_size {batch_size} "
                f"--result_path {output_dir}")


class GraphBPWrapper(Pocket3DWrapper):
    """GraphBP: Graph-based 3D molecule generation in binding pockets."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/sample.yml')
        return (f"python GraphBP/main_eval.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--output_dir {output_dir}")


class SBDD3DWrapper(Pocket3DWrapper):
    """3D-SBDD: 3D structure-based drug design."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/sample.yml')
        return (f"python models/sample.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--result_path {output_dir}")


class MolCRAFTWrapper(Pocket3DWrapper):
    """MolCRAFT: BFN-based SBDD in continuous parameter space (ICML 2024)."""

    def prepare_input(self, target, input_dir):
        inputs = super().prepare_input(target, input_dir)
        # MolCRAFT needs both protein PDB and reference ligand SDF
        if 'ref_sdf' not in inputs:
            from utils.smiles_standardizer import generate_3d_conformer
            mol = generate_3d_conformer(target['drug_smiles'])
            if mol is not None:
                ref_sdf = os.path.join(input_dir, f"{target['name']}_ref.sdf")
                from rdkit import Chem
                writer = Chem.SDWriter(ref_sdf)
                writer.write(mol)
                writer.close()
                inputs['ref_sdf'] = ref_sdf
        return inputs

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'MolCRAFT/configs/sample.yml')
        return (f"python MolCRAFT/sample_for_pocket.py --config {config_path} "
                f"--protein_path {inputs['pdb_path']} "
                f"--ligand_path {inputs.get('ref_sdf', '')} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class MolPilotWrapper(Pocket3DWrapper):
    """MolPilot: Optimal scheduling for SBDD (ICML 2025)."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'MolPilot/configs/sample.yml')
        return (f"python MolPilot/sample_for_pocket.py --config {config_path} "
                f"--protein_path {inputs['pdb_path']} "
                f"--ligand_path {inputs.get('ref_sdf', '')} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class IPDiffWrapper(Pocket3DWrapper):
    """IPDiff: Interaction-aware diffusion for SBDD."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/sampling.yml')
        return (f"python sample_split.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--result_path {output_dir}")


class IRDiffWrapper(Pocket3DWrapper):
    """IRDiff: Inverse-regularized diffusion for SBDD."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/sampling.yml')
        return (f"python sample_split.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--result_path {output_dir}")


class DrugFlowWrapper(Pocket3DWrapper):
    """DrugFlow: Equivariant flow matching for SBDD."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/generate.yml')
        return (f"python src/generate.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class DualDiffWrapper(Pocket3DWrapper):
    """DualDiff: Dual-path diffusion for SBDD."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/sampling.yml')
        return (f"python scripts/sample_for_pocket.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--result_path {output_dir}")


class LiGANWrapper(Pocket3DWrapper):
    """LiGAN: Generative models for structure-based drug design (voxel-based)."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/generate.yml')
        return (f"python generate.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class DrugHIVEWrapper(Pocket3DWrapper):
    """DrugHIVE: Hierarchical variational inference for SBDD."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'config/generate.yml')
        return (f"python generate_molecules.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class Lingo3DMolWrapper(Pocket3DWrapper):
    """Lingo3DMol: 3D molecule generation with linguistic representations."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/inference.yml')
        return (f"python inference/inference.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class TamGentWrapper(Pocket3DWrapper):
    """TamGent: Target-aware molecule generation with 3D transformer."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        return (f"python eval_lm.py --pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class PAFlowWrapper(Pocket3DWrapper):
    """PAFlow: Position-aware flow matching for SBDD."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/sampling_guide.yml')
        return (f"python scripts/sample_for_pocket.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--result_path {output_dir}")


class SILVRWrapper(Pocket3DWrapper):
    """SILVR: Scaffold-guided latent variable regression for SBDD."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/generate.yml')
        return (f"python generate.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class MolSnapperWrapper(Pocket3DWrapper):
    """MolSnapper: Fragment-based molecule generation in pockets."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/generate.yml')
        return (f"python scripts/generate.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class DESERTWrapper(Pocket3DWrapper):
    """DESERT: Distance-aware pocket-to-molecule generation."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/generating.yaml')
        return (f"python generate.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class PocketFlowWrapper(Pocket3DWrapper):
    """PocketFlow: Flow-based generation in protein pockets."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/generate.yml')
        return (f"python main_generate.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class SMILES3DGPTWrapper(Pocket3DWrapper):
    """3DSMILES-GPT: GPT-based 3D molecule generation."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/smiles_pocket.yml')
        return (f"python sample_from_ligand_msms.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class Mol3DFormerWrapper(Pocket3DWrapper):
    """3DMolFormer: Transformer-based 3D molecule generation."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        return (f"python generation_finetuning.py "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class PocketXMolWrapper(Pocket3DWrapper):
    """PocketXMol: Cross-modal molecule generation in pockets."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/generate.yml')
        return (f"python evaluate/generate.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class DeepBlockWrapper(Pocket3DWrapper):
    """DeepBlock: Block-based molecule generation in pockets."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/generate.yml')
        return (f"python generate.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


# Registry mapping tool names to wrapper classes
POCKET3D_WRAPPERS = {
    'TargetDiff': TargetDiffWrapper,
    'DecompDiff': DecompDiffWrapper,
    'DiffSBDD': DiffSBDDWrapper,
    'Pocket2Mol': Pocket2MolWrapper,
    'GraphBP': GraphBPWrapper,
    '3D-SBDD': SBDD3DWrapper,
    'MolCRAFT': MolCRAFTWrapper,
    'MolPilot': MolPilotWrapper,
    'IPDiff': IPDiffWrapper,
    'IRDiff': IRDiffWrapper,
    'DrugFlow': DrugFlowWrapper,
    'DualDiff': DualDiffWrapper,
    'LiGAN': LiGANWrapper,
    'DrugHIVE': DrugHIVEWrapper,
    'Lingo3DMol': Lingo3DMolWrapper,
    'TamGent': TamGentWrapper,
    'PAFlow': PAFlowWrapper,
    'SILVR': SILVRWrapper,
    'MolSnapper': MolSnapperWrapper,
    'DESERT': DESERTWrapper,
    'PocketFlow': PocketFlowWrapper,
    '3DSMILES-GPT': SMILES3DGPTWrapper,
    '3DMolFormer': Mol3DFormerWrapper,
    'PocketXMol': PocketXMolWrapper,
    'DeepBlock': DeepBlockWrapper,
}
