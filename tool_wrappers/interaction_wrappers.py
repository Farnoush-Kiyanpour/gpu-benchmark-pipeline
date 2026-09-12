#!/usr/bin/env python3
"""
Wrappers for 3 protein-ligand interaction-conditioned tools.
"""
import os
from typing import Dict, List, Tuple
from .base_wrapper import InteractionWrapper


class DrugGENWrapper(InteractionWrapper):
    """DrugGEN: GAN-based target-specific drug design with graph transformers."""

    def prepare_input(self, target, input_dir):
        inputs = super().prepare_input(target, input_dir)
        # DrugGEN uses target protein ID for pre-trained models
        inputs['target_name'] = target['name']
        return inputs

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        return (f"python inference.py "
                f"--target {inputs['target_name']} "
                f"--num_samples {num_samples} "
                f"--device cuda "
                f"--out_dir {output_dir}")


class DeepICLWrapper(InteractionWrapper):
    """DeepICL: Interaction-conditioned ligand generation (Nat Commun 2024)."""

    def prepare_input(self, target, input_dir):
        inputs = super().prepare_input(target, input_dir)
        # DeepICL needs interaction-conditioned pocket data
        # Generate demo-style input from PDB + reference ligand
        inputs['config_path'] = os.path.join(self.repo_path, 'configs/generate.yml')
        return inputs

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = inputs.get('config_path', os.path.join(self.repo_path, 'configs/generate.yml'))
        return (f"python script/generate.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--ref_ligand {inputs.get('ref_sdf', '')} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class Assay2MolWrapper(InteractionWrapper):
    """Assay2Mol: Assay-conditioned molecule generation."""

    def prepare_input(self, target, input_dir):
        inputs = super().prepare_input(target, input_dir)
        # Assay2Mol needs assay data; use reference ligand as proxy
        return inputs

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        return (f"python generate.py "
                f"--pdb_path {inputs['pdb_path']} "
                f"--ref_ligand {inputs.get('ref_sdf', '')} "
                f"--num_samples {num_samples} "
                f"--device cuda "
                f"--out_dir {output_dir}")


INTERACTION_WRAPPERS = {
    'DrugGEN': DrugGENWrapper,
    'DeepICL': DeepICLWrapper,
    'Assay2Mol': Assay2MolWrapper,
}
