#!/usr/bin/env python3
"""
Wrappers for 2 pharmacophore/shape-conditioned tools.
"""
import os
from typing import Dict, List, Tuple
from .base_wrapper import PharmacophoreWrapper


class TransPharmerWrapper(PharmacophoreWrapper):
    """TransPharmer: Transformer-based pharmacophore-conditioned generation."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/benchmark.yaml')
        return (f"python generate.py --config {config_path} "
                f"--pdb_path {inputs['pdb_path']} "
                f"--pharmacophore {inputs.get('pharm_path', '')} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class SQUIDWrapper(PharmacophoreWrapper):
    """SQUID: Shape-conditioned molecule generation."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        return (f"python generate.py "
                f"--pdb_path {inputs['pdb_path']} "
                f"--ref_ligand {inputs.get('ref_sdf', '')} "
                f"--num_samples {num_samples} "
                f"--device cuda "
                f"--out_dir {output_dir}")


PHARMACOPHORE_WRAPPERS = {
    'TransPharmer': TransPharmerWrapper,
    'SQUID': SQUIDWrapper,
}
