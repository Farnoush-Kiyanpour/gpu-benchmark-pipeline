#!/usr/bin/env python3
"""
Wrapper for AutoGrow4 — classical genetic algorithm (CPU-only).
"""
import os
import json
from typing import Dict, List, Tuple
from .base_wrapper import BaseWrapper


class AutoGrow4Wrapper(BaseWrapper):
    """AutoGrow4: Genetic algorithm with click chemistry and Vina docking."""

    def prepare_input(self, target, input_dir):
        """Prepare AutoGrow4 config JSON for the target."""
        os.makedirs(input_dir, exist_ok=True)

        # Protein PDBQT
        pdbqt_path = os.path.join(input_dir, f"{target['name']}_protein.pdbqt")
        pdb_source = os.path.join(self.bench_root, 'targets', 'pdb', f"{target['pdb_id']}.pdb")
        if not os.path.exists(pdbqt_path):
            # Convert PDB to PDBQT
            from utils.vina_docking import prepare_receptor_pdbqt
            prepare_receptor_pdbqt(pdb_source, pdbqt_path)

        # Source compounds (click chemistry fragments)
        source_file = os.path.join(self.repo_path, 'source_compounds', 'clickchem_fragments.smi')
        if not os.path.exists(source_file):
            # Use default fragments
            source_file = os.path.join(self.repo_path, 'source_compounds', 'Fragment_MW_up_to_100.smi')

        # Create config JSON
        pb = target['pocket_box']
        config = {
            "source_compound_file": source_file,
            "filename_of_receptor": pdbqt_path,
            "num_generations": 5,
            "number_of_mutants_first_generation": 20,
            "number_of_crossovers_first_generation": 15,
            "number_of_mutants": 20,
            "number_of_crossovers": 15,
            "number_elitism_advance_from_previous_gen_first_generation": 5,
            "number_elitism_advance_from_previous_gen": 5,
            "top_mols_to_seed_next_generation_first_generation": 15,
            "top_mols_to_seed_next_generation": 15,
            "diversity_mols_to_seed_first_generation": 10,
            "diversity_seed_depreciation_per_gen": 0.1,
            "number_of_processors": 1,
            "multithread_mode": "serial",
            "selector_choice": "Roulette_Selector",
            "dock_choice": "VinaDocking",
            "docking_executable": "vina",
            "scoring_choice": "VINA",
            "docking_exhaustiveness": 4,
            "docking_num_modes": 1,
            "conversion_choice": "MGLToolsConversion",
            "mgltools_directory": "/opt/ADFRsuite/",
            "mgl_python": "/opt/ADFRsuite/bin/pythonsh",
            "prepare_ligand4.py": "/opt/ADFRsuite/CCSBpckgs/AutoDockTools/Utilities24/prepare_ligand4.py",
            "prepare_receptor4.py": "/opt/ADFRsuite/CCSBpckgs/AutoDockTools/Utilities24/prepare_receptor4.py",
            "max_variants_per_compound": 1,
            "gypsum_thoroughness": 1,
            "gypsum_timeout_limit": 30,
            "docking_timeout_limit": 60,
            "filter_source_compounds": "True",
            "start_a_new_run": "True",
            "LipinskiStrictFilter": True,
            "reduce_file_sizes": "False",
            "generate_plot": "False",
            "cache_prerun": "True",
            "debug_mode": False,
            "root_output_folder": f"{input_dir}/output/",
            "center_x": pb["center_x"],
            "center_y": pb["center_y"],
            "center_z": pb["center_z"],
            "size_x": pb["size_x"],
            "size_y": pb["size_y"],
            "size_z": pb["size_z"],
        }

        config_path = os.path.join(input_dir, f"{target['name']}_autogrow4.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        return {'config_path': config_path, 'pdb_path': pdb_source}

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        return f"python run_autogrow.py -j {inputs['config_path']}"

    def parse_output(self, output_dir):
        """Parse AutoGrow4 ranked .smi files from all generations."""
        import glob
        from utils.smiles_standardizer import parse_smi_file

        all_mols = []
        run_dir = os.path.join(output_dir, 'Run_0')
        if not os.path.isdir(run_dir):
            # Check if output_dir IS the run dir
            run_dir = output_dir

        for ranked_file in sorted(glob.glob(os.path.join(run_dir, 'generation_*/generation_*_ranked.smi'))):
            mols = parse_smi_file(ranked_file)
            all_mols.extend(mols)

        # Deduplicate
        seen = set()
        unique = []
        for smiles, name in all_mols:
            if smiles not in seen:
                seen.add(smiles)
                unique.append((smiles, name))

        return unique

    def is_gpu_required(self):
        return False


CLASSICAL_WRAPPERS = {
    'AutoGrow4': AutoGrow4Wrapper,
}
