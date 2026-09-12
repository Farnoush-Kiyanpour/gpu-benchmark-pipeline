#!/usr/bin/env python3
"""
Wrappers for 5 protein sequence-conditioned tools.
"""
import os
from typing import Dict, List, Tuple
from .base_wrapper import SequenceWrapper


class DrugGPTWrapper(SequenceWrapper):
    """DrugGPT: GPT-2 based drug generation conditioned on protein sequence."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        return (f"python -c \""
                f"from transformers import AutoTokenizer, AutoModelForCausalLM; "
                f"import torch; "
                f"tokenizer = AutoTokenizer.from_pretrained('liyuesen/druggpt'); "
                f"model = AutoModelForCausalLM.from_pretrained('liyuesen/druggpt').to('cuda'); "
                f"seq = open('{inputs['fasta_path']}').read().split('\\n',1)[1].strip()[:1000]; "
                f"prompt = tokenizer.encode(seq, return_tensors='pt').to('cuda'); "
                f"mols = []; "
                f"[mols.append(tokenizer.decode(model.generate(prompt, max_length=512, top_k=5, top_p=0.6, do_sample=True, num_return_sequences={batch_size})[i])) for _ in range({num_samples}//{batch_size})]; "
                f"open('{output_dir}/molecules.smi','w').write('\\n'.join(mols))\"")

    def parse_output(self, output_dir):
        """DrugGPT outputs raw text that needs SMILES extraction."""
        from utils.smiles_standardizer import parse_output as parse_out, _regex_extract_smiles
        smi_path = os.path.join(output_dir, 'molecules.smi')
        if os.path.exists(smi_path):
            return parse_out(smi_path, 'smi')
        return []


class DrugGenWrapper(SequenceWrapper):
    """DrugGen: RL-finetuned GPT-2 for drug generation."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        return (f"python drugGen_generator_cli.py "
                f"--protein_fasta {inputs['fasta_path']} "
                f"--num_molecules {num_samples} "
                f"--batch_size {batch_size} "
                f"--device cuda "
                f"--output {output_dir}/molecules.smi")


class AlphaDrugWrapper(SequenceWrapper):
    """AlphaDrug: MCTS + Transformer for target-specific drug design."""

    def prepare_input(self, target, input_dir):
        inputs = super().prepare_input(target, input_dir)
        # AlphaDrug needs the sequence in a specific format
        return inputs

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        return (f"python generate.py "
                f"--protein_fasta {inputs['fasta_path']} "
                f"--num_samples {num_samples} "
                f"--device cuda "
                f"--out_dir {output_dir}")


class ProtoBindDiffWrapper(SequenceWrapper):
    """ProtoBind-Diff: Protein-conditioned diffusion with ESM embeddings."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        config_path = os.path.join(self.repo_path, 'configs/masked_diffusion.yaml')
        return (f"python protobind_diff/esm_inference.py --config {config_path} "
                f"--protein_fasta {inputs['fasta_path']} "
                f"--num_samples {num_samples} "
                f"--out_dir {output_dir}")


class TargetVAEWrapper(SequenceWrapper):
    """TargetVAE: VAE-based target-conditioned molecule generation."""

    def build_command(self, inputs, output_dir, num_samples=100, batch_size=32):
        return (f"python generate_with_specific_target.py "
                f"--protein_fasta {inputs['fasta_path']} "
                f"--num_samples {num_samples} "
                f"--device cuda "
                f"--out_dir {output_dir}")


SEQUENCE_WRAPPERS = {
    'DrugGPT': DrugGPTWrapper,
    'DrugGen': DrugGenWrapper,
    'AlphaDrug': AlphaDrugWrapper,
    'ProtoBind-Diff': ProtoBindDiffWrapper,
    'TargetVAE': TargetVAEWrapper,
}
