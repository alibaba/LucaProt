#!/usr/bin/env python
# encoding: utf-8
'''
*Copyright (c) 2023, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.

@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2023/3/20 13:23
@project: DeepProtFunc
@file: predict_structure
@desc: xxxx
'''

import os
import sys
import esm
import torch
import random
from timeit import default_timer as timer
from esm import BatchConverter, pretrained
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
from torch.distributed.fsdp.wrap import enable_wrap, wrap
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../src")
try:
    from utils import fasta_reader
    from biotoolbox.structure_file_reader import *
    from biotoolbox.contact_map_builder import *
    from biotoolbox.contact_map_generator import *
except ImportError:
    from src.utils import fasta_reader
    from src.biotoolbox.structure_file_reader import *
    from src.biotoolbox.contact_map_builder import *
    from src.biotoolbox.contact_map_generator import *


def enable_cpu_offloading(model):
    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:%d" % (7000 + random.randint(0, 1000)), world_size=1, rank=0
    )
    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model


def init_model_on_gpu_with_cpu_offloading(model):
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model


def predict_pdb(sample, trunc_type, num_recycles=4, truncation_seq_length=4096, chunk_size=64, cpu_type="cpu-offload"):
    '''
    use sequence to predict protein 3D-structure
    :param sample:
    :param trunc_type:
    :param num_recycles:
    :param truncation_seq_length:
    :param chunk_size:
    :param cpu_type:
    :return: pdb, mean_plddt, ptm, processed_seq_len
    '''
    assert cpu_type is None or cpu_type in ["cpu-offload", "cpu-only"]
    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    model.set_chunk_size(chunk_size)
    if cpu_type == "cpu_only":
        model.esm.float()  # convert to fp32 as ESM-2 in fp16 is not supported on CPU
        model.cpu()
    elif cpu_type == "cpu_offload":
        model = init_model_on_gpu_with_cpu_offloading(model)
    else:
        model.cuda()
    start = timer()
    protein_id, protein_seq = sample[0], sample[1]
    if len(protein_seq) > truncation_seq_length:
        if trunc_type == "left":
            protein_seq = protein_seq[-truncation_seq_length:]
        else:
            protein_seq = protein_seq[:truncation_seq_length]
    cur_seq_len = len(protein_seq)
    processed_seq = protein_seq[:truncation_seq_length] if cur_seq_len > truncation_seq_length else protein_seq
    with torch.no_grad():
        try:
            output = model.infer([processed_seq], num_recycles=num_recycles)
            output = {key: value.cpu() for key, value in output.items()}
            mean_plddt = output["mean_plddt"][0]
            ptm = output["ptm"][0]
            pdb = model.output_to_pdb(output)[0]
            use_time = timer() - start
            print("predict pdb use time: %f" % use_time)
            return pdb, mean_plddt, ptm, processed_seq
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {sample[0]} of length {len(sample[1])}.")
            else:
                print(e)
    return None, None, None, None


def calc_distance_maps(pdb_filepath, chain, sequence):
    """
    Use the pdb file (3d-structure) of the protein to construct the adjacent Euclidean space distance matrix (diagonalized) between amino acids (residues),
        including two ways C_alpha (alpha carbon) and C_beta (beta carbon)
    :param pdb_filepath:
    :param chain: specified chain name
    :param sequence: the amino acids sequence
    :return: contact map
    """
    if ".cif" in pdb_filepath:
        ca = {}
        ca[chain] = {}
        ca[chain]['contact-map'] = ContactMap(pdb_filepath, None, chain=chain, c_atom_type="CA")
        cb = {}
        cb[chain] = {}
        cb[chain]['contact-map'] = ContactMap(pdb_filepath, None, chain=chain, c_atom_type="CB")
        return ca, cb
    else:
        pdb_handle = None
        if os.path.exists(pdb_filepath): # from file
            pdb_handle = open(pdb_filepath, 'r')
            pdb_content = pdb_handle.read()
        else:
            # input is pdb content
            pdb_content = pdb_filepath
        structure_container = build_structure_container_for_pdb(pdb_content, chain).with_seqres(sequence)
        # structure_container.chains = {chain: structure_container.chains[chain]}

        mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)  # start with CA distances
        ca = mapper.generate_map_for_pdb(structure_container)
        cb = mapper.set_atom("CB").generate_map_for_pdb(structure_container)
        if pdb_handle:
            pdb_handle.close()

        return ca.chains, cb.chains


model, alphabet = None, None


def predict_embedding(sample, trunc_type, embedding_type, repr_layers=[-1], truncation_seq_length=4094, device=None):
    '''
    use sequence to predict protein embedding matrix or vector(bos)
    :param sample: [protein_id, protein_sequence]
    :param trunc_type:
    :param embedding_type: bos or representations
    :param repr_layers: [-1]
    :param truncation_seq_length: [4094,2046,1982,1790,1534,1278,1150,1022]
    :param device:
    :return: embedding, processed_seq_len
    '''
    global model, alphabet
    assert embedding_type in ["bos", "representations", "matrix"]
    protein_id, protein_seq = sample[0], sample[1]
    if len(protein_seq) > truncation_seq_length:
        if trunc_type == "left":
            protein_seq = protein_seq[-truncation_seq_length:]
        else:
            protein_seq = protein_seq[:truncation_seq_length]
    if model is None or alphabet is None:
        model, alphabet = pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    else:
        model_device = next(model.parameters()).device
        if device != model_device:
            model = model.to(device)
    """
    if torch.cuda.is_available():
        model = model.cuda()
        # print("Transferred model to GPU")
    """
    converter = BatchConverter(alphabet, truncation_seq_length)
    protein_ids, raw_seqs, tokens = converter([[protein_id, protein_seq]])
    with torch.no_grad():
        # if torch.cuda.is_available():
        # tokens = tokens.to(device="cuda", non_blocking=True)
        tokens = tokens.to(device=device, non_blocking=True)
        try:
            out = model(tokens, repr_layers=repr_layers, return_contacts=False)
            truncate_len = min(truncation_seq_length, len(raw_seqs[0]))
            if embedding_type in ["representations", "matrix"]:
                embedding = out["representations"][36].to(device="cpu")[0, 1: truncate_len + 1].clone().numpy()
            else:
                embedding = out["representations"][36].to(device="cpu")[0, 0].clone().numpy()
            return embedding, protein_seq
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {sample[0]} of length {len(sample[1])}.")
                print("Please reduce the 'truncation_seq_length'")
            if device.type == "cpu":
                # insufficient cpu memory
                raise Exception(e)
            else:
                # failure in GPU, return None to continue using CPU
                return None, None


if __name__ == "__main__":
    '''
    >act_sludge_ORF9_SRR7976301_k141_125069_flag1_multi4_len1195
    SDVCSSDLVNFAVEVPIEDYVAETVIAEFKETLDIQMGSNAGLRSDLMLPLLCEHRNVES
    NGKGRKATLARYGELDPRDQVYGPEDLNDFLGMLEDEVKKLNVVEKRLTTRGCPQSHVLS
    FPTLNAINYAAFLGARHFFPDLKGQGYGDDFIGSSESREACELVLKAREGFGMMTNTTAT
    GISRGGERGLAVFCEMVFSTLDGSLIENAKPKPVNAFFRSLHANATMTGYVTHVEDLMDV
    SRMLEKRHQEVFKIIKDGVEPSKPTMAYRQTYPRFILNRIIRKDVGIDSAIATKRSAEEI
    FAFLESQILLFTPSSKNAPARVDPRSQSVGLLVSQEFHRKMSRALERYSNTKEFKFRDFP
    AFANAGIALLAIRKVFVDEDMEEIKGILGIRQ
    '''
    seq = "SDVCSSDLVNFAVEVPIEDYVAETVIAEFKETLDIQMGSNAGLRSDLMLPLLCEHRNVESNGKGRKATLARYGELDPRDQVYGPEDLNDFLGMLEDEVKKLNVVEKRL" \
          "TTRGCPQSHVLSFPTLNAINYAAFLGARHFFPDLKGQGYGDDFIGSSESREACELVLKAREGFGMMTNTTATGISRGGERGLAVFCEMVFSTLDGSLIENAKPKPVNAFFRS" \
          "LHANATMTGYVTHVEDLMDVSRMLEKRHQEVFKIIKDGVEPSKPTMAYRQTYPRFILNRIIRKDVGIDSAIATKRSAEEIFAFLESQILLFTPSSKNAPARVDPRSQSVGLLVSQ" \
          "EFHRKMSRALERYSNTKEFKFRDFPAFANAGIALLAIRKVFVDEDMEEIKGILGIRQ"
    chain = "A"
    c_alpha, c_beta = calc_distance_maps("../../pdbs/protein_1.pdb", chain, seq)
    cmap_thresh = 10
    cmap = c_alpha[chain]['contact-map']
    cmap = np.less_equal(cmap, cmap_thresh).astype(np.int32)
    print(cmap)
    print(c_alpha[chain]['contact-map'])
    print(c_beta[chain]['contact-map'])




