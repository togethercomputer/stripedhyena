# Copyright (c) 2023, Michael Poli.

# Checkpoint conversion code from safari-neox or savanna to this repo's format


import os
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml 

import glob
from collections import OrderedDict
from src.utils import dotdict
from src.mpu import initialize_model_parallel, print_rank_0
from src.model import StripedHyena
# from src.tokenizer import HFAutoTokenizer
from src.tokenizer import CharLevelTokenizer
from src.generation import Generator
import argparse


KEY_UPDATE_DICT_HYENA = {
    # ssm
    "attention.mixer.filter.kernel.B": "filter.B",  # savanna style
    "attention.mixer.filter.kernel.C": "filter.C",  # savanna style
    "attention.mixer.long_conv_bias": "filter.D",  # savanna style
    "attention.mixer.filter.kernel.log_dt": "filter.log_dt",  # savanna style
    "attention.mixer.filter.kernel.inv_A_real": "filter.inv_A_real",  # savanna style
    "attention.mixer.filter.kernel.A_imag": "filter.A_imag",  # savanna style
    # short conv
    "attention.hyena_proj_conv.short_conv_weight": "filter.short_filter_weight",  # savanna style
    "attention.hyena_proj_conv.short_conv_bias": "filter.short_filter_bias",  # savanna style
    "attention.mixer.short_conv_weight": "",
    "attention.hyena_proj_conv.weight": "filter.short_filter_weight",
    "attention.hyena_proj_conv.bias": "filter.short_filter_bias",
    # rope
    "attention.rotary_emb.inv_freq": "rotary_emb.inv_freq",
    # qkv proj
    "attention.query_key_value.weight": "projections.weight",  # savanna style
    "attention.query_key_value.bias": "projections.bias",  # savanna style

    # mlp
    "mlp.w1.weight": "mlp.l1.weight",  # savanna style
    "mlp.w2.weight": "mlp.l2.weight",  # dont forget to swap w2 and w3!  for 7B SH model
    "mlp.w3.weight": "mlp.l3.weight",  # dont forget to swap w2 and w3!  for 7B SH model
    # dense layers
    "attention.dense.weight": "out_filter_dense.weight",  # savanna style
    "attention.dense.bias": "out_filter_dense.bias",  # savanna style
    # to scrap
    "mlp.w1._extra_state": "",
    "mlp.w2._extra_state": "",
    "mlp.w3._extra_state": "",
    "attention.dense._extra_state": "",
    "post_attention_layernorm.scale": "",
    "outer_mlp_layernorm.scale": "",
    "attention.query_key_value._extra_state": "",
    #  layer norm.  # confirmed same order: norm, mixer, norm, mlp, residual
    "input_layernorm.scale": "pre_norm.scale",  # savanna style
    "pre_mlp_layernorm.scale": "post_norm.scale",  # savanna style
    # 
    'attention.mixer.filter.act.freq': "",
    'attention.mixer.filter.pos_emb.t': "",
    'attention.mixer.filter.pos_emb.z': "",
    'attention.mixer.filter.implicit_filter.0.weight': "",
    'attention.mixer.filter.implicit_filter.0.bias': "",
    'attention.mixer.filter.implicit_filter.1.freq': "",
    'attention.mixer.filter.implicit_filter.2.weight': "",
    'attention.mixer.filter.implicit_filter.2.bias': "",
    'attention.mixer.filter.implicit_filter.3.freq': "",
    'attention.mixer.filter.implicit_filter.4.weight': "",
    'attention.mixer.filter.implicit_filter.4.bias': "",
    'attention.mixer.filter.implicit_filter.5.freq': "",
    'attention.mixer.filter.final_filter.weight': "",
    'attention.mixer.filter.modulation.weight': "",
    # 
    'mlp.gate_proj.weight': "mlp.l1.weight",
    'mlp.up_proj.weight': "mlp.l2.weight",
    'mlp.down_proj.weight': "mlp.l3.weight",
    # misc
    "word_embeddings.weight": "word_embeddings.weight",  # place to make sure it's added to new state dict
    "norm.scale": "norm.scale",  # place to make sure it's added to new state dict
}


KEY_UPDATE_DICT_ATTENTION = {
    "attention.query_key_value.weight": "inner_mha_cls.Wqkv.weight",  # savanna style
    "attention.query_key_value.bias": "inner_mha_cls.Wqkv.bias",  # savanna style
    "attention.dense.weight": "inner_mha_cls.out_proj.weight",  # savanna style
    "attention.dense.bias": "inner_mha_cls.out_proj.bias",  # savanna style
    "attention.o_proj.weight": "inner_mha_cls.out_proj.weight",
    "attention.o_proj.bias": "inner_mha_cls.out_proj.bias",
    # rope
    "attention.rotary_emb.inv_freq": "inner_mha_cls.rotary_emb.inv_freq",  # savanna style, dummy var, will be empty
    # "attention.rotary_emb.inv_freq": "",  
    # to scrap
    "mlp.w1._extra_state": "",
    "mlp.w2._extra_state": "",
    "mlp.w3._extra_state": "",
    "attention.dense._extra_state": "",
    "post_attention_layernorm.scale": "",    # savanna style
    "outer_mlp_layernorm.scale": "",  # savanna style
    "attention.query_key_value._extra_state": "",
    'attention.q_proj.weight': "", 
    'attention.k_proj.weight': "", 
    'attention.v_proj.weight': "",
    # mlp
    "mlp.w1.weight": "mlp.l1.weight",  # savanna style
    "mlp.w2.weight": "mlp.l2.weight",  # dont forget to swap w2 and w3!  for 7B SH model
    "mlp.w3.weight": "mlp.l3.weight",  # dont forget to swap w2 and w3!  for 7B SH model
    #  layer norm
    "input_layernorm.scale": "pre_norm.scale",  # savanna style
    "pre_mlp_layernorm.scale": "post_norm.scale",  # savanna style
    # 
    'mlp.gate_proj.weight': "mlp.l1.weight",
    'mlp.up_proj.weight': "mlp.l2.weight",
    'mlp.down_proj.weight': "mlp.l3.weight",
    # misc
    "final_linear.weight": "word_embeddings.weight",
    "word_embeddings.weight": "word_embeddings.weight",  # place to make sure it's added to new state dict
    "norm.weight": "norm.scale",
    "norm.scale": "norm.scale",  # place to make sure it's added to new state dict
}

# main conversion logic here...
def filter_state_dict(state_dict):
    
    block_type = "attention"

    # automatic (brittle) detection of attention vs hyena
    keys = list(state_dict.keys())
    # print(keys)
    
    # set key update to attention by default
    KEY_UPDATE_DICT = KEY_UPDATE_DICT_ATTENTION
    
    # loop through all keys, check if the word 'hyena' is in any part of any of the keys in the list, then change key update dict to hyena
    for key in keys:
        if "hyena" in key:
            KEY_UPDATE_DICT = KEY_UPDATE_DICT_HYENA
            # print("Detected hyena block -------------------------------- *****************")
            block_type = "hyena"
            break

    print("************* Block type: {}".format(block_type))

    # # not sure what this does
    params_to_merge = [
        'attention.q_proj.weight', 'attention.k_proj.weight', 'attention.v_proj.weight']

    new_state_dict = OrderedDict()
    params_to_merge_values = []
    
    # loops thru all state dict keys in old format
    for k in state_dict.keys():
        # removes some "module." prefix from keys
        if k.startswith("module."):
            k = k[7:]
        
        # find key in dict, if not found, print, if found, then swap keys
        if k in KEY_UPDATE_DICT.keys():
            new_k = KEY_UPDATE_DICT[k]
            
            if new_k == "":
                print("{} found, but scrapping this!".format(k))
                continue
            else:
                print(f"Found key: {k}, swapping to {new_k}")
        # not found
        else:
            print(f"Key {k} not found in dict, skipping ***********************!!!!")
            continue
        
        if new_k != "":
            # print(state_dict[k].shape)  # don't need to print
            new_state_dict[new_k] = state_dict[k]
            
        # breakpoint()

    # convert modal SSM parameters to pole / residue form
    if "filter.inv_A_real" in new_state_dict.keys():
        print("Detected alternative parametrization")
        A_real = -torch.exp(new_state_dict["filter.inv_A_real"])
        A_imag = new_state_dict["filter.A_imag"]

        dt = torch.exp(new_state_dict["filter.log_dt"])[:, None]
        dt_A = dt * (A_real + 1j * A_imag)
        C = new_state_dict["filter.C"]
        C = torch.view_as_complex(C.to(torch.float32))
        B = new_state_dict["filter.B"]
        B = torch.view_as_complex(B.to(torch.float32))

        # pop old keys
        new_state_dict.pop("filter.inv_A_real")
        new_state_dict.pop("filter.A_imag")
        new_state_dict.pop("filter.log_dt")
        new_state_dict.pop("filter.B")
        new_state_dict.pop("filter.C")

        residues = 2 * B * C * (1.0 - dt_A / 2).reciprocal() * dt
        poles = (1.0 + dt_A / 2) / (1.0 - dt_A / 2)

        new_state_dict["filter.poles"] = torch.view_as_real(poles).squeeze()[:,:,None]
        new_state_dict["filter.residues"] = torch.view_as_real(residues).squeeze()[:,:,None]

    return new_state_dict 


def checkpoint_conversion(checkpoint_path, new_checkpoint_path, converted_checkpoint_path):
    # loads checkpoint in deepspeed format ("layer-{idx}-model_00-model_states.pt")
    # assumes model parallel 1
    files = glob.glob(os.path.join(checkpoint_path, "layer*states.pt"))
    files = sorted(files, key=lambda x: int(x.split("/")[-1].split("_")[1].split("-")[0]))
    for idx, file in enumerate(files):
        state_dict = torch.load(file)
        # print(f"Loading {file}, keys: {state_dict.keys()}", end="\n\n")
        
        state_dict = filter_state_dict(state_dict)

        new_file = file.split("/")[-1]
        torch.save(state_dict, os.path.join(new_checkpoint_path, f"layer_{idx:02d}.pt"))


def main():
    
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="/home/cirrascale/etnguyen/checkpoints/350m_striped_20k/global_step20000", help="Path to original checkpoint dir")
    parser.add_argument("--config_path", type=str, default="/home/cirrascale/etnguyen/stripedhyena/sh_inference_config.yml", help="Path to new config")
    parser.add_argument("--converted_checkpoint_path", type=str, default="/home/cirrascale/etnguyen/checkpoints/350m_striped_20k", help="Path to converted checkpoint output dir")
    parser.add_argument("--string", type=str, default="ACTG", help="String to generate from, uppercase only")
    parser.add_argument("--num_tokens", type=int, default=32, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=1, help="Top k for sampling")
    parser.add_argument("--top_p", type=float, default=1, help="Top p for sampling")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose printing")
    parser.add_argument("--force_convert", type=bool, default=False, help="Force conversion of checkpoint")
    parser.add_argument("--stop_at_eos", type=bool, default=True, help="Stop generation at EOS token")
    args = parser.parse_args()
    
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    config = dotdict(yaml.load(open(args.config_path), Loader=yaml.FullLoader))

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6029"
    torch.distributed.init_process_group(backend="nccl")
    initialize_model_parallel(1)

    model = StripedHyena(config)
    print("Loaded new state dict")

    complete_ckpt = "complete_ckpt.pt"

    # loop thru all files in a directory path, and check if the phrase "complete_ckpt.pt" is in the filename
    files = os.listdir(args.converted_checkpoint_path)

    found = False

    if args.force_convert:
        print("Forcing conversion...")
        found = False
    else:  # will check if already converted
        for file_name in files:
            if complete_ckpt in file_name:
                model.load_state_dict(torch.load(os.path.join(args.converted_checkpoint_path, file_name)), strict=True)
                found = True
                print("Complete checkpoint loaded!!!")
                break

    # if complete ckpt not found, then start the conversion process
    # this will load the original sharded ckpts, convert them to the new key format, load layer-by-layer into the model, then write a single complete ckpt to file
    if not found:
        print("Converting checkpoint, then loading layer-by-layer")
        # turn on conversion if it hasn't been done yet, which will save per layer and the complete ckpt
        checkpoint_conversion(args.checkpoint_path, args.converted_checkpoint_path, args.converted_checkpoint_path)
        
        print("Loading layer-by-layer....!")
        model.load_from_split_converted_state_dict(args.converted_checkpoint_path)
        
        # save complete ckpt for next time
        torch.save(model.state_dict(), os.path.join(args.converted_checkpoint_path, "complete_ckpt.pt"))
        

    # tokenizer = HFAutoTokenizer(config.vocab_file)
    tokenizer = CharLevelTokenizer(config.vocab_size)  # 512 for now
    device = torch.device("cuda:0")

    # with open("prompt.txt", "r") as f:
    #     string = f.read()
    #     print(args.string)

    input_ids = None #torch.load('./x_test_v2/original_input_ids_2', map_location='cpu').to(device) 

    # input = tokenizer.tokenize(args.string)
    # input_tensor = torch.tensor(input).to(device).unsqueeze(0)

    # print(input_tensor.shape)
    model.to_bfloat16_except_poles_residues()
    model = model.to(device)
    
    # put in eval mode
    model.eval()

    g = Generator(model, tokenizer, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)
    output_ids, logits = g.generate(
        num_tokens=args.num_tokens,
        cached_generation=True,
        input_string=args.string,
        input_ids=input_ids,
        device=device,
        print_generation=True,
        verbose=args.verbose,
        stop_at_eos=args.stop_at_eos,
    )
    
    # breakpoint()  # main inspection for full model

    # slice_len = 2048
    # # idea: evaluate perplexity of next token prediction of the last slice of `slice_len` tokens
    # # by sweeping different lengths of the input (starting from the end)
    # with torch.inference_mode():
    #     for seqlen in [2049, 4096, 8192, 16384, 32768, 65536, 131072]:
    #         print(f"Sequence length: {seqlen}")
    #         input_slice = input[-seqlen:]
    #         input_slice = input_slice.unsqueeze(0)
    #         print(f"Input slice: {input_slice.shape}")
    #         outs = m(input_slice)
    #         logits = outs[0]
    #         logits = logits[:, -slice_len - 1: -1, :] # slice last prediction
    #         target = input_slice[:, -slice_len:]
    #         print(f"Logits: {logits.shape}, target: {target.shape}")
    #         perplexity = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction='none')
    #         print(f"Perplexity: {perplexity.mean()}")

if __name__ == "__main__":
    main()