import argparse

import pytest
import torch
import torch.nn as nn
import yaml

from src.layers import RMSNorm
from src.model import StripedHyena
from src.utils import dotdict


def test_long_prefill(pytestconfig):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # torch.cuda.memory._record_memory_history()

    seqlen = 2000
    config_path = "./configs/7b-sh-32k-v1.yml"
    config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))
    vocab_size = config.vocab_size
    config.max_seqlen = seqlen
    config.prefill_style = "recurrence"
    config.compile = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randint(0, vocab_size, (1, seqlen), device=device, requires_grad=False)
    model = StripedHyena(config)
    model.to_bfloat16_except_poles_residues()
    model = model.to(device)
    model = model.eval()

    # def run_model(x, inference_params_dict_out):
    #     with torch.inference_mode():
    #         logits, inference_params_dict_out = model(
    #             x,
    #             inference_params_dict=inference_params_dict_out,
    #         )
    #     return logits, inference_params_dict_out

    # c_run_model = torch.compile(run_model, fullgraph=False, dynamic=False, mode="reduce-overhead", backend="inductor")

    # with torch.inference_mode():
    #     c_run_model(x, model.initialize_inference_params())
    # assert False

    inference_params_dict_out = model.initialize_inference_params()

    logits_rec, inference_params_dict_out = model(
        x,
        inference_params_dict=inference_params_dict_out,
    )
    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
    assert False


def test_recurrent_prefill(pytestconfig):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    L = 128

    config_path = "./configs/sh-stem-test.yml"
    config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))
    config.use_flashfft = True
    config.seqlen = L
    vocab_size = config.vocab_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randint(0, vocab_size, (1, L), device=device)

    model = StripedHyena(config)
    model.to_bfloat16_except_poles_residues()
    model = model.to(device)
    model = model.eval()

    inference_params_dict_out = model.initialize_inference_params()

    logits_fft, inference_params_dict_out = model(
        x,
        inference_params_dict=inference_params_dict_out,
    )

    # we only test the last iir_state
    state_fft = inference_params_dict_out["hyena"].state_dict[3].clone()

    for module in model.modules():
        if hasattr(module, "config"):
            module.config.prefill_style = "recurrence"

    inference_params_dict_out = model.initialize_inference_params()
    inference_params_dict_out["mha"].seqlen_offset += 1
    inference_params_dict_out["hyena"].seqlen_offset += 1

    logits_rec, inference_params_dict_out = model(
        x,
        inference_params_dict=inference_params_dict_out,
    )

    state_rec = inference_params_dict_out["hyena"].state_dict[3].clone()

    if pytestconfig.getoption("verbose") > 0:
        print(state_fft)
        print(state_rec)
        print(logits_fft)
        print(logits_rec)

    assert torch.allclose(state_fft, state_rec, atol=1e-3)
    assert torch.allclose(logits_fft, logits_rec, atol=1e-4)
