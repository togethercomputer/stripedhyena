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
    # config.prefill_style = "recurrence"
    config.compile = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fix the input
    x = torch.ones(1, seqlen, device=device, requires_grad=False, dtype=torch.long)
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
    # latest
    # tensor([[[ -39.7500,   64.5000,  -85.5000,  ...,   38.7500,  -76.5000,
    #           -131.0000],
    #          [ -93.5000,   45.7500,  -99.5000,  ...,   58.0000,  -68.0000,
    #           -145.0000],
    #          [ -78.0000,   32.5000, -121.0000,  ...,  -10.7500, -127.0000,
    #            -44.2500],
    #          ...,
    #          [ -44.0000,   96.5000, -130.0000,  ...,   88.5000, -256.0000,
    #            -34.2500],
    #          [ -44.2500,   96.5000, -128.0000,  ...,   88.0000, -258.0000,
    #            -32.7500],
    #          [ -42.2500,   96.5000, -127.5000,  ...,   88.0000, -256.0000,
    #            -32.0000]]], device='cuda:0', dtype=torch.bfloat16,
    #        grad_fn=<UnsafeViewBackward0>)

    # 2095a87b98fb58b67603fbf29effda0d958b6627
    # tensor([[[  12.3125,  -68.5000,    6.9688,  ..., -134.0000,   43.5000,
    #           -113.0000],
    #          [ -11.1875, -126.5000,  139.0000,  ...,  -82.5000,  -58.5000,
    #            -89.0000],
    #          [ -68.0000,  -15.8125,   48.5000,  ...,  -76.5000,  -98.5000,
    #             22.0000],
    #          ...,
    #          [ -34.7500,  -99.0000,  122.0000,  ...,  -98.5000,  -61.0000,
    #             10.9375],
    #          [ -36.2500,  -96.5000,  121.5000,  ...,  -97.0000,  -60.5000,
    #             10.4375],
    #          [ -35.2500,  -97.0000,  120.5000,  ...,  -98.0000,  -61.7500,
    #              9.8750]]], device='cuda:0', dtype=torch.bfloat16,
    #        grad_fn=<UnsafeViewBackward0>)

    print(logits_rec)
    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
    assert False


def test_recurrent_prefill(pytestconfig):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    L = 128

    config_path = "./configs/sh-stem-test.yml"
    config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))
    config.use_flashfft = False  # True
    config.seqlen = L
    vocab_size = config.vocab_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.ones(1, L, device=device, requires_grad=False, dtype=torch.long)

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

    # latest
    # tensor([[[ 0.1718-1.7217e-04j,  0.1700+1.1340e-04j],
    #          [-0.2949+2.2665e-04j, -0.2977-1.4746e-05j],
    #          [ 2.7513+2.4811e-03j,  2.7531-2.2736e-03j],
    #          ...,
    #          [-0.0897-1.1076e-04j, -0.0894+5.4259e-06j],
    #          [ 0.0879+9.2938e-05j,  0.0885-1.3169e-04j],
    #          [ 0.0057-4.8962e-07j,  0.0057-7.6752e-06j]]], device='cuda:0',
    #        grad_fn=<CloneBackward0>)
    # tensor([[[ 0.1729-1.7357e-04j,  0.1709+1.1396e-04j],
    #          [-0.2910+2.2411e-04j, -0.2930-1.4484e-05j],
    #          [ 2.7812+2.5024e-03j,  2.7812-2.3041e-03j],
    #          ...,
    #          [-0.0913-1.1253e-04j, -0.0908+5.5134e-06j],
    #          [ 0.0879+9.2983e-05j,  0.0884-1.3161e-04j],
    #          [ 0.0064-5.5879e-07j,  0.0064-8.8215e-06j]]], device='cuda:0',
    #        grad_fn=<CloneBackward0>)
    # tensor([[[ -59.2500,   30.5000,  -84.0000,  ...,  -56.5000,  -67.0000,
    #            -55.5000],
    #          [ -24.8750,   69.5000,  -63.5000,  ..., -104.5000,  -50.7500,
    #             -1.6875],
    #          [ -17.2500,   67.0000, -118.0000,  ...,  -64.0000,  -41.0000,
    #             15.9375],
    #          ...,
    #          [ -36.0000,   78.0000, -134.0000,  ...,  -26.8750,   -2.0625,
    #             58.2500],
    #          [ -36.0000,   78.0000, -134.0000,  ...,  -26.8750,   -2.0625,
    #             58.5000],
    #          [ -35.7500,   78.0000, -134.0000,  ...,  -27.0000,   -2.0938,
    #             58.2500]]], device='cuda:0', dtype=torch.bfloat16,
    #        grad_fn=<UnsafeViewBackward0>)
    # tensor([[[ -59.2500,   30.3750,  -84.5000,  ...,  -56.2500,  -66.5000,
    #            -55.5000],
    #          [ -25.1250,   70.0000,  -63.5000,  ..., -104.0000,  -51.0000,
    #             -1.7969],
    #          [ -18.2500,   67.0000, -119.0000,  ...,  -63.7500,  -40.2500,
    #             15.7500],
    #          ...,
    #          [ -36.7500,   78.0000, -133.0000,  ...,  -27.2500,   -2.0156,
    #             58.2500],
    #          [ -36.5000,   77.5000, -134.0000,  ...,  -27.1250,   -1.0703,
    #             58.5000],
    #          [ -36.7500,   78.0000, -133.0000,  ...,  -27.2500,   -2.0156,
    #             58.2500]]], device='cuda:0', dtype=torch.bfloat16,
    #        grad_fn=<UnsafeViewBackward0>)

    # 2095a87b98fb58b67603fbf29effda0d958b6627
    # tensor([[[ 0.1718-1.7217e-04j,  0.1700+1.1340e-04j],
    #          [-0.2949+2.2665e-04j, -0.2977-1.4746e-05j],
    #          [ 2.7513+2.4811e-03j,  2.7531-2.2736e-03j],
    #          ...,
    #          [-0.0897-1.1076e-04j, -0.0894+5.4259e-06j],
    #          [ 0.0879+9.2938e-05j,  0.0885-1.3169e-04j],
    #          [ 0.0057-4.8962e-07j,  0.0057-7.6752e-06j]]], device='cuda:0',
    #        grad_fn=<CloneBackward0>)
    # tensor([[[ 0.1718+0.1727j,  0.1700+0.1730j],
    #          [-0.2949-0.2927j, -0.2977-0.2930j],
    #          [ 2.7513+2.7525j,  2.7531+2.7477j],
    #          ...,
    #          [-0.0897-0.0904j, -0.0894-0.0903j],
    #          [ 0.0879+0.0885j,  0.0885+0.0882j],
    #          [ 0.0057+0.0057j,  0.0057+0.0057j]]], device='cuda:0',
    #        grad_fn=<CloneBackward0>)
    # tensor([[[ -59.2500,   30.5000,  -84.0000,  ...,  -56.5000,  -67.0000,
    #            -55.5000],
    #          [ -24.8750,   69.5000,  -63.5000,  ..., -104.5000,  -50.7500,
    #             -1.6875],
    #          [ -17.2500,   67.0000, -118.0000,  ...,  -64.0000,  -41.0000,
    #             15.9375],
    #          ...,
    #          [ -36.0000,   78.0000, -134.0000,  ...,  -26.8750,   -2.0625,
    #             58.2500],
    #          [ -36.0000,   78.0000, -134.0000,  ...,  -26.8750,   -2.0625,
    #             58.5000],
    #          [ -35.7500,   78.0000, -134.0000,  ...,  -27.0000,   -2.0938,
    #             58.2500]]], device='cuda:0', dtype=torch.bfloat16,
    #        grad_fn=<UnsafeViewBackward0>)
    # tensor([[[ -59.2500,   30.5000,  -84.0000,  ...,  -56.5000,  -67.0000,
    #            -55.5000],
    #          [ -24.8750,   69.5000,  -63.5000,  ..., -104.5000,  -50.7500,
    #             -1.6875],
    #          [ -17.2500,   67.0000, -118.0000,  ...,  -64.0000,  -41.0000,
    #             15.9375],
    #          ...,
    #          [ -36.0000,   78.0000, -134.0000,  ...,  -26.8750,   -2.0625,
    #             58.2500],
    #          [ -36.0000,   78.0000, -134.0000,  ...,  -26.8750,   -2.0625,
    #             58.5000],
    #          [ -35.7500,   78.0000, -134.0000,  ...,  -27.0000,   -2.0938,
    #             58.2500]]], device='cuda:0', dtype=torch.bfloat16,
    #        grad_fn=<UnsafeViewBackward0>)

    assert torch.allclose(state_fft, state_rec, atol=1e-3)
    assert torch.allclose(logits_fft, logits_rec, atol=1e-4)
