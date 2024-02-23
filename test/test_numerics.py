import argparse

import pytest
import torch
import torch.nn as nn
import yaml
from src.layers import RMSNorm
from src.utils import dotdict


def test_aa_fp_error(pytestconfig):
    input_dim = 1000
    output_dim = 1000
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    linear = nn.Linear(input_dim, output_dim).to(device).to(dtype)

    x1 = torch.randn(1, input_dim)
    x4 = x1.repeat(4, 1).to(dtype).to(device)

    y1 = linear(x1)
    y4 = linear(x4)

    if pytestconfig.getoption("verbose") > 0:
        print(y1[0])
        print(y4[0])

    assert False


def test_batched_norm(pytestconfig):
    config = {
        "eps": 1e-5,
        "hidden_size": 64,
        "params_dtype": torch.float32,
        "use_flash_rmsnorm": True,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = dotdict(config)
    rmsnorm = RMSNorm(config).to(device).to(torch.bfloat16)

    inputs = torch.randn(1, 64, dtype=torch.bfloat16, device=device)
    inputs = inputs.repeat(4, 1, 1)
    outputs_1 = rmsnorm(inputs[:1])
    outputs_4 = rmsnorm(inputs)

    if pytestconfig.getoption("verbose") > 0:
        print(outputs_1)
        print(outputs_4)

    assert False
