import argparse

import pytest
import torch
import torch.nn as nn
import yaml

from src.layers import RMSNorm
from src.model import StripedHyena
from src.utils import dotdict


@pytest.mark.skip(reason="Not implemented")
def test_batched_forward(pytestconfig):
    torch.set_printoptions(precision=16, sci_mode=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_path = "./configs/sh-stem-test.yml"
    config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))
    vocab_size = config.vocab_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    input_ids = torch.randint(0, vocab_size, (1, 8), device=device)
    input_ids = input_ids.repeat(4, 1)

    model = StripedHyena(config).to(dtype)
    model = model.to(device)
    model = model.eval()

    with torch.no_grad():
        input_ids_1 = input_ids[:1]
        logits_1 = model(input_ids_1)

        input_ids_4 = input_ids
        logits_4 = model(input_ids_4)

    assert torch.allclose(logits_1[0][0], logits_4[0][0])
