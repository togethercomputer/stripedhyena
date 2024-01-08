import torch 
import pytest 
import yaml 
import argparse

from src.model import StripedHyena
from src.utils import dotdict


def test_batched_forward(pytestconfig):
    config_path = "./configs/sh-stem-test.yml"
    config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))
    vocab_size = config.vocab_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 
    input_ids = torch.randint(0, vocab_size, (1, 8), device=device)
    input_ids = input_ids.repeat(4, 1)

    model = StripedHyena(config).to(dtype)
    model = model.to(device)
    model = model.eval()

    if pytestconfig.getoption('verbose') > 0:
        print([p for p in model.parameters()])

    with torch.no_grad():
        input_ids_1 = input_ids[:1]
        logits_1 = model(input_ids_1)
        if pytestconfig.getoption('verbose') > 0:
            print(logits_1)

        input_ids_4 = input_ids
        logits_4 = model(input_ids_4)
        if pytestconfig.getoption('verbose') > 0:
            print(logits_4)
    
    assert torch.allclose(logits_1[0], logits_4[0])