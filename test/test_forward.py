import argparse

import pytest
import torch
import torch.nn as nn
import yaml
from src.layers import RMSNorm
from src.model import StripedHyena
from src.utils import dotdict
from torch.autograd import grad

try:
    from flashfftconv import FlashFFTConv
except:
    FlashFFTConv = None


def ref_fftconv(x, h):
    fft_s = 2 * x.shape[-1]
    x = x.to(torch.float32)
    h = h.to(torch.float32)
    y = torch.fft.irfft(torch.fft.rfft(x, n=fft_s) * torch.fft.rfft(h, n=fft_s) / fft_s, n=fft_s, norm="forward")
    y = y[..., : x.shape[-1]]
    return y


def test_batched_forward(pytestconfig):
    torch.set_printoptions(precision=16, sci_mode=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_path = "./configs/sh-stem-test.yml"
    config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))
    vocab_size = config.vocab_size
    config

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


# TODO: parametrize for better coverage
def test_custom_fftconv_siso(pytestconfig, dtype=torch.float16):
    L = 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fn = FlashFFTConv(2 * L, dtype=dtype).to(device)

    x = torch.randn(1, 1, L, dtype=dtype).to(device)
    h = torch.randn(1, L, dtype=torch.float32).to(device)
    # mask = torch.exp(-0.2 * torch.arange(0, L, device=device))
    # h = h * mask

    y_fn = fn(x, h)
    y_ref = ref_fftconv(x, h)

    print(y_fn[0, 0, :20])
    print(y_ref[0, 0, :20], end="\n")

    assert torch.allclose(y_fn, y_ref, atol=1e-1)


def test_custom_fftconv_causality(pytestconfig, dtype=torch.float16):
    L = 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fn = FlashFFTConv(2 * L, dtype=dtype).to(device)

    x = torch.randn(1, 1, L, dtype=dtype, requires_grad=True).to(device)
    h = torch.randn(1, L, dtype=torch.float32).to(device)
    y_fn = fn(x, h)

    for i in range(L):
        g = grad(y_fn[0, 0, i], x, retain_graph=True, allow_unused=True)[0]
        print(g.shape, i, g[0, 0, i + 1 :].max())
        assert torch.allclose(g[0, 0, i + 1 :], torch.zeros_like(g[0, 0, i + 1 :]), atol=1e-2), ""


def test_custom_fftconv_hsiso(pytestconfig, dtype=torch.float16):
    L = 128
    D = 16
    H = 4
    M = D // H
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fn = FlashFFTConv(2 * L, dtype=dtype).to(device)

    k = torch.randn(1, D, L, dtype=dtype, device=device)
    v = torch.randn(1, D, L, dtype=dtype, device=device)

    h = 0.1 * torch.randn(1, H * M * M, L, dtype=torch.float32, device=device)
    k = k.reshape(1, H, M, 1, L)
    v = v.reshape(1, H, 1, M, L)
    kv = k * v

    kv_ = kv.reshape(1, -1, L)
    print(kv_.shape, h.shape)
    y_fn = fn(kv_, h)
    y_fn = y_fn.reshape(1, H, M, M, L)

    h = h.reshape(1, H, M, M, L)
    y_ref = ref_fftconv(kv, h)

    print(y_fn[0, 0, :, :, :10])
    print(y_ref[0, 0, :, :, :10], end="\n")

    assert False
