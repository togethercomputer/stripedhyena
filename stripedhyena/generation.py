# Copyright (c) Together
# This software is distributed under the terms of the Apache License, Version 2.0
# Author: Michael Poli

# Barebones generation class for standalone inference.

import torch

from stripedhyena.sample import sample
from stripedhyena.tokenizer import CharLevelTokenizer
from stripedhyena.utils import print_rank_0


class Generator:
    def __init__(self, model, tokenizer, top_k=50, top_p=0.7, temperature=1):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.untils = ["\n\n"]

    def generate(
        self,
        device,
        input_string=None,
        input_ids=None,
        num_tokens=32,
        cached_generation=False,
        print_generation=True,
        verbose=False,
        skip_special_tokens=False,
        stop_at_eos=True,
        max_seqlen=None,
    ):
        if isinstance(self.tokenizer.eos, int):
            eos_token_ids = torch.LongTensor([self.tokenizer.eos]).to(device)
        else:
            # is a tensor
            eos_token_ids = self.tokenizer.tokenize(self.tokenizer.eos).to(device)

        if input_ids is None:
            input = self.tokenizer.tokenize(input_string)
            if isinstance(input, list):
                input = torch.LongTensor(input).unsqueeze(0).to(device)
            # is a tensor
            else:
                input = input.unsqueeze(0).to(device)

        else:
            input = input_ids
        x = input

        if max_seqlen is not None:
            x = x[:, -max_seqlen:]

        prompt_len = x.shape[-1]

        num_tokens = int(num_tokens)
        tot_length = prompt_len + num_tokens
        batch_size = x.shape[0]

        generation = torch.empty(
            x.shape[0],
            num_tokens,
            dtype=torch.long,
            device=x.device,
        )

        scores = torch.empty(
            x.shape[0],
            num_tokens,
            self.tokenizer.vocab_size,
            dtype=torch.float,
            device=x.device,
        )

        if cached_generation:
            inference_params_dict_out = self.model.initialize_inference_params()
            inference_params_dict_out["mha"].max_batch_size = batch_size
            inference_params_dict_out["hyena"].max_batch_size = batch_size
        else:
            inference_params_dict_out = None

        if verbose:
            mem_after_tok = torch.cuda.memory_allocated(device=x.device) / 1e9
            print_rank_0(f"Memory after tokenization: {mem_after_tok} GB")
            print_rank_0("Starting generation...")
            if input_string is not None:
                print_rank_0("Prompt: " + input_string)
            else:
                print_rank_0(f"Prompt ids: {input_ids} {input_ids.shape}")

        for i in range(int(num_tokens)):
            post_prefill = cached_generation and i > 0
            # prefill then process only the last token
            if post_prefill:
                x = x[:, -1:]
                seqlen_offset = inference_params_dict_out["mha"].seqlen_offset

                if seqlen_offset == 0:
                    seqlen_offset = input.shape[-1]
                    inference_params_dict_out["hyena"].seqlen_offset = seqlen_offset
                    inference_params_dict_out["mha"].seqlen_offset = seqlen_offset
                else:
                    inference_params_dict_out["mha"].seqlen_offset += 1
                    inference_params_dict_out["hyena"].seqlen_offset += 1

            # do forward pass with no gradient
            with torch.no_grad():
                logits, inference_params_dict_out = self.model(
                    x,
                    inference_params_dict=inference_params_dict_out,
                )

            last_logits = logits[:, -1]

            new_idx = sample(
                last_logits,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
            )

            if stop_at_eos and (generation[0, -2:] == eos_token_ids).all():
                print_rank_0("Stopping generation at EOS")

            if print_generation and verbose and batch_size == 1:
                print_rank_0(
                    f"{self.tokenizer.detokenize([new_idx.item()])}",
                    end=" ",
                )

            scores[:, i] = last_logits
            generation[:, i] = new_idx

            if post_prefill:
                x = new_idx[:, None]
            else:
                x = torch.cat([x, new_idx[:, None]], dim=-1)

        if verbose:
            kwargs = {}
            if not isinstance(self.tokenizer, CharLevelTokenizer):
                kwargs["skip_special_tokens"] = skip_special_tokens
            y = self.tokenizer.detokenize_batch(generation[:, : i + 1], **kwargs)

            for until in self.untils:
                if until in y:
                    y = y.split(until)[0]
                    break

            print_rank_0(f"\nInput: {input_string}, Output: {y}")

            mem_end = torch.cuda.memory_allocated(device=x.device) / 1e9
            print_rank_0(f"Memory after generation: {mem_end} GB")

        return generation[:, : i + 1], scores[:, : i + 1]
