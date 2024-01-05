import argparse
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer


def main(args):
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=sys.maxsize,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.use_cache = True
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
    ).to(device)
    print(args)

    while True:
        if args.input_file is None:
            prompt_text = input("> ")
        else:
            input(f"Press enter to read {args.input_file} ")
            prompt_text = open(args.input_file, encoding="utf=8").read()
            print(prompt_text)

        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        streamer = TextStreamer(tokenizer)
        print(args)
        model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k,
            top_p=args.top_p,
            penalty_alpha=args.penalty_alpha,
            do_sample=args.temperature is not None,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--penalty-alpha", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)

    main(parser.parse_args())
