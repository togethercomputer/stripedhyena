# based on https://github.com/EleutherAI/gpt-neox/blob/main/megatron/tokenizer/tokenizer.py
import torch
from tokenizers import Tokenizer
import json
import tqdm
import pathlib


class HFAutoTokenizer:
    def __init__(self, vocab_file):
        self.tokenizer = Tokenizer.from_file(vocab_file)
        self.eos = "</s>"
        self.bos = "<s>"
        self.eos_id = self.tokenize(self.eos)
        self.bos_id = self.tokenize(self.bos)
        self.vsize = 32000

    def encode_to_list(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tokenize_file(self, input_file, output_file, verbose=False):
        if verbose:
            print(f"Tokenizing file: {input_file}")

        if pathlib.Path(output_file).exists():
            print(f"Output file {output_file} already exists, skipping")
            return
        with open(input_file, "r") as fin, open(output_file, "w") as fout:
            for line in tqdm.tqdm(fin):
                if verbose:
                    print(f"Tokenizing line: {line[-200:]}")
                data = json.loads(line.strip())
                if "text" not in data.keys():
                    break
                tokenized_data = self.tokenize(data["text"])
                fout.write(json.dumps({"tokens": tokenized_data}) + "\n")

    def tokenize(self, text: str, *args, **kwargs):
        ids = self.tokenizer.encode(text)
        if type(ids) == list:
            return torch.tensor(ids)
        else:
            return torch.tensor(ids.ids)

    def tokenize_batch(self, text_batch):
        return self.tokenizer.encode_batch(text_batch)

    def detokenize(self, token_ids, skip_special_tokens=False):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def detokenize_batch(self, token_ids_batch, skip_special_tokens=False):
        out = []
        for token_ids in token_ids_batch:
            out.append(
                self.detokenize(
                    [t.item() for t in token_ids], skip_special_tokens=skip_special_tokens
                )
            )
        return out

    @property
    def eod(self):
        return self.eod_id

    @property
    def vocab_size(self):
        return 32000
