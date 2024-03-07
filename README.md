# StripedHyena

Minimal implementation of a StripedHyena model. 

<p align="center">
  <img src="https://github.com/togethercomputer/stripedhyena/assets/34561392/ce11c785-c8c5-43b2-a34a-41ee9cba3958" width="60%" />
</p>

## About

StripedHyena is the **first alternative model architecture competitive with the best open-source Transformers** of similar sizes in short and long-context evaluations.

StripedHyena is a deep signal processing, hybrid architecture composed of rotary (grouped) attention and gated convolutions arranged in [Hyena](https://arxiv.org/abs/2302.10866) blocks, with improved scaling over decoder-only Transformers. 
StripedHyena is designed to leverage the specialization of each of its layer classes, with Hyena layers implementing the bulk of the computation required for sequence processing and attention layers supplementing the ability to perform targeted pattern recall.

- Efficient autoregressive generation via a recurrent mode (>500k generation with a single 80GB GPU)
- Low latency, faster decoding and higher throughput than Transformers.
- Significantly faster training and finetuning at long context (>3x at 131k)
- Improved scaling laws over state-of-the-art architectures (e.g., Transformer++) on both natural language and biological sequences.
- Robust to training beyond the compute-optimal frontier e.g., training way beyond Chinchilla-optimal token amounts

## Models

### Biology: Evo-1-7B

**Evo** is a **biological foundation model** capable of long-context modeling and design.

Evo uses the StripedHyena architecture to enable modeling of sequences at a single-nucleotide, byte-level resolution with near-linear scaling of compute and memory relative to context length. Evo has 7 billion parameters and is trained on OpenGenome, a prokaryotic whole-genome dataset containing ~300 billion tokens.

- Read more here in the [preprint](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v1.full.pdf).

### Language: StripedHyena-7B

**StripedHyena-Nous-7B (SH-N 7B)** is our **chat model** for this release, and was developed with our collaborators at [Nous Research](https://nousresearch.com/).

- Read more here in [our blog](https://www.together.ai/blog/stripedhyena-7b).
- Play with the model on our [playground](https://api.together.xyz/playground/language/togethercomputer/StripedHyena-Hessian-7B). [Chat here](https://api.together.xyz/playground/chat/togethercomputer/StripedHyena-Nous-7B)!
- Dive into the details of our [standalone implementation](https://github.com/togethercomputer/stripedhyena), and our related research: [1](https://arxiv.org/abs/2302.10866), [2](https://arxiv.org/abs/2310.18780), [3](https://arxiv.org/abs/2311.05908).

SH-N 7B uses this prompt format: `### Instruction:\n{prompt}\n\n### Response:\n{response}`


## Quick Start

The most direct way to test StripedHyena models is via our playground, which includes a variety
of architecture-specific optimizations. 

Playground:


### Standalone 

#### Checkpoints

We provide a checkpoint for StripedHyena-Hessian 7B, our base model. Download `pytorch-model.bin` from the [HuggingFace repository](https://huggingface.co/togethercomputer/StripedHyena-Hessian-7B). As an alternative, we also provide HuggingFace compatible checkpoints for AutoClasses.

#### Environment Setup

To run our standalone StripedHyena implementation, you will need to install the packages in `requirements.txt`, as well as rotary and normalization kernels from `flash_attn`. 

The easiest way to ensure all requirements are installed is to build a Docker image using `Dockerfile`, or follow the steps detailed in the Dockerfile itself in a different virtual environment. For example, to build a Docker image, run:
```
docker build --tag sh:test .
```
Installing the dependencies and kernels could take several minutes. Then run the container interactively with:
```
docker run -it --gpus all --network="host" --shm-size 900G -v=<path_to_this_repo>:/mnt:rw --rm sh:test
```

#### Environment Setup

Once the environment is set up, you will be able to generate text with:
```
python generate.py --config_path ./configs/7b-sh-32k-v1.yml \
--checkpoint_path <path_to_ckpt> --cached_generation \
--prompt_file ./test_prompt.txt
```

If you are generating with `prompt.txt`, set `prefill_style: fft` in the config. For very long prompts, you may want to opt for `prefill_style: recurrence`, which will be slower but use less memory.

If the installation was correct, test prompt will generate the following paragraph
```
The four species of hyenas are the striped hyena (Hyaena hyaena), the brown hyena (Parahyaena brunnea), the spotted hyena (Crocuta crocuta), and the aardwolf (Proteles cristata).\n\nThe striped hyena is the most widespread species, occurring in Africa, the Middle East, and Asia.
```

### HuggingFace

We also provide an entry script to generate with StripedHyena models hosted on HuggingFace. The model ids are:

* Base model: `togethercomputer/StripedHyena-Hessian-7B`
* Chat model: `togethercomputer/StripedHyena-Nous-7B`

Choose your model id, then run the following command:
```
python generate_transformers.py --model-name <model_id> --input-file ./test_prompt.txt
```

## Testing Correctness

We report `lm-evaluation-harness` (10-shot) scores to use as a proxy for (standalone) model correctness in your environment.

* arc_challenge: 0.570 (acc norm)
* hellaswag: 0.816 (acc norm)
* winogrande: 0.735 (acc)

More extensive benchmarks results are provided in the blog post and on HuggingFace.

### Optional Dependencies

The standalone implementation provides integration with some custom kernels for StripedHyena such as [FlashFFTConv](https://github.com/HazyResearch/flash-fft-conv) (see the model config `7b-sh-32k-v1.yml` for more information). These additional kernels are not required to run the model.

## Issues

Several issues can be resolved by reinstalling the latest version of `flash_attn` (`pip freeze | grep flash-attn` should return a version `>= 2.0.0`).

StripedHyena is a mixed precision model. Make sure to keep your `poles` and `residues` in `float32` precision.

## Cite

If have found the pretrained models or architecture useful for you research or application, consider citing: 
```
@software{stripedhyena,
  title        = {{StripedHyena: Moving Beyond Transformers with Hybrid Signal Processing Models}},
  author       = { Poli, Michael and Wang, Jue and Massaroli, Stefano and Quesnelle, Jeffrey and Carlow, Ryan and Nguyen, Eric and Thomas, Armin},
  month        = 12,
  year         = 2023,
  url          = { https://github.com/togethercomputer/stripedhyena },
  doi          = { 10.57967/hf/1595 },
}
```
