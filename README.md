# StripedHyena

Minimal implementation of a StripedHyena model. 

<p align="center">
  <img src="https://github.com/togethercomputer/stripedhyena/assets/34561392/ce11c785-c8c5-43b2-a34a-41ee9cba3958" width="60%" />
</p>

## About

One of the focus areas at Together Research is new architectures for long context, improved training, and inference performance over the Transformer architecture. Spinning out of a research program from our team and academic collaborators, with roots in **signal processing-inspired sequence models**, we are excited to introduce the **StripedHyena** models. 

StripedHyena is the **first alternative model competitive with the best open-source Transformers** of similar sizes in short and long-context evaluations.

**StripedHyena-Nous-7B (SH-N 7B)** is our **chat model** for this release, and was developed with our collaborators at [Nous Research](https://nousresearch.com/).

- Read more here in [our blog](https://www.together.ai/blog/stripedhyena-7b).
- Play with the model on our [playground](https://api.together.xyz/playground/language/togethercomputer/StripedHyena-Hessian-7B). [Chat here](https://api.together.xyz/playground/chat/togethercomputer/StripedHyena-Nous-7B)!
- Dive into the details of our [standalone implementation](https://github.com/togethercomputer/stripedhyena), and our related research: [1](https://arxiv.org/abs/2302.10866), [2](https://arxiv.org/abs/2310.18780), [3](https://arxiv.org/abs/2311.05908).

SH-N 7B uses this prompt format: `### Instruction:\n{prompt}\n\n### Response:\n{response}`

### Model Architecture

StripedHyena is a hybrid architecture composed of multi-head, grouped-query attention and gated convolutions arranged in [Hyena](https://arxiv.org/abs/2302.10866) blocks, different from traditional decoder-only Transformers.  
  - Costant memory decoding in Hyena blocks via representation of convolutions as state-space models (modal or canonical form), or as truncated filters.
  - Low latency, faster decoding and higher throughput than Transformers. 
  - Improvement to training and inference-optimal scaling laws, compared to optimized Transformer architectures such as Llama-2.
  - Trained on sequences of up to 32k, allowing it to process longer prompts.

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
python generate_transformers.py --model_id <model_id> --input-file ./test_prompt.txt
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

