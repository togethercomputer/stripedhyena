# StripedHyena

Minimal implementation of a StripedHyena model. 


### Quick Start


#### Standalone 

To run our standalone StripedHyena implementation, you will need to install the packages in `requirements.txt`, as well as rotary and normalization kernels from `flash_attn`. 

The easiest way to ensure all requirements are installed is to build a Docker image using `Dockerfile`, or follow the steps detailed in the Dockerfile itself in a different virtual environment.

Once the environment is set up, you will be able to generate text with:
```
python generate.py --config_path ./configs/7b-sh-32k-base.yml \
--checkpoint_path <path_to_ckpt> --cached_generation \
--prompt_file ./test_promp.txt
```

If you are generating with `prompt.txt`, set `prefill_style: fft` in the config. For very long prompts, you may want to opt for `prefill_style: recurrence`, which will be slower but use less memory.

#### HuggingFace


#### Testing Correctness

We report `lm-evaluation-harness` (10-shot) scores to use as a proxy for (standalone) model correctness in your environment.

* arc_challenge: 0.570 (acc norm)
* hellaswag: 0.816 (acc norm)
* winogrande: 0.735 (acc)

More extensive benchmarks results are provided in the blog post and on HuggingFace.

### Issues

If you encounter any issues, especially with the container, reinstall the latest version of `flash_attn` (including `norm` and `rotary` kernels).


