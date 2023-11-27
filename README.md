# StripedHyena

Minimal implementation of a StripedHyena model. 


### Quick Start


#### Standalone 

To run our standalone StripedHyena implementation, you will need to install the packages in `requirements.txt`, as well as rotary and normalization kernels from `flash_attn`. 

The easiest way to ensure all requirements are installed is to build a Docker image using `Dockerfile`, or follow the steps detailed in the Dockerfile itself in a different virtual environment. For example, to build a Docker image, run:
```
docker build --tag sh:test .
```
Installing the dependencies and kernels could take several minutes. Then run the container interactively with:
```
docker run -it --gpus all --network="host" --shm-size 900G -v=<path_to_this_repo>:/mnt:rw --rm sh:test
```

Once the environment is set up, you will be able to generate text with:
```
python generate.py --config_path ./configs/7b-sh-32k-v1.yml \
--checkpoint_path <path_to_ckpt> --cached_generation \
--prompt_file ./test_prompt.txt
```

If you are generating with `prompt.txt`, set `prefill_style: fft` in the config. For very long prompts, you may want to opt for `prefill_style: recurrence`, which will be slower but use less memory.

#### HuggingFace


#### Testing Correctness

We report `lm-evaluation-harness` (10-shot) scores to use as a proxy for (standalone) model correctness in your environment.

* arc_challenge: 0.570 (acc norm)
* hellaswag: 0.816 (acc norm)
* winogrande: 0.735 (acc)

More extensive benchmarks results are provided in the blog post and on HuggingFace.

### Optional Dependencies

### Issues

Many issues can be resolved by reinstall the latest version of `flash_attn` (`pip freeze | grep flash-attn` should return a version `>= 2.0.0`).


