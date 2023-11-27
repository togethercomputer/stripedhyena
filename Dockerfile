FROM nvcr.io/nvidia/pytorch:23.06-py3

WORKDIR /workdir

RUN rm -rf ./flash-attention/* && \
    pip uninstall flash_attn -y && \
    git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/csrc/rotary && python setup.py install && \
    cd ../layer_norm && python setup.py install && \ 
    cd ../../ && python setup.py install 

RUN pip install ninja tokenizers==0.14.1 einops transformers==4.34.1 

python generate.py --config_path ./configs/7b-sh-32k-v0.1.yml --checkpoint_path /mnt/checkpoints/sh7b_30b_32k_v2/global_step6000/complete_ckpt.pt --cached_generation --prompt_file ./test_prompt.txt