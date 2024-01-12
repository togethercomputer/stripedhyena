FROM nvcr.io/nvidia/pytorch:23.06-py3

WORKDIR /workdir

RUN rm -rf ./flash-attention/* && \
    pip uninstall flash_attn -y && \
    git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/csrc/rotary && python setup.py install && \
    cd ../layer_norm && python setup.py install && \ 
    cd ../../ && python setup.py install 

RUN pip install ninja tokenizers==0.14.1 einops triton transformers==4.34.1 

