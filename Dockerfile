FROM nvcr.io/nvidia/pytorch:23.04-py3

WORKDIR /workdir
RUN rm -rf ./flash-attention/*
RUN git clone https://github.com/Dao-AILab/flash-attention.git
RUN cd flash-attention/csrc/layer_norm && python setup.py install

WORKDIR /workdir 
RUN cd flash-attention/csrc/rotary && python setup.py install

RUN pip install ninja deepspeed==0.10.3 tokenizers==0.14.1 einops transformers==4.34.1 

WORKDIR /workdir
RUN cd flash-attention/ && python setup.py install

