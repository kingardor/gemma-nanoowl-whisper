FROM nvcr.io/nvidia/deepstream:6.4-gc-triton-devel as base

# Add open GL libraries
RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends \
        pkg-config \
        wget \
        python3-dev \
        python3-pip \
        python3-setuptools \
        tree \
        git \
        libportaudio2

# Cleanup
RUN rm -rf /var/lib/apt/lists/* && \
        apt autoremove

WORKDIR /app

COPY models /app/models
COPY src /app/src
COPY app.py /app/app.py

RUN pip3 install \
    torch \
    torchvision

RUN pip3 install openai \
    openai_whisper \
    opencv_python \
    pillow \
    scipy \
    matplotlib \
    sounddevice \
    transformers

RUN pip3 install tensorrt --extra-index-url https://pypi.nvidia.com
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && cd torch2trt && python3 setup.py install --user && cd ..
RUN git clone https://github.com/NVIDIA-AI-IOT/nanoowl && cd nanoowl && python3 setup.py install --user && cd ..

RUN update-ca-certificates

# To get video driver libraries at runtime (libnvidia-encode.so/libnvcuvid.so)
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video,compute,graphics,utility

ENV CUDA_HOME=/usr/local/cuda
ENV CFLAGS="-I$CUDA_HOME/include $CFLAGS"
ENV RUNNING_IN_DOCKER true