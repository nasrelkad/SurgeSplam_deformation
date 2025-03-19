FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing -y 
#RUN apt-get install -y --allow-change-held-packages  libcudnn8=8.1.1.33-1+cuda11.2 libcudnn8-dev=8.1.1.33-1+cuda11.2
RUN apt-get install -y ffmpeg libsm6 libxrender1 libxtst6 zip
# Library components for av
RUN apt-get install -y \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
RUN apt-get install -y python3 python3-pip git python3-dev pkg-config htop

ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;8.9+PTX"

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio
RUN pip3 install pyiqa numba pandas scikit_learn scipy scikit-image tqdm openpyxl torchsummary wandb
COPY requirements.txt .
RUN pip3 install -r requirements.txt 
WORKDIR /app/script