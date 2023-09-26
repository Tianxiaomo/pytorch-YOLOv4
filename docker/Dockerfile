ARG CUDA_VER
ARG UBUNTU_VER

FROM nvidia/cuda:${CUDA_VER}-cudnn8-devel-ubuntu${UBUNTU_VER}

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update \
 && apt-get install -y  \
      wget \
      git \
      python3 \
      python3-pip \
      sudo \
 && apt-get clean \
 && pip install --upgrade pip \
 && pip install wheel setuptools
 
# matplotlib dependencies   
RUN apt-get update -y \
 && apt-get install -y \
    libfontconfig1-dev \
    libagg-dev \
    libpng-dev \
    libqhull-dev \
    ffmpeg \
    libsm6 \
    libxext6

# project dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# user
RUN useradd -m -s /bin/bash yolo \
  && yes password | passwd yolo \
  && usermod -aG sudo yolo

WORKDIR /repo
RUN chown -R yolo /repo

USER yolo
    
ENTRYPOINT ["chown yolo /data && chown yolo /repo && bash"]

