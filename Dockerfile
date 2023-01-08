FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

WORKDIR /home

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN conda install -n base ipykernel --update-deps --force-reinstall -y

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt