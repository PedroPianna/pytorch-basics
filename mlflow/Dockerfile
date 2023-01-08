FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
WORKDIR /home

RUN apt-get update 

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN apt install git -y