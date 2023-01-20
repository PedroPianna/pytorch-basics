FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

WORKDIR /home

RUN apt-get update 
RUN apt install git -y

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt