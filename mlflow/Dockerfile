FROM python:3.9.23-slim-trixie

WORKDIR /home

RUN apt-get update 
RUN apt install git -y

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
