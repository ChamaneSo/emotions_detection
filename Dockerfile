#FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
#FROM ubuntu:20.04

FROM fizmath/gpu-opencv:latest

# ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get install -y tzdata

# RUN  apt-get update \
#  &&  apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 wget \
#  &&  rm -rf /var/lib/apt/lists/*

RUN  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update
RUN yes | apt-get install nvidia-driver-525

WORKDIR /app

COPY requirements.txt ./

# RUN apt-get update && \
#   apt-get install -y python3 python3-pip && \
#   apt-get clean && \
#   rm -rf /var/lib/apt/lists/*$

RUN pip install -r requirements.txt

COPY . .



#EXPOSE 8000