FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
#FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y tzdata

RUN  apt-get update \
 &&  apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 wget \
 &&  rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN yes | apt-get install nvidia-driver-525

RUN apt-get update && \
  apt-get install -y python3 python3-pip && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*$

RUN apt-get update
RUN apt-get install xcb

ENV QT_QPA_PLATFORMTHEME=xcb

WORKDIR /app

COPY . .

RUN python3 -m pip install --upgrade pip

RUN pip install -r requirements.txt

#RUN cd src && python3 emotions.py --mode train

#CMD [ "uvicorn", "web:app" , "--host 0.0.0.0" , "--port 8000" , "--reload" ]