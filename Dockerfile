FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

RUN apt update
RUN apt-get update

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get install -y libenchant1c2a
RUN cp /etc/localtime/Japan /etc/Japan && rm -r /etc/localtime && mv /etc/Japan /etc/localtime

RUN apt-get install -y python3.8 python3-pip git libsndfile1 ffmpeg gstreamer-1.0 gstreamer1.0-libav libgirepository1.0-dev libcairo2-dev tesseract-ocr imagemagick libicu-dev libicu-dev espeak


RUN python3.8 -m pip install -r ./requirements.txt
RUN python3.8 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /object_detection

