FROM ubuntu:18.04

RUN apt update  \
    && apt install -y htop python3-dev python3-pip

COPY . src/
RUN /bin/bash -c "cd src \
       && pip3 install -r requirements.txt"
