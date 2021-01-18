FROM tensorflow/tensorflow:latest

RUN apt update && \
    apt install git -y

COPY ./requirement.txt /requirement.txt

RUN pip install -r /requirement.txt && \
    rm /requirement.txt