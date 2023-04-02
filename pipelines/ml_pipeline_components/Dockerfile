FROM python:3.9-slim-buster

WORKDIR /app/ml_pipeline_components

RUN sudo apt install python3-pip -y
RUN pip3 install kfp --upgrade --user
RUN echo 'export PATH=$PATH:~/.local/bin' | tee -a ~/.bashrc

COPY . .
