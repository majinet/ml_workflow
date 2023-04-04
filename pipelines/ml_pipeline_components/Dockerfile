FROM python:3.9-slim-buster

WORKDIR /app/ml_pipeline_components

#RUN apt install python3-pip -y
RUN pip3 install kfp --upgrade --user
RUN echo 'export PATH=$PATH:~/.local/bin' | tee -a ~/.bashrc

COPY ./pipelines/ml_pipeline_components .
COPY ./pipelines/main.py ../
