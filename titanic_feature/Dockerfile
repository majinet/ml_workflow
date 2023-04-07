#FROM python:3.9-slim-buster
FROM alpine:latest

WORKDIR titanic_feature

RUN apk add --update python3=~3.9 py3-pip
RUN pip3 install minio feast psycopg2-binary isoduration--upgrade --user

COPY ./titanic_feature .

RUN ls .

CMD ["python3", "titanic_repo.py"]