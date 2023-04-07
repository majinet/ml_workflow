#FROM python:3.9-slim-buster
FROM python:3.9-alpine

WORKDIR titanic_feature

RUN pip3 install minio feast psycopg2-binary isoduration--upgrade --user

COPY ./titanic_feature .

RUN ls .

CMD ["python3", "titanic_repo.py"]