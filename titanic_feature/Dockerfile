FROM python:3.9-slim-buster

WORKDIR titanic_feature

RUN pip3 install feast --upgrade --user

COPY ./titanic_feature .

RUN ls .

CMD ["python3", "feast_main.py"]