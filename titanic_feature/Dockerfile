FROM python:3.9-slim-buster

WORKDIR titanic_feature

RUN pip3 install feast --upgrade --user
RUN pip3 install psycopg2-binary --upgrade --user
RUN pip3 install isoduration --upgrade --user

COPY ./titanic_feature .

RUN ls .

CMD ["python3", "titanic_repo.py"]