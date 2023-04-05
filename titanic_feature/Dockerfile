FROM python:3.9-slim-buster

WORKDIR titanic_feature

RUN pip3 install feast --upgrade --user

COPY ./feature_repo .

CMD ["feast", "apply"]