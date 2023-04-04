FROM majinetudacitylearn/module_ml_pipeline_components:v0.0.1

WORKDIR /app/data_preprocessing_train

COPY ./pipelines/data_preprocessing_train .

RUN ls /app/data_preprocessing_train

CMD [ "python3", "data_preprocessing_train.py"]