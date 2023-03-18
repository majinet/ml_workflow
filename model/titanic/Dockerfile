FROM tensorflow/tensorflow:2.4.0
RUN pip install tensorflow_datasets pandas scikit-learn keras
WORKDIR /app
COPY titanic_model.py /app
ENTRYPOINT ["python", "/app/titanic_model.py", "--batch_size", "64", "--learning_rate", "0,1", "--optimizer", "adam"]