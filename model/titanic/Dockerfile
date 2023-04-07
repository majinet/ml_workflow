FROM tensorflow/tensorflow:2.11.1
RUN pip3 install pandas scikit-learn keras #tensorflow_datasets
WORKDIR titanic
COPY titanic_model.py .
ENTRYPOINT ["python3", "titanic_model.py", "--batch_size", "64", "--learning_rate", "0,1", "--optimizer", "adam"]