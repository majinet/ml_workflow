import argparse
import io
import logging
import json
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.random import seed
import tensorflow as tf
import tensorflow_io as tfio
tf.random.set_seed(221)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
logging.getLogger().setLevel(logging.INFO)

from minio import Minio

minio_access_key = os.environ['MINIO_ACCESS_KEY_ID']
minio_secret_key = os.environ['MINIO_SECRET_ACCESS_KEY']
s3_endpoint = os.environ['S3_ENDPOINT']

def load_dataset():

    client = Minio(
        "minio.kubeflow.svc.cluster.local:9000",
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )

    obj = client.get_object("demo-bucket", "titanic_train_final.parquet")
    data = io.BytesIO()
    data.write(obj.read())
    data.seek(0)

    df = pd.read_parquet(data)

    x = df.drop('Survived', axis=1)
    x = x.drop('event_timestamp', axis=1)
    x = x.drop('created', axis=1)
    x = x.drop('PassengerId', axis=1)
    x = x.astype(np.float32)
    y = df['Survived']
    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=111)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train = train_dataset.cache().shuffle(2000).repeat()

    return train, test_dataset

def export_model(model_dir: str,export_bucket: str,model_name: str,model_version: int,):
    import boto3
    from botocore.client import Config
    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://{s3_endpoint}",
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        config=Config(signature_version="s3v4"),
        use_ssl=False,
        verify=False,
    )

    # Create export bucket if it does not yet exist
    response = s3.list_buckets()
    export_bucket_exists = False

    for bucket in response["Buckets"]:
        if bucket["Name"] == export_bucket:
            export_bucket_exists = True

        if not export_bucket_exists:
            s3.create_bucket(ACL="public-read-write", Bucket=export_bucket)

    # Save model files to S3
    for root, dirs, files in os.walk(model_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            s3_path = os.path.relpath(local_path, model_dir)

            s3.upload_file(
                local_path,
                export_bucket,
                f"models/{model_name}/{model_version}/{s3_path}",
                ExtraArgs={"ACL": "public-read"},
            )

    response = s3.list_objects(Bucket=export_bucket)
    print(f"All objects in {export_bucket}:")

    for file in response["Contents"]:
        print("{}/{}".format(export_bucket, file["Key"]))

def model(args):
    seed(1)
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=11))
    #model.add(BatchNormalization())
    model.add(Dense(10, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    opt = args.optimizer
    model.compile(optimizer=opt, loss = 'binary_crossentropy', metrics=['accuracy'])
    tf.keras.backend.set_value(model.optimizer.learning_rate, args.learning_rate)
    return model
def main(args):

    #MultiWorkerMirroredStrategy creates copies of all variables in the model's
    #layers on each device across all workers
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=tf.distribute.experimental.CollectiveCommunication.AUTO)
    logging.debug(f"num_replicas_in_sync: {strategy.num_replicas_in_sync}")

    BATCH_SIZE_PER_REPLICA = args.batch_size
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    # Datasets need to be created after instantiation of `MultiWorkerMirroredStrategy`
    train_dataset, test_dataset = load_dataset()
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)

    # See: https://www.tensorflow.org/api_docs/python/tf/data/experimental/DistributeOptions
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
    tf.data.experimental.AutoShardPolicy.DATA
    train_datasets_sharded  = train_dataset.with_options(options)
    test_dataset_sharded = test_dataset.with_options(options)

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = model(args)
        # Keras' `model.fit()` trains the model with specified number of epochs and
        # number of steps per epoch.
        multi_worker_model.fit(train_datasets_sharded, epochs=50, steps_per_epoch=30)
        eval_loss, eval_acc = multi_worker_model.evaluate(test_dataset_sharded, verbose=0, steps=10)
        # Log metrics for Katib
        logging.info("loss={:.4f}".format(eval_loss))
        logging.info("accuracy={:.4f}".format(eval_acc))

        #multi_worker_model.save(tf.io.gfile.join("s3://demo-bucket", "model"))
        multi_worker_model.save("titanic_model")
        export_model("titanic_model", "demo-bucket", "titanic", 1)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch-size",
                     type=int,
                     default=32,
                     metavar="N",
                     help="Batch size for training (default: 128)")
  parser.add_argument("--learning-rate",
                     type=float,
                     default=0.1,
                     metavar="N",
                     help='Initial learning rate')
  parser.add_argument("--optimizer",
                     type=str,
                     default='adam',
                     metavar="N",
                     help='optimizer')
  parsed_args, _ = parser.parse_known_args()
  main(parsed_args)