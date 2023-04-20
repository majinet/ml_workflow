import pandas as pd
from minio import Minio
from minio.error import S3Error
from datetime import datetime, timedelta
import argparse
from pathlib import Path

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def to_parquet(path: str, df: pd.DataFrame):
    df['event_timestamp'] = datetime.now() - timedelta(days=2)
    df.to_parquet(path)

def main(train_path: str, train_filename: str, test_path: str, test_filename: str, minio_access_key: str, minio_secret_key: str):
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio(
        "minio.kubeflow.svc.cluster.local:9000",
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )

    # Make 'demo-bucket' bucket if not exist.
    found = client.bucket_exists("demo-bucket")
    if not found:
        client.make_bucket("demo-bucket")
    else:
        print("Bucket 'demo-bucket' already exists")


    # Upload '/home/user/Photos/asiaphotos.zip' as object name
    # 'asiaphotos-2015.zip' to bucket 'asiatrip'.

    df_titanic_train = read_csv(f"{train_path}/{train_filename}.csv")
    to_parquet(f"{train_path}/{train_filename}.parquet", df_titanic_train)

    client.fput_object(
        "demo-bucket", f"{train_filename}.parquet", f"{train_path}/{train_filename}.parquet",
    )

    df_titanic_test = read_csv(f"{test_path}/{test_filename}.csv")
    to_parquet(f"{test_path}/{test_filename}.parquet", df_titanic_test)

    client.fput_object(
        "demo-bucket", f"{test_filename}.parquet", f"{test_path}/{test_filename}.parquet",
    )

    print(
        "'input/fe-course-data/concrete.csv' is successfully uploaded as "
        "object 'concrete.csv' to bucket 'demo-bucket'."
    )


if __name__ == "__main__":
    try:
        # Defining and parsing the command-line arguments
        parser = argparse.ArgumentParser(description='My program description')

        # Paths must be passed in, not hardcoded
        parser.add_argument('--train-path', type=str, help='Path of the local file containing the Input 1 data.')
        parser.add_argument('--train-filename', type=str, help='Path of the local file containing the Input 1 data.')
        parser.add_argument('--test-path', type=str, help='Path of the local file containing the Input 1 data.')
        parser.add_argument('--test-filename', type=str, help='Path of the local file containing the Input 1 data.')
        parser.add_argument('--minio-access-key', type=str, help='Minio Access Key.')
        parser.add_argument('--minio-secret-key', type=str, help='Minio Secret Key.')
        args = parser.parse_args()

        main(args.train_path, args.train_filename, args.test_path, args.test_filename, args.minio_access_key, args.minio_secret_key)

    except S3Error as exc:
        print("error occurred.", exc)