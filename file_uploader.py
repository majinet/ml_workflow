import pandas as pd
from minio import Minio
from minio.error import S3Error
from datetime import datetime, timedelta

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def to_parquet(path: str, df: pd.DataFrame):
    df['event_timestamp'] = datetime.now() - timedelta(days=2)
    df.to_parquet(path)

def main():
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio(
        "127.0.0.1:9000",
        access_key="91v98eLB1zOwDPo8",
        secret_key="6ZDwLVoC14AMOVCJozvJtIUjjwZfa0Ma",
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

    df_titanic_train = read_csv("input/fe-course-data/titanic_train.csv")
    to_parquet("feast_demo/feature_repo/data/titanic_train.parquet", df_titanic_train)

    client.fput_object(
        "demo-bucket", "titanic_train.parquet", "feast_demo/feature_repo/data/titanic_train.parquet",
    )

    df_titanic_test = read_csv("input/fe-course-data/titanic_test.csv")
    to_parquet("feast_demo/feature_repo/data/titanic_test.parquet", df_titanic_test)

    client.fput_object(
        "demo-bucket", "titanic_test.parquet", "feast_demo/feature_repo/data/titanic_test.parquet",
    )
    print(
        "'input/fe-course-data/concrete.csv' is successfully uploaded as "
        "object 'concrete.csv' to bucket 'demo-bucket'."
    )


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)