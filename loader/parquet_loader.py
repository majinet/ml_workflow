import kfp
from typing import NamedTuple
import pandas as pd


def load_parquet_from_minio_to_postgresql(filename: str):
    import os
    from minio import Minio
    from minio.error import S3Error
    import pandas as pd
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta

    client = Minio(
        "10.1.173.113:9000",
        access_key="91v98eLB1zOwDPo8",
        secret_key="6ZDwLVoC14AMOVCJozvJtIUjjwZfa0Ma",
        secure=False,
    )

    parquet_file = filename + '.parquet'

    client.fget_object("demo-bucket", parquet_file, parquet_file)

    # Define the PostgreSQL connection parameters
    hostname = '10.152.183.45'
    port = '5432'
    database = 'feast'
    username = 'feast'
    password = 'feast'

    # Create a SQLAlchemy engine object
    engine = create_engine(f'postgresql://{username}:{password}@{hostname}:{port}/{database}')

    # Define the DataFrame
    df = pd.read_parquet(parquet_file)

    df['event_timestamp'] = datetime.now()
    df['created'] = datetime.now()

    # Load the DataFrame into the PostgreSQL database
    table_name = filename
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    # Close the database connection
    engine.dispose()

put_parquet_op = kfp.components.create_component_from_func(
    func=load_parquet_from_minio_to_postgresql,
    base_image='python:3.9',
    packages_to_install=['minio', 'SQLAlchemy', 'pandas', 'psycopg2', 'pyarrow', 'fastparquet']
)


@kfp.dsl.pipeline(
    name='first_pipeline',
    description='ML Pipeline'
)
def ml_pipeline():
    put_parquet_op(filename="titanic_train")
    put_parquet_op(filename="titanic_train_preprocessed")
    put_parquet_op(filename="titanic_pca_feature")


if __name__ == '__main__':
    # the namespace in which you deployed Kubeflow Pipelines
    namespace = "kubeflow"

    client = kfp.Client(host=f"http://127.0.0.1:8080")

    client.create_run_from_pipeline_func(
        ml_pipeline,
        arguments={

        }
    )
