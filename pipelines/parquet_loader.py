import kfp
import ml_pipeline_components.components as comp
import ml_pipeline_components.session as sess


def load_parquet_from_minio_to_postgresql(filename: str):
    import os
    from minio import Minio
    from minio.error import S3Error
    import pandas as pd
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    from  feast.utils import make_df_tzaware
    from dateutil.tz import tzlocal
    from pytz import utc

    client = Minio(
        "10.1.211.147:9000",
        access_key="GZSTYEG6UFVFWMKTTF3W",
        secret_key="oWXpZtCB0CR1p6d+1VJ+YWHdZVRWfNS1M3haFtpp",
        secure=False,
    )

    parquet_file = filename + '.parquet'

    client.fget_object("demo-bucket", parquet_file, parquet_file)

    # Define the PostgreSQL connection parameters
    hostname = '10.152.183.19'
    port = '5432'
    database = 'feast'
    username = 'feast'
    password = 'feast'

    # Create a SQLAlchemy engine object
    engine = create_engine(f'postgresql://{username}:{password}@{hostname}:{port}/{database}')

    # Define the DataFrame
    df = pd.read_parquet(parquet_file)

    df['event_timestamp'] = datetime.now()

    df = make_df_tzaware(df)
    df['created'] = datetime.now()

    # Load the DataFrame into the PostgreSQL database
    table_name = filename
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    print(f"write to table: {df}")

    # Close the database connection
    engine.dispose()

put_parquet_op = kfp.components.create_component_from_func(
    func=load_parquet_from_minio_to_postgresql,
    base_image='python:3.9',
    packages_to_install=['minio', 'SQLAlchemy', 'pandas', 'psycopg2', 'pyarrow', 'fastparquet', 'feast']
)


@kfp.dsl.pipeline(
    name='first_pipeline',
    description='ML Pipeline'
)
def ml_pipeline():
    task_titanic_train_target = put_parquet_op(filename="titanic_train_target")
    task_titanic_train_features = put_parquet_op(filename="titanic_train_features")
    task_titanic_train_pca_features = put_parquet_op(filename="titanic_train_pca_features")
    put_parquet_op(filename="titanic_test_features")
    put_parquet_op(filename="titanic_test_pca_features")

    task_titanic_train_target.after(task_titanic_train_features)
    task_titanic_train_target.after(task_titanic_train_pca_features)


if __name__ == '__main__':
    KUBEFLOW_ENDPOINT = "http://10.64.140.43.nip.io:80"
    KUBEFLOW_USERNAME = "admin"
    KUBEFLOW_PASSWORD = "admin"

    auth_session = sess.get_istio_auth_session(
        url=KUBEFLOW_ENDPOINT,
        username=KUBEFLOW_USERNAME,
        password=KUBEFLOW_PASSWORD
    )

    client = kfp.Client(host=f"{KUBEFLOW_ENDPOINT}/pipeline",
                        namespace="admin",
                        cookies=auth_session["session_cookie"])

    result = client.create_run_from_pipeline_func(
        ml_pipeline,
        arguments={

        }
    )

    client.wait_for_run_completion(result.run_id, 900)
