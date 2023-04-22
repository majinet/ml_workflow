import argparse
import kfp
from ml_pipeline_components import steps_op
from ml_pipeline_components import session


load_df_op = kfp.components.create_component_from_func(
    func=steps_op.load_df_from_postgresql,
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'feast', 'SQLAlchemy', 'psycopg2']
)

build_train_data_op = kfp.components.create_component_from_func(
    func=steps_op.build_train_data,
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'psycopg2', 'fastparquet', 'scikit-learn', 'xgboost', 'feast', 'minio', 'redis']
)

put_parquet_op = kfp.components.create_component_from_func(
    func=steps_op.put_parquet_into_minio,
    base_image='python:3.9',
    packages_to_install=['minio']
)

@kfp.dsl.pipeline(
    name='pipeline-build-train-data',
    description='ML Pipeline for train_data'
)
def ml_pipeline(minio_access_key, minio_secret_key):
    task_load_df = load_df_op(filename="titanic_train_entity.parquet")
    task_build_train_data = build_train_data_op(task_load_df.output)
    task_put_parquet = put_parquet_op(task_build_train_data.output, filename="titanic_train_final.parquet", minio_access_key=minio_access_key, minio_secret_key=minio_secret_key)

    task_build_train_data.after(task_load_df)
    task_put_parquet.after(task_build_train_data)

if __name__ == '__main__':
    # the namespace in which you deployed Kubeflow Pipelines
    KUBEFLOW_ENDPOINT = "http://10.64.140.43.nip.io:80"
    KUBEFLOW_USERNAME = "admin"
    KUBEFLOW_PASSWORD = "admin"

    parser = argparse.ArgumentParser()
    parser.add_argument("--minio-access-key",
                        type=str,
                        help="Minio Access Key")
    parser.add_argument("--minio-secret-key",
                        type=str,
                        help='Minio Secret Key')
    parsed_args, _ = parser.parse_known_args()

    auth_session = session.get_istio_auth_session(
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
            'minio_access_key': parsed_args.minio_access_key,
            'minio_secret_key': parsed_args.minio_secret_key
        },
        namespace = "admin",
    )