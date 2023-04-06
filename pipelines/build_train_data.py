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
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost', 'feast']
)

put_parquet_op = kfp.components.create_component_from_func(
    func=steps_op.put_parquet_into_minio,
    base_image='python:3.9',
    packages_to_install=['minio']
)

@kfp.dsl.pipeline(
    name='train_data_pipeline',
    description='ML Pipeline for train_data'
)
def ml_pipeline():
    task_load_df = load_df_op(filename="titanic_train_entity.parquet")
    task_build_train_data = build_train_data_op(task_load_df.output)
    task_put_parquet = put_parquet_op(task_build_train_data.output, filename="titanic_train_final.parquet")

    task_build_train_data.after(task_load_df)
    task_put_parquet.after(task_build_train_data)

if __name__ == '__main__':
    # the namespace in which you deployed Kubeflow Pipelines
    KUBEFLOW_ENDPOINT = "http://10.64.140.43.nip.io:80"
    KUBEFLOW_USERNAME = "admin"
    KUBEFLOW_PASSWORD = "admin"

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

        },
        namespace = "admin",
    )