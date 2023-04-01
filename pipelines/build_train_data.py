import kfp
import ml_pipeline_components.components as comp
import ml_pipeline_components.session as sess


load_parquet_op = kfp.components.create_component_from_func(
    func=comp.load_parquet_from_minio,
    base_image='python:3.9',
    packages_to_install=['minio']
)

build_train_data_op = kfp.components.create_component_from_func(
    func=comp.build_train_data,
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost', 'feast']
)

put_parquet_op = kfp.components.create_component_from_func(
    func=comp.put_parquet_into_minio,
    base_image='python:3.9',
    packages_to_install=['minio']
)

@kfp.dsl.pipeline(
    name='train_data_pipeline',
    description='ML Pipeline for train_data'
)
def ml_pipeline():
    task_load_parquet_op = load_parquet_op(filename="titanic_train_entity.parquet")
    task_build_train_data_op = build_train_data_op(task_load_parquet_op.output)
    task_put_parquet_op = put_parquet_op(task_build_train_data_op.output, filename="titanic_train_final.parquet")

    task_build_train_data_op.after(task_load_parquet_op)
    task_put_parquet_op.after(task_build_train_data_op)

if __name__ == '__main__':
    # the namespace in which you deployed Kubeflow Pipelines
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