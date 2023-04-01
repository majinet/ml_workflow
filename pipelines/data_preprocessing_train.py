import kfp
import ml_pipeline_components.components as comp
import ml_pipeline_components.session as sess


def warmup():
    return None


load_parquet_op = kfp.components.create_component_from_func(
    func=comp.load_parquet_from_minio,
    base_image='python:3.9',
    packages_to_install=['minio']
)

extract_target_op = kfp.components.create_component_from_func(
    func=comp.extract_target,
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

data_clean_op = kfp.components.create_component_from_func(
    func=comp.data_clean,
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

feature_extract_op = kfp.components.create_component_from_func(
    func=comp.create_new_features,
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

put_parquet_op = kfp.components.create_component_from_func(
    func=comp.put_parquet_into_minio,
    base_image='python:3.9',
    packages_to_install=['minio']
)

warmup_op = kfp.components.create_component_from_func(
    func=warmup,
    base_image='python:3.9',
)


@kfp.dsl.pipeline(
    name='train_data_pipeline',
    description='ML Pipeline for train_data'
)
def ml_pipeline():

    task_warmup_op = warmup_op()
    task_load_parquet_op = load_parquet_op(filename="titanic_train.parquet")
    task_extract_entity_op = extract_entity_op(task_load_parquet_op.output)
    task_extract_target_op = extract_target_op(task_load_parquet_op.output)
    task_data_clean_op = data_clean_op(task_load_parquet_op.output)
    task_feature_extract_op = feature_extract_op(task_data_clean_op.output)

    task_put_parquet_ent = put_parquet_op(task_extract_entity_op.output, filename="titanic_train_entity.parquet")
    task_put_parquet_op = put_parquet_op(task_data_clean_op.output, filename="titanic_train_features.parquet")
    task_put_parquet_op_2 = put_parquet_op(task_feature_extract_op.output, filename="titanic_train_pca_features.parquet")
    task_put_parquet_op_3 = put_parquet_op(task_extract_target_op.output, filename="titanic_train_target.parquet")

    task_load_parquet_op.after(task_warmup_op)
    task_put_parquet_ent.after(task_load_parquet_op)
    task_extract_target_op.after(task_load_parquet_op)
    task_put_parquet_op_3.after(task_extract_target_op)

    task_data_clean_op.after(task_load_parquet_op)
    task_put_parquet_op.after(task_data_clean_op)

    task_feature_extract_op.after(task_data_clean_op)
    task_put_parquet_op_2.after(task_feature_extract_op)


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