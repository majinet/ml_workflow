import sys
import kfp
from ml_pipeline_components import components
from ml_pipeline_components import session


def warmup():
    return None

load_parquet_op = kfp.components.create_component_from_func(
    func=components.load_parquet_from_minio,
    base_image='python:3.9',
    packages_to_install=['minio']
)

extract_target_op = kfp.components.create_component_from_func(
    func=components.extract_target,
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

extract_entity_op = kfp.components.create_component_from_func(
    func=components.extract_entity,
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

data_clean_op = kfp.components.create_component_from_func(
    func=components.data_clean,
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

feature_extract_op = kfp.components.create_component_from_func(
    func=components.create_new_features,
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

put_parquet_op = kfp.components.create_component_from_func(
    func=components.put_parquet_into_minio,
    base_image='python:3.9',
    packages_to_install=['minio']
)

warmup_op = kfp.components.create_component_from_func(
    func=warmup,
    base_image='python:3.9',
)

"""
put_parquet_sql_op = kfp.components.create_component_from_func(
    func=components.load_parquet_to_postgresql,
    base_image='python:3.9',
    packages_to_install=['minio', 'SQLAlchemy', 'pandas', 'psycopg2', 'pyarrow', 'fastparquet', 'feast']
)
"""

@kfp.dsl.pipeline(
    name='train_data_pipeline',
    description='ML Pipeline for train_data'
)
def ml_pipeline():

    """
    task_warmup_op = warmup_op()
    task_load_parquet_op = load_parquet_op(filename="titanic_train.parquet")
    task_extract_entity_op = extract_entity_op(task_load_parquet_op.output)
    task_extract_target_op = extract_target_op(task_load_parquet_op.output)
    task_data_clean_op = data_clean_op(task_load_parquet_op.output)
    task_feature_extract_op = feature_extract_op(task_data_clean_op.output)

    #task_titanic_train_entity = put_parquet_sql_op(task_extract_entity_op.output, filename="titanic_train_entity")
    #task_titanic_train_target = put_parquet_sql_op(task_extract_target_op.output, filename="titanic_train_target")
    #task_titanic_train_features = put_parquet_sql_op(task_data_clean_op.output, filename="titanic_train_features")
    #task_titanic_train_pca_features = put_parquet_sql_op(task_feature_extract_op.output, filename="titanic_train_pca_features")

    task_load_parquet_op.after(task_warmup_op)
    task_extract_entity_op.after(task_load_parquet_op)
    task_extract_target_op.after(task_load_parquet_op)

    #task_titanic_train_entity.after(task_extract_entity_op)
    #task_titanic_train_target.after(task_extract_target_op)

    task_data_clean_op.after(task_load_parquet_op)
    #task_titanic_train_features.after(task_data_clean_op)

    task_feature_extract_op.after(task_data_clean_op)
    #task_titanic_train_pca_features.after(task_feature_extract_op)
    """

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

    """
    result = client.create_run_from_pipeline_func(
        ml_pipeline,
        arguments={

        }
    )

    #client.wait_for_run_completion(result.run_id, 900)
    """

    sys.exit(0)