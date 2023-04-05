import sys
import kfp
from kfp import dsl
from kfp.components import InputPath, OutputPath, create_component_from_func

from ml_pipeline_components import steps_op
from ml_pipeline_components import session


# add_op is a task factory function that creates a task object when given arguments
startup_check_op = create_component_from_func(
    func=steps_op.startup_check,
    base_image='python:3.9', # Optional
)

# add_op is a task factory function that creates a task object when given arguments
load_parquet_from_minio_op = create_component_from_func(
    func=steps_op.load_parquet_from_minio,
    base_image='python:3.9', # Optional
    packages_to_install=['minio']
)

# add_op is a task factory function that creates a task object when given arguments
put_parquet_into_minio_op = create_component_from_func(
    func=steps_op.put_parquet_into_minio,
    base_image='python:3.9', # Optional
    packages_to_install=['minio']
)

# add_op is a task factory function that creates a task object when given arguments
extract_entity_op = create_component_from_func(
    func=steps_op.extract_entity,
    base_image='python:3.9', # Optional
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

# add_op is a task factory function that creates a task object when given arguments
extract_target_op = create_component_from_func(
    func=steps_op.extract_target,
    base_image='python:3.9', # Optional
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

# add_op is a task factory function that creates a task object when given arguments
data_clean_op = create_component_from_func(
    func=steps_op.data_clean,
    base_image='python:3.9', # Optional
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

# add_op is a task factory function that creates a task object when given arguments
create_new_features_op = create_component_from_func(
    func=steps_op.create_new_features,
    base_image='python:3.9', # Optional
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

# add_op is a task factory function that creates a task object when given arguments
build_train_data_op = create_component_from_func(
    func=steps_op.build_train_data,
    base_image='python:3.9', # Optional
    packages_to_install=['minio', 'SQLAlchemy', 'pandas', 'psycopg2', 'pyarrow', 'fastparquet', 'feast']
)

# add_op is a task factory function that creates a task object when given arguments
load_parquet_to_postgresql_op = create_component_from_func(
    func=steps_op.load_parquet_to_postgresql,
    base_image='python:3.9', # Optional
    packages_to_install=['minio', 'SQLAlchemy', 'pandas', 'psycopg2', 'pyarrow', 'fastparquet', 'feast']
)

@dsl.pipeline(
    name='train-data-pipeline',
    description='ML Pipeline for train_data'
)
def ml_pipeline():

    task_startup_check = startup_check_op()
    task_load_parquet = load_parquet_from_minio_op(filename="titanic_train.parquet")

    task_load_parquet.after(task_startup_check)

    """
    task_extract_entity_op = extract_entity_op(task_load_parquet_op.output)
    task_extract_target_op = extract_target_op(task_load_parquet_op.output)
    task_data_clean_op = data_clean_op(task_load_parquet_op.output)
    task_feature_extract_op = feature_extract_op(task_data_clean_op.output)

    #task_titanic_train_entity = put_parquet_sql_op(task_extract_entity_op.output, filename="titanic_train_entity")
    #task_titanic_train_target = put_parquet_sql_op(task_extract_target_op.output, filename="titanic_train_target")
    #task_titanic_train_features = put_parquet_sql_op(task_data_clean_op.output, filename="titanic_train_features")
    #task_titanic_train_pca_features = put_parquet_sql_op(task_feature_extract_op.output, filename="titanic_train_pca_features")

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

    print(auth_session["session_cookie"])

    client = kfp.Client(host=f"{KUBEFLOW_ENDPOINT}/pipeline",
                        namespace='admin',
                        cookies=auth_session["session_cookie"])

    result = client.create_run_from_pipeline_func(
        ml_pipeline,
        arguments={

        },
        namespace='admin'
    )