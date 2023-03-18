import kfp
from typing import NamedTuple
import pandas as pd
import ml_pipeline_components.components as comp

def warmup():
    return None


load_parquet_op = kfp.components.create_component_from_func(
    func=comp.load_parquet_from_minio,
    base_image='python:3.9',
    packages_to_install=['minio']
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
    name='train_test_pipeline',
    description='ML Pipeline for test data'
)
def ml_pipeline():

    task_warmup_op = warmup_op()
    task_load_parquet_op = load_parquet_op(filename="titanic_test.parquet")
    task_data_clean = data_clean_op(task_load_parquet_op.output)
    task_feature_extract = feature_extract_op(task_data_clean.output)
    task_put_parquet_op = put_parquet_op(task_data_clean.output, filename="titanic_test_features.parquet")
    task_put_parquet_op_2 = put_parquet_op(task_feature_extract.output, filename="titanic_test_pca_features.parquet")

    task_load_parquet_op.after(task_warmup_op)

    task_data_clean.after(task_load_parquet_op)
    task_put_parquet_op.after(task_data_clean)

    task_feature_extract.after(task_data_clean)
    task_put_parquet_op_2.after(task_feature_extract)

    """
    task_score_op = factorize_op(task_csv_op.output)
    task_mi_scores_op = mi_scores_op(task_score_op.outputs['X'], task_score_op.outputs['y'], task_score_op.outputs['discrete_features'])

    task_score_op.after(task_csv_op)
    task_mi_scores_op.after(task_score_op)
    """


if __name__ == '__main__':
    # the namespace in which you deployed Kubeflow Pipelines
    namespace = "kubeflow"

    client = kfp.Client(host=f"http://127.0.0.1:8080")

    client.create_run_from_pipeline_func(
        ml_pipeline,
        arguments={

        }
    )
