import kfp
import ml_pipeline_components.components as comp
import ml_pipeline_components.session as sess


"""
def load_csv_from_minio(output_csv_path: kfp.components.OutputPath('CSV')):
    import os
    from minio import Minio
    from minio.error import S3Error

    client = Minio(
        "10.1.173.113:9000",
        access_key="91v98eLB1zOwDPo8",
        secret_key="6ZDwLVoC14AMOVCJozvJtIUjjwZfa0Ma",
        secure=False,
    )

    os.system(f"ls -lrt {output_csv_path}")

    client.fget_object("demo-bucket", "autos.csv", output_csv_path)

    os.system(f"ls -lrt {output_csv_path}")
"""

def warmup():
    return None


""" def make_mi_scores(X, y, discrete_features):
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import mutual_info_regression

    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    print(f"mi_scores: {mi_scores[::3]}")

    return mi_scores


def factorize(file_path: kfp.components.InputPath('CSV')) -> NamedTuple('output', [('X', None), ('y', None), ('discrete_features', None)]):
    import numpy as np
    import pandas as pd
    from collections import namedtuple

    df = pd.read_csv(file_path)
    df.head()

    print(f"df: {df.head()}")

    X = df.copy()
    y = X.pop("price")

    # Label encoding for categoricals
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()

    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X.dtypes == int

    outputs = namedtuple('output', ['X', 'y', 'discrete_features'])

    return outputs(X, y, discrete_features)

csv_op = kfp.components.create_component_from_func(
    func=load_csv_from_minio,
    base_image='python:3.9',
    packages_to_install=['minio']
)


factorize_op = kfp.components.create_component_from_func(
    func=factorize,
    base_image='python:3.9',
    packages_to_install=['scikit-learn', 'pandas', 'numpy']
)


mi_scores_op = kfp.components.create_component_from_func(
    func=make_mi_scores,
    base_image='python:3.9',
    packages_to_install=['scikit-learn', 'pandas', 'numpy']
)
"""

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
    task_extract_target_op = extract_target_op(task_load_parquet_op.output)
    task_data_clean_op = data_clean_op(task_load_parquet_op.output)
    task_feature_extract_op = feature_extract_op(task_data_clean_op.output)
    task_put_parquet_op = put_parquet_op(task_data_clean_op.output, filename="titanic_train_features.parquet")
    task_put_parquet_op_2 = put_parquet_op(task_feature_extract_op.output, filename="titanic_train_pca_features.parquet")
    task_put_parquet_op_3 = put_parquet_op(task_extract_target_op.output, filename="titanic_train_target.parquet")

    task_load_parquet_op.after(task_warmup_op)
    task_extract_target_op.after(task_load_parquet_op)
    task_put_parquet_op_3.after(task_extract_target_op)

    task_data_clean_op.after(task_load_parquet_op)
    task_put_parquet_op.after(task_data_clean_op)

    task_feature_extract_op.after(task_data_clean_op)
    task_put_parquet_op_2.after(task_feature_extract_op)

    """
    task_score_op = factorize_op(task_csv_op.output)
    task_mi_scores_op = mi_scores_op(task_score_op.outputs['X'], task_score_op.outputs['y'], task_score_op.outputs['discrete_features'])

    task_score_op.after(task_csv_op)
    task_mi_scores_op.after(task_score_op)
    """

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
"""

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

    client.create_run_from_pipeline_func(
        ml_pipeline,
        arguments={

        }
    )