import kfp
from typing import NamedTuple
import pandas as pd

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

def load_parquet_from_minio(output_path: kfp.components.OutputPath(str), filename: str):
    import os
    from minio import Minio
    from minio.error import S3Error

    client = Minio(
        "10.1.173.113:9000",
        access_key="91v98eLB1zOwDPo8",
        secret_key="6ZDwLVoC14AMOVCJozvJtIUjjwZfa0Ma",
        secure=False,
    )

    client.fget_object("demo-bucket", filename, output_path)

def put_parquet_into_minio(file_path: kfp.components.InputPath(str), filename: str):
    import os
    from minio import Minio
    from minio.error import S3Error

    client = Minio(
        "10.1.173.113:9000",
        access_key="91v98eLB1zOwDPo8",
        secret_key="6ZDwLVoC14AMOVCJozvJtIUjjwZfa0Ma",
        secure=False,
    )

    os.system(f"ls -lrt {file_path}")

    client.fput_object("demo-bucket", filename, file_path)

def data_clean(file_path: kfp.components.InputPath(str), output_path: kfp.components.OutputPath(str)):
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    df = pd.read_parquet(file_path)

    print(f"df: {df}")

    preprocessed_df = df.copy()

    preprocessed_df = preprocessed_df.dropna()
    preprocessed_df = preprocessed_df.reset_index(drop=True)

    columns = [
        "PassengerId",
        "Pclass",
        "Age",
        "Sex",
        "SibSp",
        "Parch",
        "Fare",
        "Survived",
    ]

    preprocessed_df = preprocessed_df.loc[:, columns]
    preprocessed_df.to_parquet(output_path)

def create_new_features(file_path: kfp.components.InputPath(str), output_path: kfp.components.OutputPath(str)):
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBRegressor

    def apply_pca(X, standardize=True):
        # Standardize
        if standardize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        # Create principal components
        pca = PCA()
        X_pca = pca.fit_transform(X)
        # Convert to dataframe
        component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
        X_pca = pd.DataFrame(X_pca, columns=component_names)
        # Create loadings
        loadings = pd.DataFrame(
            pca.components_.T,  # transpose the matrix of loadings
            columns=component_names,  # so the columns are the principal components
            index=X.columns,  # and the rows are the original features
        )
        return pca, X_pca, loadings

    df = pd.read_parquet(file_path)

    features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
    ]

    X = df.copy()
    X = X.loc[:, features]

    # `apply_pca`, defined above, reproduces the code from the tutorial
    pca, X_pca, loadings = apply_pca(X)

    X_pca['PassengerId'] = df['PassengerId']

    print(f"X_pca: {X_pca.head()}")

    X_pca.to_parquet(output_path)

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
    func=load_parquet_from_minio,
    base_image='python:3.9',
    packages_to_install=['minio']
)

data_clean_op = kfp.components.create_component_from_func(
    func=data_clean,
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

feature_extract_op = kfp.components.create_component_from_func(
    func=create_new_features,
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)

put_parquet_op = kfp.components.create_component_from_func(
    func=put_parquet_into_minio,
    base_image='python:3.9',
    packages_to_install=['minio']
)

warmup_op = kfp.components.create_component_from_func(
    func=warmup,
    base_image='python:3.9',
)


@kfp.dsl.pipeline(
    name='first_pipeline',
    description='ML Pipeline'
)
def ml_pipeline():

    task_warmup_op = warmup_op()
    task_load_parquet_op = load_parquet_op(filename="titanic_train.parquet")
    task_data_clean = data_clean_op(task_load_parquet_op.output)
    task_feature_extract = feature_extract_op(task_data_clean.output)
    task_put_parquet_op = put_parquet_op(task_data_clean.output, filename="titanic_train_preprocessed.parquet")
    task_put_parquet_op_2 = put_parquet_op(task_feature_extract.output, filename="feast_demo/feature_repo/data/titanic_pca_feature.parquet")

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
