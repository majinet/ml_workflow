import kfp
from typing import NamedTuple
import pandas as pd

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



def make_mi_scores(X, y, discrete_features):
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


@kfp.dsl.pipeline(
    name='first_pipeline',
    description='ML Pipeline'
)
def ml_pipeline():

    task_csv_op = csv_op()
    task_score_op = factorize_op(task_csv_op.output)
    task_mi_scores_op = mi_scores_op(task_score_op.outputs['X'], task_score_op.outputs['y'], task_score_op.outputs['discrete_features'])

    task_score_op.after(task_csv_op)
    task_mi_scores_op.after(task_score_op)


if __name__ == '__main__':
    # the namespace in which you deployed Kubeflow Pipelines
    namespace = "kubeflow"

    client = kfp.Client(host=f"http://127.0.0.1:8080")

    client.create_run_from_pipeline_func(
        ml_pipeline,
        arguments={

        }
    )
