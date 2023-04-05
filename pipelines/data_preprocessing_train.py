import sys
import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, InputPath, OutputPath, Dataset, Model

from ml_pipeline_components import session


@dsl.component(
    base_image='python:3.9'
)
def startup_check():
    print("startup check")

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['minio']
)
def load_parquet_from_minio(output_path: OutputPath(str), filename: str):
    import os
    from minio import Minio
    from minio.error import S3Error

    client = Minio(
        "minio.kubeflow.svc.cluster.local",
        access_key="QM3BXB99A35ACSX4WI3G",
        secret_key="5Adjl44njceCYbz+6B7n34y8dwpG0nhY0SsKP+ZT",
        secure=False,
    )

    client.fget_object("demo-bucket", filename, output_path)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['minio']
)
def put_parquet_into_minio(file_path: InputPath(str), filename: str):
    import os
    from minio import Minio
    from minio.error import S3Error

    client = Minio(
        "minio.kubeflow.svc.cluster.local",
        access_key="QM3BXB99A35ACSX4WI3G",
        secret_key="5Adjl44njceCYbz+6B7n34y8dwpG0nhY0SsKP+ZT",
        secure=False,
    )

    os.system(f"ls -lrt {file_path}")

    client.fput_object("demo-bucket", filename, file_path)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)
def extract_entity(file_path: InputPath(str), output_path: OutputPath(str)):
    """
    Extracts the target column from a Parquet file and saves it to another Parquet file.

    Args:
        file_path (kfp.components.InputPath(str)): The path to the input Parquet file.
        output_path (kfp.components.OutputPath(str)): The path to the output Parquet file.

    Returns:
        None
    """
    import pandas as pd

    # Load the input Parquet file into a pandas dataframe
    df = pd.read_parquet(file_path)

    # Make a copy of the dataframe for preprocessing
    preprocessed_df = df.copy()

    # Remove any rows with missing values and reset the index
    preprocessed_df = preprocessed_df.dropna()
    preprocessed_df = preprocessed_df.reset_index(drop=True)

    # Select only the columns "PassengerId"
    columns = [
        "PassengerId",
    ]
    preprocessed_df = preprocessed_df.loc[:, columns]

    # Save the preprocessed dataframe to the output Parquet file
    preprocessed_df.to_parquet(output_path)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)
def extract_target(file_path: InputPath(str), output_path: OutputPath(str)):
    """
    Extracts the target column from a Parquet file and saves it to another Parquet file.

    Args:
        file_path (kfp.components.InputPath(str)): The path to the input Parquet file.
        output_path (kfp.components.OutputPath(str)): The path to the output Parquet file.

    Returns:
        None
    """
    import pandas as pd

    # Load the input Parquet file into a pandas dataframe
    df = pd.read_parquet(file_path)

    # Make a copy of the dataframe for preprocessing
    preprocessed_df = df.copy()

    # Remove any rows with missing values and reset the index
    preprocessed_df = preprocessed_df.dropna()
    preprocessed_df = preprocessed_df.reset_index(drop=True)

    # Select only the columns "PassengerId" and "Survived"
    columns = [
        "PassengerId",
        "Survived",
    ]
    preprocessed_df = preprocessed_df.loc[:, columns]

    # Save the preprocessed dataframe to the output Parquet file
    preprocessed_df.to_parquet(output_path)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)
def data_clean(file_path: InputPath(str), output_path: OutputPath(str)):
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    # create dictionary of ordinal to integer mapping
    col_sex_encode = {'male': 0,
                        'female': 1,
                    }
    # apply using map

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
    ]

    preprocessed_df = preprocessed_df.loc[:, columns]
    preprocessed_df['Sex'] = preprocessed_df.Sex.map(col_sex_encode)

    preprocessed_df.to_parquet(output_path)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)
def create_new_features(file_path: InputPath(str), output_path: OutputPath(str)):
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

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['numpy', 'pandas', 'fastparquet', 'scikit-learn', 'xgboost']
)
def build_train_data(file_path: InputPath(str), output_path: OutputPath(str)):
    import pandas as pd
    from feast import FeatureStore

    # Load the input Parquet file into a pandas dataframe
    entity_df = pd.read_parquet(file_path)

    fs = FeatureStore(repo_path="titanic_feature/feature_repo")

    feature_service = fs.get_feature_service("titanic_train_fv")

    training_df = fs.get_historical_features(
        features=feature_service,
        entity_df=entity_df
    ).to_df()

    training_df.to_parquet(output_path)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['minio', 'SQLAlchemy', 'pandas', 'psycopg2', 'pyarrow', 'fastparquet', 'feast']
)
def load_parquet_to_postgresql(file_path: InputPath(str), filename: str):
    import os
    import pandas as pd
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    from  feast.utils import make_df_tzaware

    # Define the PostgreSQL connection parameters
    hostname = 'postgresql.default.svc.cluster.local'
    port = '5432'
    database = 'feast'
    username = 'feast'
    password = 'feast'

    # Create a SQLAlchemy engine object
    engine = create_engine(f'postgresql://{username}:{password}@{hostname}:{port}/{database}')

    # Define the DataFrame
    df = pd.read_parquet(file_path)

    df['event_timestamp'] = datetime.now()

    df = make_df_tzaware(df)
    df['created'] = datetime.now()

    # Load the DataFrame into the PostgreSQL database
    table_name = filename
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    print(f"write to table: {df}")

    # Close the database connection
    engine.dispose()


@dsl.pipeline(
    name='train-data-pipeline',
    description='ML Pipeline for train_data'
)
def ml_pipeline():


    task_startup_check_op = startup_check()

    """
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
        namespace="admin",
        username=KUBEFLOW_USERNAME,
        password=KUBEFLOW_PASSWORD
    )

    print(auth_session["session_cookie"])

    client = kfp.Client(host=f"{KUBEFLOW_ENDPOINT}/_/pipeline",
                        cookies=auth_session["session_cookie"])

    print(f"health: {client.get_kfp_healthz()}")

    result = client.create_run_from_pipeline_func(
        ml_pipeline,
        arguments={

        }
    )

    print(f"result: {result}")

    #client.wait_for_run_completion(result.run_id, 900)