import sys
import kfp
from kfp import dsl
from kfp.components import InputPath, OutputPath, create_component_from_func

def startup_check():
    print("startup check")

def load_parquet_from_minio(output_path: OutputPath(str), filename: str):
    import os
    from minio import Minio
    from minio.error import S3Error

    client = Minio(
        "minio.kubeflow.svc.cluster.local:9000",
        access_key="QM3BXB99A35ACSX4WI3G",
        secret_key="5Adjl44njceCYbz+6B7n34y8dwpG0nhY0SsKP+ZT",
        secure=False,
    )

    client.fget_object("demo-bucket", filename, output_path)

def put_parquet_into_minio(file_path: InputPath(str), filename: str):
    import os
    from minio import Minio
    from minio.error import S3Error

    client = Minio(
        "minio.kubeflow.svc.cluster.local:9000",
        access_key="QM3BXB99A35ACSX4WI3G",
        secret_key="5Adjl44njceCYbz+6B7n34y8dwpG0nhY0SsKP+ZT",
        secure=False,
    )

    os.system(f"ls -lrt {file_path}")

    client.fput_object("demo-bucket", filename, file_path)

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

def build_train_data(file_path: InputPath(str), output_path: OutputPath(str)):
    import os
    import pandas as pd
    from feast import FeatureStore
    from minio import Minio
    from minio.error import S3Error

    os.system("mkdir feature_repo")

    client = Minio(
        "minio.kubeflow.svc.cluster.local:9000",
        access_key="QM3BXB99A35ACSX4WI3G",
        secret_key="5Adjl44njceCYbz+6B7n34y8dwpG0nhY0SsKP+ZT",
        secure=False,
    )

    client.fget_object("demo-bucket", "feature_store.yaml", "feature_repo/feature_store.yaml")

    # Load the input Parquet file into a pandas dataframe
    entity_df = pd.read_parquet(file_path)

    fs = FeatureStore(repo_path="../feature_repo")

    feature_service = fs.get_feature_service("titanic_survive_svc_v1")

    training_df = fs.get_historical_features(
        features=feature_service,
        entity_df=entity_df
    ).to_df()

    print("----- Feature schema -----\n")
    print(training_df.info())

    print()
    print("----- Example features -----\n")
    print(training_df.head())

    training_df.to_parquet(output_path)

def load_parquet_to_postgresql(file_path: InputPath(str), filename: str):
    import os
    import pandas as pd
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    from feast.utils import make_df_tzaware

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

def load_df_from_postgresql(filename: str, output_path: OutputPath(str)):
    import os
    import pandas as pd
    from sqlalchemy import create_engine
    from datetime import datetime, timedelta
    from feast.utils import make_df_tzaware

    # Define the PostgreSQL connection parameters
    hostname = 'postgresql.default.svc.cluster.local'
    port = '5432'
    database = 'feast'
    username = 'feast'
    password = 'feast'

    # Create a SQLAlchemy engine object
    engine = create_engine(f'postgresql://{username}:{password}@{hostname}:{port}/{database}')

    # Read the DataFrame from the PostgreSQL database
    df = pd.read_sql_table('titanic_train_entity', engine)

    df['event_timestamp'] = datetime.now()

    df = make_df_tzaware(df)
    df['created'] = datetime.now()

    print(f"write to df: {df}")

    df.to_parquet(output_path)

    # Close the database connection
    engine.dispose()
