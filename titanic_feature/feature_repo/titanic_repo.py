# This is an example feature definition file
import argparse
from datetime import timedelta

import pandas as pd

from feast import (
    FeatureStore,
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
    BatchFeatureView,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, String

from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from isoduration.types import Duration

from minio import Minio
from minio.error import S3Error


# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
passenger = Entity(name="passenger", join_keys=["PassengerId"])

titanic_train_pca_source = PostgreSQLSource(
    name="titanic_train_pca_features",
    table="titanic_train_pca_features",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

titanic_train_target_source = PostgreSQLSource(
    name="titanic_train_target",
    table="titanic_train_target",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

titanic_train_source = PostgreSQLSource(
    name="titanic_train_features",
    table="titanic_train_features",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

"""titanic_pca_source = FileSource(
    name="titanic_pca_feature",
    path="/mnt/c/Users/majin/PycharmProjects/ml_workflow/feast_demo/feature_repo/data/titanic_pca_feature.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)"""

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
titanic_train_pca_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="titanic_train_pca_fv",
    entities=[passenger],
    ttl=timedelta(days=1),
    schema=[
        Field(name="PC1", dtype=Float32),
        Field(name="PC2", dtype=Float32),
        Field(name="PC3", dtype=Float32),
        Field(name="PC4", dtype=Float32),
        Field(name="PC5", dtype=Float32),
    ],
    online=True,
    source=titanic_train_pca_source,
)

titanic_train_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="titanic_train_fv",
    entities=[passenger],
    ttl=timedelta(days=1),
    schema=[
        Field(name="Pclass", dtype=Int64),
        Field(name="Age", dtype=Float32),
        Field(name="Sex", dtype=String),
        Field(name="SibSp", dtype=Int64),
        Field(name="Parch", dtype=Int64),
        Field(name="Fare", dtype=Float32),
    ],
    online=True,
    source=titanic_train_source,
)

titanic_target_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="titanic_target_fv",
    entities=[passenger],
    ttl=timedelta(days=1),
    schema=[
        Field(name="Survived", dtype=Int64),
    ],
    online=True,
    source=titanic_train_target_source,
)

"""
# This groups features into a model version
titanic_survive_svc_v1 = FeatureService(
    name="titanic_survive_svc_v1",
    features=[
        titanic_train_pca_fv,  # Sub-selects a feature from a feature view
        titanic_train_fv,
        titanic_target_fv,
    ],
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--minio-access-key",
                        type=str,
                        help="Minio Access Key")
    parser.add_argument("--minio-secret-key",
                        type=str,
                        help='Minio Secret Key')
    parsed_args, _ = parser.parse_known_args()

    fs = FeatureStore(repo_path="feature_repo")
    fs.apply(
        [titanic_train_pca_source, titanic_train_target_source, titanic_train_source, passenger, titanic_train_pca_fv,
         titanic_train_fv, titanic_target_fv, titanic_survive_svc_v1],
        partial=False,
    )

    client = Minio(
        "minio.kubeflow.svc.cluster.local:9000",
        access_key=parsed_args.minio_access_key,
        secret_key=parsed_args.minio_secret_key,
        secure=False,
    )

    client.fput_object("demo-bucket", "feature_store.yaml", "feature_repo/feature_store.yaml")

# Defines a way to push data (to be available offline, online or both) into Feast.
titanic_survive_push_source = PushSource(
    name="titanic_survive_push_source",
    batch_source=titanic_pca_source,
)

# Defines a slightly modified version of the feature view from above, where the source
# has been changed to the push source. This allows fresh features to be directly pushed
# to the online store for this feature view.
titanic_survive_fresh_fv = FeatureView(
    name="titanic_survive_fresh",
    entities=[passenger],
    schema=[
        Field(name="PC1", dtype=Float32),
        Field(name="PC2", dtype=Float32),
        Field(name="PC3", dtype=Float32),
        Field(name="PC4", dtype=Float32),
        Field(name="PC5", dtype=Float32),
    ],
    online=True,
    source=titanic_survive_push_source,  # Changed from above
)

titanic_survive_svc_v2 = FeatureService(
    name="titanic_survive_svc_v2",
    features=[titanic_survive_fresh_fv],
)

fs.refresh_registry()

fs.apply(
    [titanic_pca_source, passenger, titanic_survive_fv, titanic_survive_svc_v1, titanic_survive_push_source, titanic_survive_fresh_fv, titanic_survive_svc_v2],
    partial=False,
)"""