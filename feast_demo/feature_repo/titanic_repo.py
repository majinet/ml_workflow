# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, String

# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
passenger = Entity(name="passenger", join_keys=["PassengerId"])

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
titanic_pca_source = FileSource(
    name="titanic_pca_source",
    path="/mnt/c/Users/majin/PycharmProjects/ml_workflow/feast_demo/feature_repo/data/titanic_pca_feature.parquet",
    event_timestamp_column="event_timestamp",
)

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
titanic_survive_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="titanic_survive_fv",
    entities=[passenger],
    ttl=None,
    # The list of features defined below act as a schema to both define features
    # for both materialization of features into a store, and are used as references
    # during retrieval for building a training dataset or serving features
    schema=[
        Field(name="PC1", dtype=Float32),
        Field(name="PC2", dtype=Float32),
        Field(name="PC3", dtype=Float32),
        Field(name="PC4", dtype=Float32),
        Field(name="PC5", dtype=Float32),
    ],
    online=True,
    source=titanic_pca_source,
    # Tags are user defined key/value pairs that are attached to each
    # feature view
    tags={"team": "titanic_survive_view"},
)

# This groups features into a model version
titanic_survive_svc_v1 = FeatureService(
    name="titanic_survive_svc_v1",
    features=[
        titanic_survive_fv,  # Sub-selects a feature from a feature view
    ],
)

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
    ttl=None,
    schema=[
        Field(name="PC1", dtype=Float32),
        Field(name="PC2", dtype=Float32),
        Field(name="PC3", dtype=Float32),
        Field(name="PC4", dtype=Float32),
        Field(name="PC5", dtype=Float32),
    ],
    online=True,
    source=titanic_survive_push_source,  # Changed from above
    tags={"team": "titanic_survive_view"},
)

titanic_survive_svc_v2 = FeatureService(
    name="titanic_survive_svc_v2",
    features=[titanic_survive_fresh_fv],
)

