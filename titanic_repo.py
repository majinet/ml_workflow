# This is an example feature definition file

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
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, String

from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)

fs = FeatureStore(repo_path="/mnt/c/Users/majin/PycharmProjects/ml_workflow/feast_demo/feature_repo")

# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
passenger = Entity(name="passenger", join_keys=["PassengerId"])

titanic_pca_source = PostgreSQLSource(
    name="titanic_pca_feature",
    query="SELECT * FROM titanic_pca_feature",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
titanic_survive_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="titanic_survive_fv",
    entities=[passenger],
    schema=[
        Field(name="PC1", dtype=Float32),
        Field(name="PC2", dtype=Float32),
        Field(name="PC3", dtype=Float32),
        Field(name="PC4", dtype=Float32),
        Field(name="PC5", dtype=Float32),
    ],
    online=True,
    source=titanic_pca_source,
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

fs.apply([titanic_pca_source, passenger, titanic_survive_fv, titanic_survive_svc_v1, titanic_survive_push_source, titanic_survive_fresh_fv, titanic_survive_svc_v2],
         partial=False,
         )