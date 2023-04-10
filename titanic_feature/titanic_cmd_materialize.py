from feast import FeatureStore
from datetime import datetime, timedelta

if __name__ == '__main__':
    fs = FeatureStore(repo_path="feature_repo")
    fs.materialize_incremental(end_date=datetime.utcnow() - timedelta(minutes=5))