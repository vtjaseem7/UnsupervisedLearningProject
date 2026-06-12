from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from src.preprocessing import (
    RemoveCustomerID,
    LogTransformer,
    MedianImputer
)
from src.feature_engineering import FeatureEngineer
from sklearn.cluster import KMeans

preprocessing_pipeline = Pipeline([
    ("remove_id",RemoveCustomerID()),
    ("imputer",MedianImputer()),
    ("feature_engineering",FeatureEngineer()),
    ("log_transform",LogTransformer()),
    ("scaler",StandardScaler())
])

pca_pipeline = Pipeline([
    ("preprocessing",preprocessing_pipeline),
    ("pca",PCA(n_components=0.95))
])


# Final Pipeline

customer_segmentation_pipeline = Pipeline([
    ("processed",pca_pipeline),
    ('clustering',KMeans(
        n_clusters=6,
        random_state=42,
        n_init=10,
        init="k-means++"
    ))
])