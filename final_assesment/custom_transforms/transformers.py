import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer, FunctionTransformer


def get_categorical_pipeline():

    categorical_pipeline = Pipeline(steps=[
        ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    return categorical_pipeline


def get_numerical_pipeline(scale=False):
    if scale:
        numerical_pipeline = Pipeline(steps=[('log1p', FunctionTransformer(np.log1p))])
        return numerical_pipeline

    return 'passthrough'


def get_full_processeror(categorical_features, categorical_pipeline, numerical_features, numerical_pipeline):
    full_processor = ColumnTransformer(transformers=[
        ('category', categorical_pipeline, categorical_features),
        ('numerical', numerical_pipeline, numerical_features)
    ])
    return full_processor
