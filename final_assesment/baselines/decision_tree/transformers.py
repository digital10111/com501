import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer, FunctionTransformer

from final_assesment.custom_transforms.NA_imputer import NAColumnTransformer
from final_assesment.custom_transforms.column_drop import ColumnDropperTransformer


def get_categorical_pipeline(cat_features_to_drop, na_imputer_cols):

    categorical_pipeline = Pipeline(steps=[
        ('drop_column', ColumnDropperTransformer(cat_features_to_drop)),
        ('na_imputer', NAColumnTransformer(na_imputer_cols)),
        ('mode', SimpleImputer(strategy='most_frequent')),
        ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    return categorical_pipeline


def get_drop_col_pipeline(drop_col_list):

    drop_col_pipeline = Pipeline(steps=[
        ('drop_column', ColumnDropperTransformer(drop_col_list))
    ])
    return drop_col_pipeline


def get_imputer_pipeline():

    imputer_pipeline = Pipeline(steps=[
        ('mode', SimpleImputer(strategy='most_frequent'))
    ])
    return imputer_pipeline


def get_numerical_pipeline(scale=False):
    if scale:
        numerical_pipeline = Pipeline(steps=[('log1p', FunctionTransformer(np.log1p))])
        return numerical_pipeline

    return 'passthrough'


def get_full_processeror(all_categorical_features, cat_features_to_drop, numerical_features, numerical_pipeline, na_imputer_drop):
    categorical_pipeline = get_categorical_pipeline(cat_features_to_drop, na_imputer_drop)
    full_processor = ColumnTransformer(transformers=[
        ('category', categorical_pipeline, all_categorical_features),
        ('numerical', numerical_pipeline, numerical_features)
    ])
    return full_processor


def get_full_processeror_rfsq(categorical_features, numerical_features, numerical_pipeline):
    drop_features_list = ['family', 'temper_rolling', 'non_ageing', 'surface_finish', 'enamelability', 'bc', 'bf', 'bt', 'bw_or_me', 'bl', 'm']
    drop_features_list.extend(['chrom', 'phos', 'cbond', 'marvi', 'exptl', 'ferro', 'corr', 'blue_bright_varn_clean', 'lustre', 'jurofm', 's', 'p', 'oil', 'packing'])
    categorical_pipeline = get_categorical_pipeline()
    drop_col_pipe = get_drop_col_pipeline(drop_features_list)
    full_processor = ColumnTransformer(transformers=[
        ('drop_col', drop_col_pipe, drop_features_list),
        ('category', categorical_pipeline, categorical_features),
        ('numerical', numerical_pipeline, numerical_features)
    ])
    return full_processor
