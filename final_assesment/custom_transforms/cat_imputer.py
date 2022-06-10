import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TargetSpecificCategoricalModeImputer(BaseEstimator, TransformerMixin):
    '''
    Class used for imputing missing values in a pd.DataFrame using either mean or median of a group.

    Parameters
    ----------
    group_cols : list
        List of columns used for calculating the aggregated value
    target : str
        The name of the column to impute
    metric : str
        The metric to be used for remplacement, can be one of ['mean', 'median']
    Returns
    -------
    X : array-like
        The array with imputed values in the target column
    '''

    def __init__(self):
        self.group_cols = ["formability", "condition", "surface_quality"]
        self.target = "target"
        self.modes = {}

    def fit(self, X, y=None):

        for col in self.group_cols:
            if col == "formability":
                self.modes["formability"] = X["formability"].mode().iloc[0]
                X['formability'] = X["formability"].fillna(self.modes["formability"])
            elif col == "condition":
                X.loc[X.target == 'U', 'condition'] = X.loc[X.target == 'U', 'condition'].fillna(X["condition"].mode().iloc[0])
                X['condition'] = X.groupby('target')['condition'].apply(lambda x: x.fillna(x.mode().iat[0]))
                self.modes['condition'] = X.groupby('target')['condition'].apply(lambda x: x.mode().iat[0]).to_dict()

            elif col == 'surface_quality':
                X.loc[X.target == '5', 'surface_quality'] = X.loc[X.target == '5', 'surface_quality'].fillna(X["surface_quality"].mode().iloc[0])
                X['surface_quality'] = X.groupby('target')['surface_quality'].apply(lambda x: x.fillna(x.mode().iat[0]))
                self.modes['surface_quality'] = X.groupby('target')['surface_quality'].apply(lambda x: x.mode().iat[0]).to_dict()
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col, mode in self.modes.items():
            if col == "formability":
                X['formability'] = X["formability"].fillna(self.modes["formability"])
            elif col in ["condition", "surface_quality"]:
                X.loc[X.target == '2', col] = X.loc[X.target == '2', col].fillna(self.modes[col]["2"])
                X.loc[X.target == '5', col] = X.loc[X.target == '5', col].fillna(self.modes[col]["5"])
                X.loc[X.target == 'U', col] = X.loc[X.target == 'U', col].fillna(self.modes[col]["U"])
        return X.values