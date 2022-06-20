class NAColumnTransformer:
    def __init__(self, column):
        self.columns = column

    def transform(self, X, y=None):

        wo_form = list(set(self.columns) - {'formability'})
        X_new = X[wo_form].fillna("NA")
        X[wo_form] = X_new
        X['formability'] = X['formability'].fillna(0).astype(int)
        return X

    def fit(self, X, y=None):
        return self