class ColumnDropperTransformer:
    def __init__(self, column):
        self.columns = column

    def transform(self, X, y=None):
        X_new = X.drop(self.columns, axis=1)
        return X_new

    def fit(self, X, y=None):
        return self