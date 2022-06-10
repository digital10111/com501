class ColumnDropperTransformer:
    def __init__(self, column):
        self.columns = column

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self