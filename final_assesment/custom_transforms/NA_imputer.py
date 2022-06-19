class NAColumnTransformer:
    def __init__(self, column):
        self.columns = column

    def transform(self, X, y=None):
        X_new = X[self.columns].fillna("NA")
        X[self.columns] = X_new
        return X

    def fit(self, X, y=None):
        return self